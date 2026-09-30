"""Merged backward correction: the coupled SVD estimator with damped in-place updates.

This trainer merges the two descriptions of the backward correction:

* the **estimator** is the coupled compact SVD of the backpropagation
  derivation, i.e. the singular pairs
  of ``B*_ℓ = E[y g_{ℓ+1} (h_ℓ*)ᵀ]`` installed as a rank-``r`` append
  ``W̄^{ℓ+1} = R^{ℓ+1}(V^{ℓ+1})ᵀ + √p_{ℓ+1} U^ℓ (Q^ℓ)ᵀ`` with the analytic
  row factor ``q_i = R^ℓ v_i`` normalised to one forward coordinate;
* the **schedule** is a damped, interleaved, in-place sweep, i.e. the
  ordinary filters ``V^ℓ`` and the paired
  factors ``(U^ℓ, Q^ℓ)`` are both moved by a step ``α`` toward their freshly
  computed targets, bottom-up, with the whole network recomputed after every
  update and the readout refitted between passes.

The initialisation is one top-down pass of the column-coupled algorithm
(:class:`~.backward_coupled.CoupledBackwardTrainer` with the mean-field
chain), so "pass 0" of this trainer is exactly that algorithm and serves as
the reference rung of the ladder.

Scope: flat FFN blocks with dense lifts and scalar labels, as for the coupled
trainer.  Features are materialised in full (chunked computation, full
storage), so memory is ``O(L n p)``.

Variants (all default to the behaviour above):

* ``chain="gate"`` replaces the mean-field backward signal everywhere it is
  used -- the estimator ``B*``, the row scores and the ridge scores -- by the
  block's own derivative feature ``σ'(h)/√p`` (no propagation, boundaries
  decouple);
* ``q_mode="ridge"`` keeps ``U`` from the SVD of ``B*`` but takes ``Q`` from
  the joint label ridge of the paper draft (``backward_ridge``), in pass 0
  and in the sweeps, as a proposal interpolated with ``α``; the fit sets the
  directions and relative channel amplitudes and one common factor sets the
  block's scale to the forward-coordinate RMS; ``ridge_dictionary`` picks the
  regression basis
  (``"union"``: ``[R | √p U^{ℓ-1} | M]``, ``"alg1"``: ``[R | M]``).
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..models.spectral import SpectralModel, _Block, _Reduce
from ..utils.transforms import DenseTransform
from .backward_coupled import (
    CoupledBackwardConfig,
    CoupledBackwardTrainer,
    boundary_svd,
)
from .backward_lofi import block_derivative_features, orthonormal_union
from .backward_ridge import DEFAULT_LAMBDAS, build_dictionary, fit_paired_rows
from .eigen.prepend import linear_mean_prepend_block
from .spectral import SpectralTrainer, SpectralTrainerConfig

log = logging.getLogger(__name__)

__all__ = ["MergedBackwardConfig", "MergedBackwardTrainer"]

EPS = 1e-12


@dataclass
class MergedBackwardConfig(SpectralTrainerConfig):
    """Configuration for :class:`MergedBackwardTrainer`.

    Parameters
    ----------
    backward_fraction, backward_rank
        Rank of the paired block per boundary, as a fraction of the upper
        block's forward width or as an absolute rank.
    n_passes
        Number of bottom-up sweeps after the top-down initialisation.
    alpha
        Damping of both the row and the paired-factor updates:
        ``V ← (1-α)V + αV_c`` and likewise for ``(U, Q)``.
    n_streams, stream_k
        Modified-label streams used by the row update: ``m = min(n_streams, r)``
        filters of ``stream_k`` directions each, carved out of the block's own
        width budget.
    stream_linear_mean
        Prepend the normalised first moment to each modified-label filter.
    chunk_size
        Sample chunk used when recomputing features and gates.
    chain
        ``"mean_field"`` (the recursion with the random branch averaged) or
        ``"gate"`` (the local derivative feature, no propagation).
    q_mode
        ``"svd"``: ``Q`` from the lifted right singular vectors with the
        forward-RMS convention.  ``"ridge"``: ``Q`` from the joint label ridge
        of the paper draft, used wherever a pair is computed.
    ridge_dictionary, ridge_lambdas, ridge_holdout, ridge_select_n,
    ridge_cg_iters, ridge_cg_tol
        Basis and numerics of the ridge; see :mod:`.backward_ridge`.
    """

    backward_fraction: float = 0.25
    backward_rank: int | None = None
    n_passes: int = 3
    alpha: float = 0.30
    n_streams: int = 3
    stream_k: int = 16
    stream_linear_mean: bool = True
    chunk_size: int = 8192
    chain: str = "mean_field"
    q_mode: str = "svd"
    ridge_dictionary: str = "union"
    ridge_lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    ridge_holdout: float = 0.10
    ridge_select_n: int | None = 10_000
    ridge_cg_iters: int = 240
    ridge_cg_tol: float = 1e-4


_CHAINS = ("mean_field", "gate")
_Q_MODES = ("svd", "ridge")
_DICTIONARIES = ("union", "alg1")


def _rms(t: Tensor) -> Tensor:
    return t.pow(2).mean().clamp_min(EPS**2).sqrt()


def _orth_factor(m: Tensor, fallback: Tensor) -> Tensor:
    """``A Bᵀ`` from the SVD of ``m``, robust to a rank-deficient input.

    The divide-and-conquer driver can fail to converge on blocks that carry
    exact zero or repeated directions, which happens when a filter union is
    short of its budget.  Both uses of this factor (Procrustes alignment and
    the polar step) are gauge choices, so a failure falls back to ``fallback``
    rather than aborting the fit.
    """
    try:
        a, _s, bh = torch.linalg.svd(m.double(), full_matrices=False)
    except Exception:  # torch._C._LinAlgError and any driver-level failure
        log.warning("orthogonal factor: SVD did not converge; keeping the gauge.")
        return fallback
    return (a @ bh).to(m.dtype)


def _polar(m: Tensor) -> Tensor:
    """Closest matrix with orthonormal columns (polar factor of ``m``)."""
    fallback = torch.linalg.qr(m.double())[0].to(m.dtype)
    return _orth_factor(m, fallback)


class MergedBackwardTrainer(SpectralTrainer):
    """Damped interleaved row/column correction of a fitted forward model.

    ``fit(loader, test_loader=..., pass_callback=...)`` returns the corrected
    model and the results of the final pass.  ``pass_callback(idx, results)``
    is called after the initialisation (``idx=0``) and after every pass.
    """

    def __init__(self, model: SpectralModel, config: MergedBackwardConfig) -> None:
        if config.backward_rank is None and not 0.0 < config.backward_fraction <= 1.0:
            raise ValueError(
                f"backward_fraction must be in (0, 1], got {config.backward_fraction}"
            )
        if config.n_passes < 0:
            raise ValueError(f"n_passes must be >= 0, got {config.n_passes}")
        if not 0.0 < config.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {config.alpha}")
        if config.chain not in _CHAINS:
            raise ValueError(f"chain must be one of {_CHAINS}, got {config.chain!r}")
        if config.q_mode not in _Q_MODES:
            raise ValueError(f"q_mode must be one of {_Q_MODES}, got {config.q_mode!r}")
        if config.ridge_dictionary not in _DICTIONARIES:
            raise ValueError(
                f"ridge_dictionary must be one of {_DICTIONARIES}, "
                f"got {config.ridge_dictionary!r}"
            )
        super().__init__(model, config)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        test_loader = cast("DataLoader | None", kwargs.get("test_loader"))
        pass_callback = cast("Any", kwargs.get("pass_callback"))
        source: SpectralModel = self.model  # type: ignore[assignment]
        config: MergedBackwardConfig = self.config  # type: ignore[assignment]
        dev = torch.device(config.device)

        self._is_vector, self._n_outputs, self._is_one_hot = self._peek_label_shape(
            loader
        )
        if self._is_vector:
            raise NotImplementedError("MergedBackwardTrainer supports scalar labels.")
        for i, block in enumerate(source.blocks):
            b = cast(_Block, block)
            if cast(_Reduce, b.reduce).is_conv or b.pool is not None:
                raise NotImplementedError(f"Block {i} is convolutional; FFN only.")
            if not isinstance(b.transform, DenseTransform):
                raise NotImplementedError(f"Block {i} needs a dense lift.")
            if cast(_Reduce, b.reduce).is_identity:
                raise NotImplementedError(
                    f"Block {i} has an identity reduce; the merged correction "
                    "updates every block's filter, so each block needs one."
                )

        # Phase-I ordinary widths and filters (the Procrustes anchors).
        n_blocks = len(source.blocks)
        self._ord_k = [
            int(cast(_Reduce, cast(_Block, b).reduce).V.shape[1]) for b in source.blocks
        ]
        v0 = [cast(_Reduce, cast(_Block, b).reduce).V.clone() for b in source.blocks]

        # --- pass 0: top-down column-coupled placement (mean-field chain) ---
        init_cfg = CoupledBackwardConfig(
            device=config.device,
            verbose=False,
            alpha_min=config.alpha_min,
            alpha_max=config.alpha_max,
            alpha_num=config.alpha_num,
            backward_fraction=config.backward_fraction,
            backward_rank=config.backward_rank,
            combine_mode="add",
            chain=config.chain,
        )
        corrected, results = CoupledBackwardTrainer(source, init_cfg).fit(
            loader, test_loader=test_loader
        )
        corrected = cast(SpectralModel, corrected)

        # Paired rank actually installed at each boundary.
        self._r = [
            int(cast(_Reduce, cast(_Block, corrected.blocks[j + 1]).reduce).V.shape[1])
            - self._ord_k[j + 1]
            for j in range(n_blocks - 1)
        ]
        self.model = corrected
        X, y = self._materialise(loader, dev)

        if config.q_mode == "ridge":
            # The coupled placement only allocated the slots: refill them
            # top-down with (SVD U, ridge Q̂) on the Phase-I readout.
            results = self._ridge_pass0(corrected, source, X, y, loader, test_loader)
        results["backward"]["algorithm"] = "merged"
        results["backward"]["pass"] = 0
        results["backward"]["alpha"] = config.alpha
        results["backward"]["chain"] = config.chain
        results["backward"]["q_mode"] = config.q_mode
        results["backward"]["ridge_dictionary"] = config.ridge_dictionary
        if pass_callback is not None:
            pass_callback(0, copy.deepcopy(results))

        for p in range(1, config.n_passes + 1):
            entries = self._one_pass(corrected, X, y, v0, p)
            final = self._fit_blocked_readout(loader, test_loader)
            results = {
                "layers": entries,
                "final": final,
                "backward": {
                    "algorithm": "merged",
                    "pass": p,
                    "alpha": config.alpha,
                    "n_passes": config.n_passes,
                    "backward_fraction": config.backward_fraction,
                    "backward_rank": config.backward_rank,
                    "n_streams": config.n_streams,
                    "stream_k": config.stream_k,
                    "ranks": self._r,
                    "ord_k": self._ord_k,
                    "chain": config.chain,
                    "q_mode": config.q_mode,
                    "ridge_dictionary": config.ridge_dictionary,
                },
            }
            if pass_callback is not None:
                pass_callback(p, copy.deepcopy(results))
        return corrected, results

    # ------------------------------------------------------------------ #
    # One bottom-up pass
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _one_pass(
        self,
        model: SpectralModel,
        X: Tensor,
        y: Tensor,
        v0: list[Tensor],
        pass_idx: int,
    ) -> list[dict[str, Any]]:
        n_blocks = len(model.blocks)
        entries: list[dict[str, Any]] = []
        feats, gates = self._state(model, X)
        for j in range(n_blocks - 1):
            # (1) scores of boundary j from the current network
            g_up = self._gbar(model, X, feats, gates, stop=j + 1)
            u_old = self._installed_u(model, j)
            scores = g_up @ u_old  # (n, r_j)
            # (2) row update of block j, then refresh
            row = self._row_update(model, j, X, feats, y, scores, v0[j])
            feats, gates = self._state(model, X)
            # (3) column update of boundary j, then refresh
            col = self._column_update(model, j, X, feats, gates, y)
            feats, gates = self._state(model, X)
            entries.append({"boundary": j, "pass": pass_idx, **row, **col})
        # top row update with the plain label
        top = n_blocks - 1
        row = self._row_update(model, top, X, feats, y, None, v0[top])
        entries.append({"boundary": top, "pass": pass_idx, "top_row": True, **row})
        return entries

    # ------------------------------------------------------------------ #
    # Pieces
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _materialise(
        self, loader: DataLoader, dev: torch.device
    ) -> tuple[Tensor, Tensor]:
        xs, ys = [], []
        for x, yb in loader:
            xs.append(x.to(dev, non_blocking=True))
            yb = yb.to(dev, non_blocking=True).to(torch.float32)
            ys.append(yb if yb.ndim == 1 else yb.reshape(-1))
        return torch.cat(xs), torch.cat(ys)

    @torch.no_grad()
    def _state(
        self, model: SpectralModel, X: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Per-block outputs ``a_j`` and gates ``σ'(h_j)/√p``, computed in chunks."""
        chunk = int(self.config.chunk_size)  # type: ignore[attr-defined]
        outs: list[list[Tensor]] = [[] for _ in model.blocks]
        gts: list[list[Tensor]] = [[] for _ in model.blocks]
        for xc in X.split(chunk):
            a = xc
            for j, blk in enumerate(model.blocks):
                b = cast(_Block, blk)
                gts[j].append(block_derivative_features(b, a))
                a = b(a)
                outs[j].append(a)
        return [torch.cat(o) for o in outs], [torch.cat(g) for g in gts]

    @torch.no_grad()
    def _readout_grad(self, model: SpectralModel) -> Tensor:
        w = model.readout_weight.detach()
        fr = cast(_Reduce, model.final_reduction)
        if fr.is_identity:
            return w
        wk = w / fr.scale if fr.whiten else w
        return fr.V @ wk

    @torch.no_grad()
    def _gbar(
        self,
        model: SpectralModel,
        X: Tensor,
        feats: list[Tensor],
        gates: list[Tensor],
        stop: int,
    ) -> Tensor:
        """Backward signal at block ``stop``: mean-field recursion, or the gate."""
        if self.config.chain == "gate":  # type: ignore[attr-defined]
            # Local signal: the block's own derivative feature, no propagation.
            return gates[stop]
        top = len(model.blocks) - 1
        g = gates[top] * self._readout_grad(model).unsqueeze(0)
        for m in range(top - 1, stop - 1, -1):
            blk = cast(_Block, model.blocks[m + 1])
            red = cast(_Reduce, blk.reduce)
            t = g @ cast(Tensor, blk.weight).transpose(0, 1) / blk.c
            if red.whiten:
                t = t / red.scale
            keep = self._ord_k[m + 1]
            rand = t[:, :keep] @ red.V[:, :keep].transpose(0, 1)
            r_prev = rand.mean(dim=0, keepdim=True).expand_as(rand)
            if red.V.shape[1] > keep:
                r_prev = r_prev + t[:, keep:] @ red.V[:, keep:].transpose(0, 1)
            g = gates[m] * r_prev
        return g

    @torch.no_grad()
    def _installed_u(self, model: SpectralModel, j: int) -> Tensor:
        """``U^j`` recovered from the lift rows of block ``j+1``."""
        blk = cast(_Block, model.blocks[j + 1])
        keep = self._ord_k[j + 1]
        p_up = int(cast(Tensor, blk.weight).shape[1])
        return cast(Tensor, blk.weight)[keep:].transpose(0, 1) / math.sqrt(p_up)

    @torch.no_grad()
    def _filter(self, z: Tensor, xi: Tensor, k: int) -> Tensor:
        """``Fwd(ξ, z, k)``: normalised first moment plus top eigenvectors."""
        _vals, v = linear_mean_prepend_block(
            z,
            xi,
            k=max(k - 1, 1),
            include_linear_mean=bool(self.config.stream_linear_mean),  # type: ignore[attr-defined]
            orthogonalize=True,
        )
        return v[:, :k]

    @torch.no_grad()
    def _row_update(
        self,
        model: SpectralModel,
        j: int,
        X: Tensor,
        feats: list[Tensor],
        y: Tensor,
        scores: Tensor | None,
        v_anchor: Tensor,
    ) -> dict[str, Any]:
        config: MergedBackwardConfig = self.config  # type: ignore[assignment]
        blk = cast(_Block, model.blocks[j])
        red = cast(_Reduce, blk.reduce)
        k = self._ord_k[j]
        z_in = X if j == 0 else feats[j - 1]
        d = int(config.stream_k)
        m = 0 if scores is None else min(int(config.n_streams), scores.shape[1])
        while m > 0 and d * m >= k:
            m -= 1
        base_k = k - d * m
        f_all = self._filter(z_in, y, k)
        cols: list[Tensor] = [f_all[:, :base_k]]
        for i in range(m):
            s_i = cast(Tensor, scores)[:, i]
            xi = y * s_i / _rms(s_i)
            cols.append(self._filter(z_in, xi, d))
        if base_k < k:
            cols.append(f_all[:, base_k:])
        v_new, _kept = orthonormal_union(cols, k)
        # ``orthonormal_union`` zero-pads when the candidate span is short of
        # the budget (modified directions dependent on the ordinary ones).  A
        # zero column would shrink the filter and leaves the alignment problem
        # rank deficient, so backfill it with the Phase-I direction.
        dead = v_new.norm(dim=0) < 0.5
        n_dead = int(dead.sum().item())
        if n_dead:
            v_new = v_new.clone()
            v_new[:, dead] = v_anchor[:, dead]
        # Procrustes alignment to the Phase-I filter keeps the frozen lift's
        # neuron coordinates meaningful.
        rot = _orth_factor(
            v_new.transpose(0, 1) @ v_anchor,
            torch.eye(k, dtype=v_new.dtype, device=v_new.device),
        )
        v_c = v_new @ rot
        alpha = float(config.alpha)
        red.V[:, :k] = (1.0 - alpha) * red.V[:, :k] + alpha * v_c
        drift = float((v_c - v_anchor).norm() / max(float(v_anchor.norm()), EPS))
        return {
            "row_streams": m,
            "row_base_k": base_k,
            "row_drift": drift,
            "row_backfilled": n_dead,
        }

    @torch.no_grad()
    def _column_update(
        self,
        model: SpectralModel,
        j: int,
        X: Tensor,
        feats: list[Tensor],
        gates: list[Tensor],
        y: Tensor,
    ) -> dict[str, Any]:
        config: MergedBackwardConfig = self.config  # type: ignore[assignment]
        blk = cast(_Block, model.blocks[j])
        blk_up = cast(_Block, model.blocks[j + 1])
        red_up = cast(_Reduce, blk_up.reduce)
        keep = self._ord_k[j + 1]
        r = self._r[j]
        n = X.shape[0]
        p_up = int(cast(Tensor, blk_up.weight).shape[1])

        z_in = X if j == 0 else feats[j - 1]
        h_star = blk.expand_input(z_in)
        g_up = self._gbar(model, X, feats, gates, stop=j + 1)
        b_hat = (g_up.transpose(0, 1).double() @ (y.unsqueeze(1) * h_star).double()) / n
        u, s, v = boundary_svd(b_hat)
        r_eff = min(r, int((s > 0).sum().item()) or 1)
        u_new, v_new = u[:, :r_eff], v[:, :r_eff]

        # Feature-matched scale: one backward coordinate = one forward coordinate.
        fwd = feats[j] @ red_up.V[:, :keep]
        rms_fwd = fwd.pow(2).mean(dim=0).mean().sqrt().double()

        dtype = red_up.V.dtype
        u_old = self._installed_u(model, j).double()
        # Gauge fixing: U Qᵀ is invariant under a common r×r rotation.
        rot = _orth_factor(
            u_new.transpose(0, 1) @ u_old,
            torch.eye(r_eff, dtype=u_new.dtype, device=u_new.device),
        )
        u_new = u_new @ rot
        alpha = float(config.alpha)
        u_mix = _polar((1.0 - alpha) * u_old + alpha * u_new)
        w = cast(Tensor, blk_up.weight)
        entry: dict[str, Any] = {
            "rank": r_eff,
            "kept_forward": keep,
            "singular_values": [float(x) for x in s[:r_eff].tolist()],
            "u_alignment": float((u_new * u_old).sum(dim=0).abs().mean().item()),
            "coord_rms_forward": float(rms_fwd),
        }

        if config.q_mode == "ridge":
            # The draft's column half: Q̂ = P C* fitted against the scores of
            # the directions being installed; a proposal interpolated with α,
            # its scale left as fitted (no RMS convention, no trust radius).
            r_inst = int(u_mix.shape[1])
            scores = (g_up.double() @ u_mix).to(feats[j].dtype)
            q_hat, ridge_info = self._ridge_q(model, j, feats, y, scores, rms_fwd)
            q_old = red_up.V[:, keep : keep + r_inst].double()
            q_mix = (1.0 - alpha) * q_old + alpha * q_hat.double()
            q_mix, _f = self._block_scale(feats[j], q_mix, rms_fwd)
            red_up.V[:, keep : keep + r_inst] = q_mix.to(dtype)
            w[keep : keep + r_inst] = (
                (math.sqrt(p_up) * u_mix).transpose(0, 1).to(w.dtype)
            )
            coords = feats[j].double() @ q_mix
            entry["ridge"] = ridge_info
            entry["coord_rms_backward"] = [
                float(x) for x in coords.pow(2).mean(dim=0).sqrt().tolist()
            ]
            return entry

        q_new = cast(Tensor, blk.weight).double().transpose(0, 1) @ v_new
        coords = feats[j].double() @ q_new
        q_new = q_new * (rms_fwd / coords.pow(2).mean(dim=0).clamp_min(EPS).sqrt())
        q_new = q_new @ rot
        q_old = red_up.V[:, keep : keep + r_eff].double()
        q_mix = (1.0 - alpha) * q_old + alpha * q_new
        coords = feats[j].double() @ q_mix
        q_mix = q_mix * (rms_fwd / coords.pow(2).mean(dim=0).clamp_min(EPS).sqrt())

        red_up.V[:, keep : keep + r_eff] = q_mix.to(dtype)
        w[keep : keep + r_eff] = (math.sqrt(p_up) * u_mix).transpose(0, 1).to(w.dtype)
        return entry

    # ------------------------------------------------------------------ #
    # Ridge column half (Algorithm 1's fit, used as a proposal)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _ridge_q(
        self,
        model: SpectralModel,
        j: int,
        feats: list[Tensor],
        y: Tensor,
        scores: Tensor,
        rms_fwd: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """``Q̂ = P C*`` (p_j, r) for boundary ``j`` from the draft's label ridge.

        The dictionary is ``[R^j | √p U^{j-1} | M]`` (``"union"``) or
        ``[R^j | M]`` (``"alg1"``) with ``M = E[z_j (y s)ᵀ]``; the block's
        lift rows are ``[R^j | √p U^{j-1}]`` and ``√p U^{j-1}`` is empty or
        zero until the boundary below has been placed.

        The fit sets the directions and the *relative* amplitudes of the
        ``r`` channels; their overall scale is a linearisation artefact (the
        scores are small, so the fitted coordinates come out an order of
        magnitude above the forward ones and the block above saturates).  The
        block is therefore scaled by one common factor so that the mean
        channel RMS equals ``rms_fwd``, the SVD arm's convention applied to
        the block instead of per channel.
        """
        config: MergedBackwardConfig = self.config  # type: ignore[assignment]
        blk = cast(_Block, model.blocks[j])
        lift = cast(Tensor, blk.weight).transpose(0, 1)  # (p_j, k_j + r_{j-1})
        k_j = self._ord_k[j]
        z = feats[j]
        m = z.transpose(0, 1) @ (y.unsqueeze(1) * scores) / z.shape[0]
        parts = [lift[:, :k_j]]
        if config.ridge_dictionary == "union":
            parts.append(lift[:, k_j:])
        parts.append(m)
        dictionary = build_dictionary(torch.cat(parts, dim=1))
        q_hat, info = fit_paired_rows(
            z,
            y,
            scores,
            dictionary,
            lambdas=tuple(config.ridge_lambdas),
            holdout=float(config.ridge_holdout),
            select_n=config.ridge_select_n,
            cg_iters=int(config.ridge_cg_iters),
            cg_tol=float(config.ridge_cg_tol),
            seed=j,
        )
        q_hat, factor = self._block_scale(z, q_hat, rms_fwd)
        info["scale_factor"] = factor
        return q_hat, info

    @staticmethod
    def _block_scale(z: Tensor, q: Tensor, rms_fwd: Tensor) -> tuple[Tensor, float]:
        """Scale all columns of ``q`` by one factor: mean channel RMS = ``rms_fwd``."""
        rms = (z.double() @ q.double()).pow(2).mean(dim=0).sqrt()
        factor = float(rms_fwd.double() / rms.mean().clamp_min(EPS))
        return q * factor, factor

    @torch.no_grad()
    def _ridge_pass0(
        self,
        model: SpectralModel,
        source: SpectralModel,
        X: Tensor,
        y: Tensor,
        loader: DataLoader,
        test_loader: DataLoader | None,
    ) -> dict[str, Any]:
        """Pass 0 with the ridge: Algorithm 2 lines 43--48 on allocated slots.

        The coupled placement widened every upper block; here all paired slots
        are zeroed, the Phase-I readout is restored, and the boundaries are
        filled top-down with ``U`` from the SVD of ``B*`` and ``Q`` from the
        ridge, refreshing the network after each.  The readout is refitted
        once at the end.
        """
        model.readout_weight = nn.Parameter(
            source.readout_weight.detach().clone(), requires_grad=False
        )
        model.readout_bias = nn.Parameter(
            source.readout_bias.detach().clone(), requires_grad=False
        )
        n_blocks = len(model.blocks)
        n = X.shape[0]
        for j in range(n_blocks - 1):
            blk_up = cast(_Block, model.blocks[j + 1])
            keep, r = self._ord_k[j + 1], self._r[j]
            cast(_Reduce, blk_up.reduce).V[:, keep : keep + r] = 0.0
            cast(Tensor, blk_up.weight)[keep : keep + r] = 0.0
        entries: list[dict[str, Any]] = []
        feats, gates = self._state(model, X)
        for j in range(n_blocks - 2, -1, -1):
            blk = cast(_Block, model.blocks[j])
            blk_up = cast(_Block, model.blocks[j + 1])
            red_up = cast(_Reduce, blk_up.reduce)
            keep, r = self._ord_k[j + 1], self._r[j]
            w = cast(Tensor, blk_up.weight)
            p_up = int(w.shape[1])
            z_in = X if j == 0 else feats[j - 1]
            h_star = blk.expand_input(z_in)
            g_up = self._gbar(model, X, feats, gates, stop=j + 1)
            b_hat = (
                g_up.transpose(0, 1).double() @ (y.unsqueeze(1) * h_star).double()
            ) / n
            u, s, _v = boundary_svd(b_hat)
            r_eff = min(r, int((s > 0).sum().item()))
            u_r = torch.zeros(p_up, r, dtype=u.dtype, device=u.device)
            u_r[:, :r_eff] = u[:, :r_eff]
            w[keep : keep + r] = (math.sqrt(p_up) * u_r).transpose(0, 1).to(w.dtype)
            scores = (g_up.double() @ u_r).to(feats[j].dtype)
            fwd = feats[j] @ red_up.V[:, :keep]
            rms_fwd = fwd.pow(2).mean(dim=0).mean().sqrt().double()
            q_hat, info = self._ridge_q(model, j, feats, y, scores, rms_fwd)
            red_up.V[:, keep : keep + r] = q_hat.to(red_up.V.dtype)
            feats, gates = self._state(model, X)
            entries.append(
                {
                    "boundary": j,
                    "block_idx": j + 1,
                    "pass": 0,
                    "rank": r_eff,
                    "kept_forward": keep,
                    "singular_values": [float(x) for x in s[:r_eff].tolist()],
                    "ridge": info,
                }
            )
        final = self._fit_blocked_readout(loader, test_loader)
        return {
            "layers": entries,
            "final": final,
            "backward": {"ranks": self._r, "ord_k": self._ord_k},
        }
