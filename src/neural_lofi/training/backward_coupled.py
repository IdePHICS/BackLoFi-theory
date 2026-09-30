"""Column-coupled backward Neural LoFi (Algorithm 2 of the backward note).

The row-side backward correction of :mod:`.backward_lofi` consumes the
boundary SVD only through its left singular vectors (as effective labels
for a forward-LoFi *row* step) and redraws the lift, which discards the
pairing ``u_i ↔ v_i`` that the gradient step \\(\\Delta W^{\\ell+1} =
\\sum_i \\hat\\sigma_i u_i (R^\\ell v_i)^\\top\\) actually carries.  This trainer
keeps the pairing: at every boundary the backward information enters the
network **only through the columns of the layer above**,

    W̄_{ℓ+1} = R_{ℓ+1} V_{ℓ+1}ᵀ + √p_{ℓ+1} · U_← Q_←ᵀ,
    q_i = R_ℓ v_i / ν_i   (a pattern over the neurons of layer ℓ),

i.e. the reduce feeding block ``ℓ+1`` gains the columns ``q_i`` and its lift
gains the *deterministic* columns ``√p·u_i``.  Nothing is added to the rows,
nothing is orthogonalised, nothing is redrawn: the corrected model is the
frozen Phase-I model plus the coupled columns, and only the ridge readout is
refit at the end.

The sweep runs **top-down** with the backward feature propagated by the
chain rule through the already-corrected matrices (``chain="exact"``), or
with the random part of that propagation replaced by its per-neuron sample
mean (``chain="mean_field"``, the LoFi-analysable form).  The top of the
chain is the exact boundary condition ``g_L = a ⊙ σ'(h_L)`` with the Phase-I
ridge readout ``a``.  Two stripped variants drop the propagation entirely
(``chain="gate"``: the local derivative feature ``σ'(h)/√p``;
``chain="activation"``: the activation ``σ(h)/√p``) — ablations that test
whether the top-down signal matters.

Scope (deliberately minimal): flat FFN blocks with dense lifts and scalar
labels.  Conv blocks, SORF lifts and vector labels raise
``NotImplementedError``.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..models.block_config import BlockedModelConfig
from ..models.spectral import SpectralModel, _Block, _Reduce
from ..utils.helpers import ACTIVATION_DERIVATIVES
from ..utils.transforms import DenseTransform
from .backward_lofi import block_derivative_features
from .spectral import SpectralTrainer, SpectralTrainerConfig

log = logging.getLogger(__name__)

__all__ = [
    "CoupledBackwardConfig",
    "CoupledBackwardTrainer",
    "boundary_svd",
]

_COMBINE_MODES = ("swap", "add")
_CHAINS = ("exact", "mean_field", "gate", "activation")
_RIGHTS = ("reduced", "output")
_GATE_FORMS = ("exact", "linear")


@torch.no_grad()
def boundary_svd(b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Thin SVD ``b = U diag(s) Vᵀ`` of an ``m × n`` boundary matrix.

    Returns ``(U (m, q), s (q,), V (n, q))`` with ``q = min(m, n)``, singular
    values descending.  Wide matrices (``n`` beyond ``max(4m, 4096)``) go
    through the ``m × m`` Gram (cuSOLVER rejects very wide thin SVDs); the
    right vectors are then recovered as ``V = bᵀ U / s`` on the nonzero
    spectrum.
    """
    m, n = b.shape
    if n <= max(4 * m, 4096):
        u, s, vh = torch.linalg.svd(b, full_matrices=False)
        return u, s, vh.transpose(0, 1)
    evals, evecs = torch.linalg.eigh(b @ b.transpose(0, 1))
    order = torch.argsort(evals, descending=True)
    u = evecs[:, order]
    s = evals[order].clamp_min(0).sqrt()
    pos = s > 0
    v = torch.zeros(n, u.shape[1], dtype=b.dtype, device=b.device)
    v[:, pos] = (b.transpose(0, 1) @ u[:, pos]) / s[pos]
    return u, s, v


@dataclass
class CoupledBackwardConfig(SpectralTrainerConfig):
    """Configuration for :class:`CoupledBackwardTrainer`.

    Parameters
    ----------
    backward_fraction
        Number of coupled backward features per boundary as a fraction of the
        corrected reduce's forward width: ``r = round(f · k)``.
    backward_rank
        Absolute ``r`` per boundary; overrides ``backward_fraction`` when set.
    combine_mode
        ``"swap"`` drops the ``r`` trailing forward columns (and their lift
        columns) before appending the coupled ones, keeping the rank ``k``;
        ``"add"`` grows the rank to ``k + r``.
    chain
        ``"exact"``: backward feature propagated by the chain rule through the
        corrected matrices.  ``"mean_field"``: the random part
        ``V (R)ᵀ g`` of that propagation is replaced by its per-neuron sample
        mean; the coupled part stays exact.  ``"gate"`` / ``"activation"``:
        stripped local variants with **no propagation** — the boundary signal
        of layer ``ℓ+1`` is its own gate ``σ'(h)/√p`` (Algorithm 1's
        derivative feature) or its activation ``σ(h)/√p`` (raw, uncentred);
        boundaries are then independent and the readout enters only through
        the final refit.
    estimator_right
        Right factor of the boundary matrix.  ``"reduced"``: the reduced
        input ``h*`` of the lower block (``B = E[y g h*ᵀ]``, ``p × k``; the
        right singular vectors are lifted to neuron patterns ``q = R v``).
        ``"output"``: the wide output ``z`` of the lower block
        (``B = E[y g zᵀ]``, ``p × p``, the unrestricted gradient step); the
        right singular vectors are the neuron patterns themselves.
    gate_form
        ``"exact"``: the gate ``σ'(h)/√p`` wherever it enters the backward
        signal.  ``"linear"``: its first-order expansion ``(c̃ + d̃ h)/√p``
        with one pair ``c̃ = E[σ'(h)]``, ``d̃ = E[h σ'(h)]`` per block
        (Hermite coefficients under the unit-variance preactivation law).
    """

    backward_fraction: float = 0.25
    backward_rank: int | None = None
    combine_mode: str = "swap"
    chain: str = "exact"
    estimator_right: str = "reduced"
    gate_form: str = "exact"


@dataclass
class _Correction:
    """What the chain needs from an already-corrected boundary."""

    keep: int  # retained forward columns (the coupled ones follow)
    rank: int
    a_bar: Tensor | None = None  # (p_j,) mean random projection (mean_field)
    entry: dict[str, Any] = field(default_factory=dict)


class CoupledBackwardTrainer(SpectralTrainer):
    """Top-down column-coupled backward correction of a fitted model.

    ``model`` must be a :class:`SpectralModel` already fitted by
    :class:`SpectralTrainer` (its Phase-I reductions, lifts and ridge readout
    define both the forward features and the top of the backward chain).
    ``fit`` deep-copies it, corrects every boundary from the top down, refits
    the readout on the corrected features and returns
    ``(corrected_model, {"layers": [...], "final": {...}, "backward": {...}})``.
    The input model is never modified.
    """

    def __init__(self, model: SpectralModel, config: CoupledBackwardConfig) -> None:
        if config.combine_mode not in _COMBINE_MODES:
            raise ValueError(
                f"combine_mode must be one of {_COMBINE_MODES}, "
                f"got {config.combine_mode!r}"
            )
        if config.chain not in _CHAINS:
            raise ValueError(f"chain must be one of {_CHAINS}, got {config.chain!r}")
        if config.estimator_right not in _RIGHTS:
            raise ValueError(
                f"estimator_right must be one of {_RIGHTS}, "
                f"got {config.estimator_right!r}"
            )
        if config.gate_form not in _GATE_FORMS:
            raise ValueError(
                f"gate_form must be one of {_GATE_FORMS}, got {config.gate_form!r}"
            )
        if config.backward_rank is None and not 0.0 < config.backward_fraction <= 1.0:
            raise ValueError(
                f"backward_fraction must be in (0, 1], got {config.backward_fraction}"
            )
        if config.backward_rank is not None and config.backward_rank < 1:
            raise ValueError(f"backward_rank must be >= 1, got {config.backward_rank}")
        super().__init__(model, config)
        self._corrections: dict[int, _Correction] = {}
        self._gate_coeffs: dict[int, tuple[float, float]] = {}
        self._loader: DataLoader | None = None

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        test_loader = cast("DataLoader | None", kwargs.get("test_loader"))
        source: SpectralModel = self.model  # type: ignore[assignment]
        config: CoupledBackwardConfig = self.config  # type: ignore[assignment]

        self._is_vector, self._n_outputs, self._is_one_hot = self._peek_label_shape(
            loader
        )
        if self._is_vector:
            raise NotImplementedError(
                "CoupledBackwardTrainer supports scalar labels only."
            )
        for i, block in enumerate(source.blocks):
            b = cast(_Block, block)
            if cast(_Reduce, b.reduce).is_conv or b.pool is not None:
                raise NotImplementedError(
                    f"Block {i} is convolutional; the coupled trainer is FFN-only."
                )
            if not isinstance(b.transform, DenseTransform):
                raise NotImplementedError(
                    f"Block {i}: the coupled correction rewrites lift columns and "
                    f"needs a dense lift (got {type(b.transform).__name__})."
                )
        if float(source.readout_weight.detach().abs().sum().item()) == 0.0:
            log.warning(
                "Input model has an all-zero readout — it looks unfitted.  The "
                "top of the backward chain is the Phase-I ridge readout."
            )

        corrected = copy.deepcopy(source)
        self._corrections = {}
        self._gate_coeffs = {}
        self._loader = loader
        layers: list[dict[str, Any]] = []
        n_blocks = len(source.blocks)
        # Boundary j couples block j (patterns q_i over its neurons) and block
        # j+1 (whose reduce + lift receive the coupled columns).  Top-down.
        for j in reversed(range(n_blocks - 1)):
            if cast(_Reduce, cast(_Block, source.blocks[j + 1]).reduce).is_identity:
                log.info("Boundary %d: block %d has no reduce — skipped.", j, j + 1)
                continue
            self._corrections[j] = self._correct_boundary(source, corrected, j, loader)
            layers.append(self._corrections[j].entry)
        layers.reverse()

        self.model = corrected
        final = self._fit_blocked_readout(loader, test_loader)
        if config.verbose:
            log.info(
                "Coupled backward fit: %d boundaries corrected, test_acc=%s",
                len(self._corrections),
                final.get("test_accuracy"),
            )
        results: dict[str, Any] = {
            "layers": layers,
            "final": final,
            "backward": {
                "algorithm": "column_coupled",
                "combine_mode": config.combine_mode,
                "chain": config.chain,
                "estimator_right": config.estimator_right,
                "gate_form": config.gate_form,
                "backward_fraction": config.backward_fraction,
                "backward_rank": config.backward_rank,
                "ranks": [e["rank"] for e in layers],
                "kept_forward": [e["kept_forward"] for e in layers],
            },
        }
        return corrected, results

    # ------------------------------------------------------------------ #
    # One boundary
    # ------------------------------------------------------------------ #

    def _rank(self, k_up: int, k_low: int) -> int:
        config: CoupledBackwardConfig = self.config  # type: ignore[assignment]
        r = (
            config.backward_rank
            if config.backward_rank is not None
            else int(round(config.backward_fraction * k_up))
        )
        cap = min(k_low, k_up) if config.combine_mode == "swap" else k_low
        r = max(1, min(r, cap))
        if config.combine_mode == "swap" and r >= k_up:
            r = k_up - 1 if k_up > 1 else 1
        return r

    @torch.no_grad()
    def _correct_boundary(
        self,
        source: SpectralModel,
        corrected: SpectralModel,
        j: int,
        loader: DataLoader,
    ) -> _Correction:
        config: CoupledBackwardConfig = self.config  # type: ignore[assignment]
        dev = torch.device(config.device)
        block_j = cast(_Block, source.blocks[j])
        block_up = cast(_Block, source.blocks[j + 1])
        red_up = cast(_Reduce, block_up.reduce)
        v_up = red_up.V  # (p_j, k_up)
        w_up = cast(Tensor, block_up.weight)  # (k_up, p_{j+1})
        w_j = cast(Tensor, block_j.weight)  # (k_j, p_j)
        p_j, k_up = v_up.shape
        k_j, p_up = w_j.shape[0], w_up.shape[1]
        output_right = config.estimator_right == "output"
        dim_right = p_j if output_right else k_j
        r = self._rank(k_up, dim_right)
        keep = k_up - r if config.combine_mode == "swap" else k_up
        mean_field = config.chain == "mean_field"

        b_acc = torch.zeros(p_up, dim_right, dtype=torch.float64, device=dev)
        s_acc = torch.zeros(k_j, k_j, dtype=torch.float64, device=dev)
        m1_acc = torch.zeros(k_j, dtype=torch.float64, device=dev)
        f_acc = torch.zeros(keep, dtype=torch.float64, device=dev)
        abar_acc = torch.zeros(p_j, dtype=torch.float64, device=dev)
        count = 0
        for x, y in loader:
            x_dev = x.to(dev, non_blocking=True)
            yb = y.to(dev, non_blocking=True).to(torch.float32)
            yb = yb if yb.ndim == 2 else yb.unsqueeze(1)
            feats = source.forward_features(x_dev)
            a_prev = x_dev if j == 0 else feats[j - 1]
            a_j = feats[j]
            h_star = block_j.expand_input(a_prev)  # (n, k_j)
            g_up = self._backward_signal(source, corrected, x_dev, feats, j + 1)
            right = a_j if output_right else h_star
            b_acc += (g_up.transpose(0, 1) @ (yb * right)).double()
            if not output_right:
                # The neuron pattern is q̃ = W_jᵀ v, so ⟨q̃, a_j⟩ = vᵀ (W_j a_j):
                # its first/second moments come from those of W_j a_j.
                wa = a_j @ w_j.transpose(0, 1)  # (n, k_j)
                s_acc += (wa.transpose(0, 1) @ wa).double()
                m1_acc += wa.sum(dim=0).double()
            fwd = red_up(a_j)[:, :keep]  # forward coordinates (whitened if so)
            f_acc += (fwd * fwd).sum(dim=0).double()
            if mean_field:
                # Random part of the propagation through the kept columns.
                tk = g_up @ w_up[:keep].transpose(0, 1) / block_up.c
                if red_up.whiten:
                    tk = tk / red_up.scale[:keep]
                abar_acc += (tk @ v_up[:, :keep].transpose(0, 1)).sum(dim=0).double()
            count += x_dev.shape[0]
        if count == 0:
            raise RuntimeError(f"Boundary {j}: empty loader, cannot fit.")

        b_hat = b_acc / count
        second = s_acc / count
        first = m1_acc / count
        rms_fwd = float((f_acc / count).mean().sqrt().item())

        u, s, v = boundary_svd(b_hat)
        n_avail = int((s > 0).sum().item())
        if r > n_avail:
            log.warning(
                "Boundary %d: requested %d coupled pairs but B has rank %d; "
                "keeping %d.",
                j,
                r,
                n_avail,
                n_avail,
            )
            r = max(1, n_avail)
            keep = k_up - r if config.combine_mode == "swap" else k_up
        u_r, s_r, v_r = u[:, :r], s[:r], v[:, :r]

        if output_right:
            # Right singular vectors already are neuron patterns of block j;
            # their coordinate moments need one more streaming pass.
            q_raw = v_r  # (p_j, r)
            c2 = torch.zeros(r, dtype=torch.float64, device=dev)
            c1 = torch.zeros(r, dtype=torch.float64, device=dev)
            for x, _y in loader:
                a_j = source.forward_to_block(x.to(dev, non_blocking=True), stop=j + 1)
                coords = a_j.double() @ q_raw
                c2 += (coords * coords).sum(dim=0)
                c1 += coords.sum(dim=0)
            coord_second, coord_mean = c2 / count, c1 / count
        else:
            # Neuron patterns q̃_i = W_jᵀ v_i (p_j,) and their coordinate moments.
            q_raw = w_j.double().transpose(0, 1) @ v_r  # (p_j, r)
            coord_second = torch.einsum("ir,ij,jr->r", v_r, second, v_r)
            coord_mean = v_r.transpose(0, 1) @ first
        coord_rms = coord_second.clamp_min(0).sqrt()
        if red_up.whiten:
            q_cols = q_raw
            new_mean = coord_mean
            new_scale = (coord_second - coord_mean**2).clamp_min(1e-12).sqrt()
        else:
            q_cols = q_raw * (rms_fwd / coord_rms.clamp_min(1e-12))
            new_mean = torch.zeros(r, dtype=torch.float64, device=dev)
            new_scale = torch.ones(r, dtype=torch.float64, device=dev)
        lift_rows = (math.sqrt(p_up) * u_r).transpose(0, 1)  # (r, p_up)

        # Rewrite block j+1 of the corrected model: reduce [V_kept | Q], lift
        # [R_kept | √p U] (rows of the (k, p) weight are the lift columns).
        cblock = cast(_Block, corrected.blocks[j + 1])
        cred = cast(_Reduce, cblock.reduce)
        dtype = v_up.dtype
        new_reduce = _Reduce(
            in_width=p_j, keep=keep + r, is_conv=False, whiten=red_up.whiten
        ).to(dev)
        new_reduce.V.copy_(torch.cat([v_up[:, :keep], q_cols.to(dtype)], dim=1))
        new_reduce.mean.copy_(torch.cat([cred.mean[:keep], new_mean.to(dtype)]))
        new_reduce.scale.copy_(torch.cat([cred.scale[:keep], new_scale.to(dtype)]))
        cblock.reduce = new_reduce
        new_w = nn.Parameter(
            torch.cat([w_up[:keep], lift_rows.to(w_up.dtype)], dim=0),
            requires_grad=False,
        )
        cblock.weight = new_w
        cblock.transform = DenseTransform(new_w)
        cfgs = list(corrected.block_config.blocks)
        cfgs[j + 1] = replace(
            cfgs[j + 1], reduce=replace(cfgs[j + 1].reduce, k=keep + r)
        )
        corrected.block_config = BlockedModelConfig(
            blocks=tuple(cfgs), final_reduce=corrected.block_config.final_reduce
        )

        entry: dict[str, Any] = {
            "boundary": j,
            "block_idx": j + 1,
            "corrected": True,
            "rank": r,
            "kept_forward": keep,
            "dropped_forward": k_up - keep,
            "combine_mode": config.combine_mode,
            "chain": config.chain,
            "estimator_right": config.estimator_right,
            "gate_form": config.gate_form,
            "gate_coeffs": (
                list(self._gate_coeffs_for(source, j + 1))
                if config.gate_form == "linear"
                else None
            ),
            "singular_values": [float(x) for x in s_r.tolist()],
            "singular_values_all": [float(x) for x in s[: min(len(s), 64)].tolist()],
            "coord_rms_forward": rms_fwd,
            "coord_rms_backward_raw": [float(x) for x in coord_rms.tolist()],
            "lift_column_norm": math.sqrt(p_up),
        }
        if config.verbose:
            log.info(
                "Boundary %d: r=%d keep=%d (mode=%s) σ=%s rms_fwd=%.3g",
                j,
                r,
                keep,
                config.combine_mode,
                [round(float(x), 4) for x in s_r[:4].tolist()],
                rms_fwd,
            )
        return _Correction(
            keep=keep,
            rank=r,
            a_bar=(abar_acc / count).to(dtype) if mean_field else None,
            entry=entry,
        )

    # ------------------------------------------------------------------ #
    # Backward chain
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _readout_input_grad(self, source: SpectralModel) -> Tensor:
        """``∂f/∂a_{L-1}`` of the Phase-I readout (constant over samples)."""
        w = source.readout_weight.detach()
        fr = cast(_Reduce, source.final_reduction)
        if fr.is_identity:
            return w
        if fr.is_conv:
            raise NotImplementedError("conv final reduce is not supported.")
        wk = w / fr.scale if fr.whiten else w
        return fr.V @ wk

    @torch.no_grad()
    def _backward_signal(
        self,
        source: SpectralModel,
        corrected: SpectralModel,
        x: Tensor,
        feats: list[Tensor],
        stop: int,
    ) -> Tensor:
        """``g_stop = ∂f/∂h_stop`` (n, p_stop) propagated from the top.

        σ' is evaluated on the Phase-I preactivations (``feats`` are the
        source model's block outputs); the transposes go through the
        *corrected* matrices of the boundaries above ``stop``.
        """
        config: CoupledBackwardConfig = self.config  # type: ignore[assignment]
        if config.chain in ("gate", "activation"):
            # Local variants: no propagation.  The signal of layer ``stop`` is
            # its own gate or activation, independent of the readout and of
            # every correction above (boundaries decouple).
            a_in = x if stop == 0 else feats[stop - 1]
            if config.chain == "gate":
                return self._gate(source, stop, a_in)
            return feats[stop]
        top = len(source.blocks) - 1
        a_in = x if top == 0 else feats[top - 1]
        zt = self._gate(source, top, a_in)
        g = zt * self._readout_input_grad(source).unsqueeze(0)
        for m in range(top - 1, stop - 1, -1):
            cblock = cast(_Block, corrected.blocks[m + 1])
            cred = cast(_Reduce, cblock.reduce)
            w_bar = cast(Tensor, cblock.weight)  # (k+r, p_{m+1})
            t = g @ w_bar.transpose(0, 1) / cblock.c
            if cred.whiten:
                t = t / cred.scale
            corr = self._corrections[m]
            if config.chain == "mean_field":
                assert corr.a_bar is not None
                r_prev = corr.a_bar.unsqueeze(0) + t[:, corr.keep :] @ cred.V[
                    :, corr.keep :
                ].transpose(0, 1)
            else:
                r_prev = t @ cred.V.transpose(0, 1)
            a_in = x if m == 0 else feats[m - 1]
            zt = self._gate(source, m, a_in)
            g = zt * r_prev
        return g

    @torch.no_grad()
    def _gate(self, source: SpectralModel, m: int, a_in: Tensor) -> Tensor:
        """Gate of block ``m``: ``σ'(h)/√p`` or its linearisation ``(c̃ + d̃ h)/√p``."""
        block = cast(_Block, source.blocks[m])
        if self.config.gate_form == "exact":  # type: ignore[attr-defined]
            return block_derivative_features(block, a_in)
        c_t, d_t = self._gate_coeffs_for(source, m)
        transform = cast(Any, block.transform)
        h = transform.project(block.expand_input(a_in)) / block.c
        return (c_t + d_t * h) / math.sqrt(block.out_features)

    @torch.no_grad()
    def _gate_coeffs_for(self, source: SpectralModel, m: int) -> tuple[float, float]:
        """``(c̃, d̃) = (E[σ'(h)], E[h σ'(h)])`` of block ``m``; one pass, cached."""
        if m in self._gate_coeffs:
            return self._gate_coeffs[m]
        if self._loader is None:
            raise RuntimeError("gate coefficients need the training loader (call fit).")
        block = cast(_Block, source.blocks[m])
        sigma_prime = ACTIVATION_DERIVATIVES[block.activation]
        transform = cast(Any, block.transform)
        dev = torch.device(self.config.device)
        sp_sum = 0.0
        hsp_sum = 0.0
        n_el = 0
        for x, _y in self._loader:
            x_dev = x.to(dev, non_blocking=True)
            a_in = x_dev if m == 0 else source.forward_to_block(x_dev, stop=m)
            h = transform.project(block.expand_input(a_in)) / block.c
            sp = sigma_prime(h)
            sp_sum += float(sp.double().sum().item())
            hsp_sum += float((h * sp).double().sum().item())
            n_el += h.numel()
        self._gate_coeffs[m] = (sp_sum / n_el, hsp_sum / n_el)
        return self._gate_coeffs[m]
