"""Row-then-column backward correction by damped interpolation on dense layers.

A variant of :mod:`.backward_rowcol` (the backward step of the paper) in
which the low-degree estimate of every step is *blended* into the layer instead
of being appended to, or swapped into, its filter:

    W_l     <- (1 - alpha) W_l     + alpha  (V<-_l Gamma_row) (R<-_l)^T       (rows)
    W_{l+1} <- (1 - alpha) W_{l+1} + alpha  (Q_l Gamma_col) U_l^T             (columns)

with the same estimates as Algorithm 2: ``U, V<-`` the leading singular pairs
of ``A_l = E[y g^{l+1} z_{l-1}^T]``, ``s = U^T g^{l+1}`` the frozen scores,
``Q_l = E[z~_l (y s)^T]`` on the recomputed features, ``R<-`` a fresh Gaussian
lift, ``Gamma`` the per-column scale ("beta forward coordinates"; the forward
coordinate scale ``tau`` is measured with the fitted forward filters, kept as
frozen references, on the CURRENT features).  There is no swap, no importance
ranking and no width: every layer is a dense matrix ``W_l`` (the fitted forward
model ``R V^T`` is materialised once at the start, identity filters), each
update is a rank-``r`` outer product added to it, and every existing entry is
damped by ``1 - alpha`` at every update ("literal" rule: the middle layer,
updated twice per pass, is damped twice).  The backward signal is the
first-order one (Approx. 6' of the notes) read through the dense Phase-I copies.

Scale is a gauge.  The blocks have no bias and ReLU is positively homogeneous,
so a global factor on ``W_l`` changes no prediction: the damped update is, in
function space, the plain append of each estimate with scale
``beta alpha / (1 - alpha)``.  The interpolation is nevertheless stored
literally; to keep every feature scale (and the ridge penalty grid) where the
forward fit put them, a layer whose preactivation RMS has drifted by more than
``drift_guard`` from its forward-fit value is multiplied by an exact power of
two at the end of the pass, before the readout refit.

Scope: flat FFN blocks, dense lifts, no whitening, identity final reduction,
scalar labels, ``chain="first_order"``.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.block_config import BlockedModelConfig, ReduceConfig
from ..models.spectral import SpectralModel, _Block, _Reduce
from ..utils.transforms import DenseTransform
from .backward_rowcol import EPS, RowColBackwardTrainer, RowColConfig, _cross, _rms_cols

log = logging.getLogger(__name__)

__all__ = ["RowColDenseConfig", "RowColDenseTrainer"]


@dataclass
class RowColDenseConfig(RowColConfig):
    """Configuration of :class:`RowColDenseTrainer`.

    Inherits ``r_frac`` (rank of every boundary as a fraction of the configured
    cutoff ``k``), ``beta`` (scale of an estimate in forward coordinates),
    ``n_passes``, ``chunk_size``, ``svd_oversample``, ``svd_niter`` and
    ``lift_seed`` from :class:`RowColConfig`; ``chain`` must be
    ``"first_order"``; ``column_scale`` / ``swap_rule`` / ``tau_over`` /
    ``upper_update`` are not used.

    Parameters
    ----------
    alpha
        Interpolation weight of the estimate, in ``(0, 1]``: every update is
        ``W <- (1 - alpha) W + alpha * estimate``.
    drift_guard
        A layer whose preactivation RMS is more than this factor away from its
        forward-fit value is rescaled by an exact power of two at the end of
        the pass (no effect on the predictor).  ``0`` disables the guard.
    """

    alpha: float = 0.1
    chain: str = "first_order"
    drift_guard: float = 8.0


class RowColDenseTrainer(RowColBackwardTrainer):
    """Damped-interpolation row-then-column correction on dense layers.

    ``fit(loader, test_loader=..., pass_callback=...)`` materialises the fitted
    forward model as dense layers, runs ``n_passes`` rounds and returns
    ``(corrected, results)`` for the last pass; ``pass_callback(p, results)``
    is called after every round.  The corrected model is a
    :class:`SpectralModel` with identity filters.
    """

    def __init__(self, model: SpectralModel, config: RowColDenseConfig) -> None:
        if not 0.0 < config.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {config.alpha}")
        if config.chain != "first_order":
            raise NotImplementedError("RowColDenseTrainer: chain='first_order' only.")
        if config.drift_guard < 0.0:
            raise ValueError("drift_guard must be >= 0")
        super().__init__(model, config)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def fit(self, loader: DataLoader, **kwargs: object) -> tuple[Any, dict[str, Any]]:
        test_loader = cast("DataLoader | None", kwargs.get("test_loader"))
        pass_callback = cast("Any", kwargs.get("pass_callback"))
        config: RowColDenseConfig = self.config  # type: ignore[assignment]
        dev = torch.device(config.device)

        is_vector, _n_out, _oh = self._peek_label_shape(loader)
        if is_vector:
            raise NotImplementedError("RowColDenseTrainer supports scalar labels.")
        source: SpectralModel = self.model  # type: ignore[assignment]
        for i, block in enumerate(source.blocks):
            b = cast(_Block, block)
            red = cast(_Reduce, b.reduce)
            if red.is_conv or b.pool is not None:
                raise NotImplementedError(f"Block {i} is convolutional; FFN only.")
            if not isinstance(b.transform, DenseTransform):
                raise NotImplementedError(f"Block {i} needs a dense lift.")
            if red.is_identity or red.whiten:
                raise NotImplementedError(f"Block {i}: needs a plain fitted filter.")
        if not cast(_Reduce, source.final_reduction).is_identity:
            raise NotImplementedError("The final reduction must be the identity.")

        model = self._densify(source, dev)
        self.model = model
        n_blocks = len(model.blocks)
        cfgs = source.block_config.blocks
        self._r = [
            max(1, int(round(config.r_frac * int(cast(int, cfgs[j].reduce.k)))))
            for j in range(n_blocks - 1)
        ]
        self._gen = torch.Generator(device="cpu").manual_seed(int(config.lift_seed))

        X, y = self._materialise(loader, dev)
        self._rms0 = self._preact_rms(model, X)  # forward-fit preactivation scales
        results: dict[str, Any] = {}
        for p in range(1, config.n_passes + 1):
            entries = self._round(model, X, y)
            rescaled = self._guard(model, X)
            final = self._fit_blocked_readout(loader, test_loader)
            results = {
                "layers": entries,
                "final": final,
                "backward": {
                    "algorithm": "rowcol_dense",
                    "pass": p,
                    "n_passes": config.n_passes,
                    "r_frac": config.r_frac,
                    "beta": config.beta,
                    "alpha": config.alpha,
                    "chain": config.chain,
                    "ranks": list(self._r),
                    "rescaled": rescaled,
                },
            }
            if pass_callback is not None:
                pass_callback(p, copy.deepcopy(results))
        return model, results

    # ------------------------------------------------------------------ #
    # Dense model
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _densify(self, source: SpectralModel, dev: torch.device) -> SpectralModel:
        """The forward model as dense layers ``W_l = V_l R_l^T`` (identity
        filters); the fitted filters are kept in ``self._v0`` as references."""
        cfg = source.block_config
        dense_cfg = dataclasses.replace(
            cfg,
            blocks=tuple(
                dataclasses.replace(b, reduce=ReduceConfig(k=None)) for b in cfg.blocks
            ),
        )
        model = SpectralModel.from_block_config(
            cast(BlockedModelConfig, dense_cfg),
            input_shape=tuple(cast(Any, source.input_shape)),
            seed=0,
        ).to(dev)
        self._v0: list[Tensor] = []
        for src_b, dst_b in zip(source.blocks, model.blocks, strict=True):
            sb, db = cast(_Block, src_b), cast(_Block, dst_b)
            v = cast(_Reduce, sb.reduce).V.detach().to(dev)  # (p_in, K)
            w = cast(Tensor, sb.weight).detach().to(dev)  # (K, p)
            cast(Tensor, db.weight).copy_(v @ w)  # (p_in, p)
            cast(Tensor, db.c).copy_(sb.c.detach().to(dev))
            self._v0.append(v.clone())
        model.readout_weight = source.readout_weight
        model.readout_bias = source.readout_bias
        return model

    @torch.no_grad()
    def _preact_rms(self, model: SpectralModel, X: Tensor) -> list[float]:
        """RMS of every layer's preactivation ``h_l`` on the training set."""
        chunk = int(cast(RowColDenseConfig, self.config).chunk_size)
        out: list[float] = []
        a = X
        for blk in model.blocks:
            b = cast(_Block, blk)
            transform = cast(Any, b.transform)
            acc, count = 0.0, 0
            zs = []
            for ac in a.split(chunk):
                h = transform.project(b.expand_input(ac)) / b.c
                acc += float(h.double().pow(2).sum())
                count += h.numel()
                zs.append(b(ac))
            out.append(math.sqrt(acc / max(count, 1)))
            a = torch.cat(zs)
        return out

    @torch.no_grad()
    def _guard(self, model: SpectralModel, X: Tensor) -> list[dict[str, float]]:
        """Multiply a drifted layer by an exact power of two (predictor unchanged)."""
        config: RowColDenseConfig = self.config  # type: ignore[assignment]
        guard = float(config.drift_guard)
        if guard <= 0.0:
            return []
        done: list[dict[str, float]] = []
        rms = self._preact_rms(model, X)
        for j, (r, r0) in enumerate(zip(rms, self._rms0, strict=True)):
            ratio = r / max(r0, EPS)
            if ratio > guard or ratio < 1.0 / guard:
                factor = 2.0 ** round(-math.log2(max(ratio, EPS)))
                cast(Tensor, cast(_Block, model.blocks[j]).weight).mul_(factor)
                done.append({"layer": j, "ratio": ratio, "factor": factor})
                log.info(
                    "layer %d preactivation drift %.3g: rescaled by %g",
                    j,
                    ratio,
                    factor,
                )
                rms = self._preact_rms(model, X)  # the layers above feel it
        return done

    # ------------------------------------------------------------------ #
    # One round
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _round(
        self, model: SpectralModel, X: Tensor, y: Tensor
    ) -> list[dict[str, Any]]:
        config: RowColDenseConfig = self.config  # type: ignore[assignment]
        chunk = int(config.chunk_size)
        beta, alpha = float(config.beta), float(config.alpha)
        n_blocks = len(model.blocks)
        feats, gates = self._state(model, X)
        phase1 = copy.deepcopy(model)  # the network at the start of the pass
        g_chain = gates[-1] * self._readout_grad(model).unsqueeze(0)
        ones = y.new_ones(y.shape)
        entries: list[dict[str, Any]] = []
        for j in range(n_blocks - 2, -1, -1):
            z_prev = X if j == 0 else feats[j - 1]
            g_up = g_chain
            r = self._r[j]

            # 1. directions and frozen scores
            a_mat = _cross(g_up, z_prev, y, chunk)  # (p_{j+1}, p_{j-1})
            u, s_vals, v_read = self._thin_svd(a_mat, r)
            scores = (g_up.double() @ u).to(X.dtype)  # (n, r)
            ys = y.unsqueeze(1) * scores

            # 2. rows of layer j: W_j <- (1-a) W_j + a (V<- Gamma_row) R<-^T
            blk = cast(_Block, model.blocks[j])
            w = cast(Tensor, blk.weight)  # (p_{j-1}, p_j)
            tau_row = float(_rms_cols(z_prev, self._v0[j], chunk).mean())
            v_new = v_read.to(w.dtype)
            v_new = v_new * (
                beta * tau_row / _rms_cols(z_prev, v_new, chunk).clamp_min(EPS)
            ).to(w.dtype)
            r_new = torch.randn(r, w.shape[1], generator=self._gen).to(w)
            w.mul_(1.0 - alpha).add_(v_new @ r_new, alpha=alpha)
            feats[j], gates[j] = self._block_state(blk, z_prev, chunk)
            z_tilde = feats[j]

            # 3. columns of layer j+1: W_{j+1} <- (1-a) W_{j+1} + a (Q Gamma_col) U^T
            blk_up = cast(_Block, model.blocks[j + 1])
            w_up = cast(Tensor, blk_up.weight)  # (p_j, p_{j+1})
            q_mat = _cross(z_tilde, ys, ones, chunk).to(w_up.dtype)  # (p_j, r)
            tau_col = float(_rms_cols(z_tilde, self._v0[j + 1], chunk).mean())
            target = beta * math.sqrt(w_up.shape[1]) * tau_col
            rms_q = _rms_cols(z_tilde, q_mat, chunk)
            q_mat = q_mat * (target / rms_q.clamp_min(EPS)).to(w_up.dtype)
            w_up.mul_(1.0 - alpha).add_(
                q_mat @ u.to(w_up.dtype).transpose(0, 1), alpha=alpha
            )

            # 4. first-order signal of block j for the boundary below
            if j > 0:
                g_chain = self._first_order_dense(
                    phase1, j, g_up, gates[j], v_read, scores, z_tilde, target
                )
            entries.append(
                {
                    "boundary": j,
                    "rank": r,
                    "alpha": alpha,
                    "singular_values": [float(x) for x in s_vals[:8].tolist()],
                    "tau_row": tau_row,
                    "tau_col": tau_col,
                    "rms_q_mean": float(rms_q.mean()),
                    "rms_q_max": float(rms_q.max()),
                }
            )
        return entries

    @torch.no_grad()
    def _first_order_dense(
        self,
        phase1: SpectralModel,
        j: int,
        g_up: Tensor,
        gate_j: Tensor,
        v_read: Tensor,
        scores: Tensor,
        z_j: Tensor,
        target: float,
    ) -> Tensor:
        """Approx. 6' on dense Phase-I layers: ``g^j = gate_j * (rbar + spike)``
        with ``rbar`` the sample mean of ``W^0_{j+1} g_up / c`` and the spike
        ``sum_i s_i q0_i / c_{j+1}``, ``q0_i = W^0_j v_i / c_j`` rescaled so that
        ``RMS<q0_i, z_j>`` is the column target of this boundary."""
        chunk = int(cast(RowColDenseConfig, self.config).chunk_size)
        blk_up = cast(_Block, phase1.blocks[j + 1])
        w_up = cast(Tensor, blk_up.weight)  # (p_j, p_{j+1})
        r_bar = ((g_up @ w_up.transpose(0, 1)) / blk_up.c).mean(dim=0, keepdim=True)
        blk = cast(_Block, phase1.blocks[j])
        w = cast(Tensor, blk.weight)  # (p_{j-1}, p_j)
        q0 = (v_read.to(w.dtype).transpose(0, 1) @ w) / blk.c  # (r, p_j)
        rms = _rms_cols(z_j, q0.transpose(0, 1), chunk)
        q0 = q0 * (target / rms.clamp_min(EPS)).to(q0.dtype).unsqueeze(1)
        spike = (scores @ q0) / blk_up.c
        return gate_j * (r_bar + spike)
