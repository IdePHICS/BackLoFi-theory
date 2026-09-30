"""Row-then-column backward correction on the 2-conv + avg-pool + FC network.

The row-then-column backward correction of :mod:`.backward_rowcol` for the
convolutional model ``conv1 (no reduce) → [k_c] conv2 + avg-pool → [k_3,
flatten] FC → ridge`` — the mean-field / per-channel / importance arm of the
FFN campaign (:mod:`.backward_rowcol`), transcribed to conv blocks and
**streamed**: nothing of size ``n × p_c × H × W`` is ever stored; every
statistic is one chunked forward pass over the training set.

Boundary ``conv2 | FC`` (rank ``r₁ = round(r_frac · min(k_c, k_3))``):

1. ``A₁ = E[y g^FC z̄^conv1ᵀ]`` with ``g^FC = a ⊙ σ'(h^FC)/√p_f`` (the exact
   top boundary condition) and ``z̄`` the spatial mean of conv1's output —
   equal to the locations-as-samples matrix with ``g`` broadcast to every
   location.  Thin SVD: write directions ``U`` (FC neurons), read directions
   ``V_←`` (conv1 channels).  Frozen scores ``s = Uᵀ g^FC``.
2. *Rows of conv2*: the ``r₁`` least-important reduce columns (channel
   directions) are replaced by ``V_←``, each scaled so that its coordinate RMS
   over samples × locations is ``β`` times the mean forward coordinate RMS,
   with a fresh ``randn`` 3×3 kernel per filter as lift.
3. *Columns of the FC*: ``Q̃ = E[z̄̃^conv2 (y s)ᵀ]`` (channel directions on the
   pooled map), replacing the ``r₁`` least-important FC reduce columns; the
   ``H'W'`` flattened lift rows of a paired channel are all ``u_i`` (coherent
   over locations) and the channel is scaled so that its per-neuron RMS
   contribution equals ``β`` times that of a forward channel (which sums
   ``H'W'`` random-sign terms): ``RMS⟨q̃, z̄̃⟩ = β √p_f τ / √(H'W')``.

Boundary ``conv1 | conv2`` (rank ``r₀ = round(r_frac · k_c)``; conv1 has no
reduce, so column-only):

4. Chain (Algorithm 2 line 8) through the corrected FC:
   ``g^conv2 = σ'(h^conv2)/√p_c ⊙ unpool(V̄_FC · reshape(R̄_FCᵀ g^FC / c_FC))``
   with the pool's adjoint as ``unpool`` (avg: each cell spread / (s·s);
   max: routed to the arg-max location of each cell) and the paired branch
   ``Q̃ s / c_FC`` per sample; ``chain="mean_field"`` replaces the random
   branch (every non-paired FC column) by its sample mean per (channel,
   location), ``chain="exact"`` keeps it per sample with the frozen
   ``g^FC`` of the top stream (the paragraph "Exact chain (top-down)"), and
   ``chain="first_order"`` (Approx. 6') takes the random branch through the
   PHASE-I FC (all its columns; Phase I = the network at the start of the
   pass) and replaces the installed paired branch by the order-zero spike:
   the frozen scores times ``q⁰_i``, the read-direction ``v_i`` of the
   conv2|FC boundary lifted through the Phase-I conv2 (channel reduce, then
   the kernel summed over its taps: the response to a spatially constant
   coordinate map), a per-channel pattern scaled to the FC column step's
   per-channel target exactly like the installed ``Q̃``.
5. ``A₀ = E[y g^conv2_x z^conv1_xᵀ]`` with locations as samples (``g`` at the
   matched location); thin SVD ``U₀``; the paired columns ``Q̃₀ = A₀ᵀ U₀``
   replace the ``r₀`` least-important conv2 columns, lifted by ``u_j`` on the
   centre tap of filter ``j`` (zero elsewhere) and scaled so that the
   per-neuron RMS equals ``β`` times a forward channel's (9 random taps):
   ``RMS⟨q̃, z^conv1_x⟩ = β √p_c √9 τ``.

*Importance* (section "Feature importance at fixed budget", exact ``Ω``) on
conv2's columns: rate per (filter ``j``, tap ``t``) ``e λ_j(a) c_{j,a,t} + d
ℓ_j(a)`` with the moments ``Λ₂(a) = E[y ĝ mean_x h_{a,x}²]`` (locations as
samples, ``ĝ = ḡ + UUᵀ(g − ḡ)`` the retained FC signal) and ``ω̄_j`` filter
``j``'s location-summed effective weight into the FC neurons (through the
pool, ``V_FC`` and the ``H'W'`` lift rows); importance = RMS over filters ×
taps.  On the FC's columns: the top-block rule with ``ω = a`` and one
coefficient per (neuron, location); RMS over neurons × locations.

The forward scale ``τ`` of every swap follows ``tau_over``: ``"random_lift"``
(the campaign) averages the coordinate RMS over the surviving forward and
read columns only, so paired columns (large coordinates, small lifts) never
inflate it; ``"all"`` averages over every survivor.

``upper_update="block"`` (Algorithm 3, "Closing the step: the rows of layer
ℓ+1") re-evaluates every feature of the FC on the updated pooled map before the
FC's swap, with no extra pass over the data: the S2 stream already holds the
signed moments ``c_y, m_y`` and ``Σ_z = m_loc`` (locations as samples, the
forward fit's own statistics), so the FC's forward group is refitted as forward
LoFi of ``z̃¹`` and rotated to the old gauge (ghost-basis Procrustes of
:func:`.backward_rowcol.refit_aligned`), and every surviving paired column of
an earlier pass is recomputed as ``E[z̄̃¹ (y s)ᵀ]`` with its stored scores at
the per-channel target.  The FC is the top block (no read columns), and conv2
is never stale: conv1 has no row step, so conv2's input never changes.

Scope: exactly three blocks (conv, conv + pool, fc), dense lifts, no whitening,
scalar labels, ``chain="mean_field"``, ``column_scale="per_channel"``,
``swap_rule="importance"``.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..models.spectral import SpectralModel, _Block, _Reduce
from ..utils.helpers import ACTIVATIONS
from ..utils.transforms import DenseConvTransform, DenseTransform
from .backward_rowcol import (
    EPS,
    FWD,
    PAIR,
    READ,
    RowColBackwardTrainer,
    RowColConfig,
    _sigma_prime,
    refit_aligned,
)

log = logging.getLogger(__name__)

CONV1, CONV2, FC = 0, 1, 2

__all__ = ["RowColCNNTrainer", "RowColConfig"]


def _cols(red: _Reduce) -> Tensor:
    """The reduce columns of a conv reduce as ``(K, p)`` channel directions."""
    return red.V.reshape(red.V.shape[0], red.V.shape[1])


def _set_cols(red: _Reduce, idx: Tensor, cols: Tensor) -> None:
    red.V[idx] = cols.to(red.V.dtype).reshape(cols.shape[0], cols.shape[1], 1, 1)


def _coords(z: Tensor, cols: Tensor) -> Tensor:
    """Coordinates ``⟨v_a, z_x⟩`` of the map ``z`` (n, p, H, W) on the channel
    directions ``cols`` (K, p): ``(n, K, H, W)``."""
    return F.conv2d(z, cols.to(z.dtype).reshape(cols.shape[0], cols.shape[1], 1, 1))


def _tau(q2: Tensor, keep: Tensor, tags: list[str], tau_over: str) -> float:
    """Mean coordinate RMS of the surviving columns that define the forward
    scale (``random_lift``: forward and read columns only, when any survive)."""
    if tau_over == "random_lift":
        rl = torch.tensor([t in (FWD, READ) for t in tags], device=keep.device)
        if bool((keep & rl).any()):
            keep = keep & rl
    return float(q2[keep].sqrt().mean())


def _loc_rows(z: Tensor) -> Tensor:
    """``(n, p, H, W)`` → ``(n·H·W, p)`` rows, locations as samples."""
    n, p = z.shape[0], z.shape[1]
    return z.permute(0, 2, 3, 1).reshape(n * z.shape[2] * z.shape[3], p)


class RowColCNNTrainer(RowColBackwardTrainer):
    """Streamed Algorithm 2 on ``conv1 → [k_c] conv2 + pool → [k_3] FC → ridge``.

    ``fit(loader, test_loader=..., pass_callback=...)`` deep-copies the model,
    runs ``n_passes`` rounds and returns ``(corrected, results)`` for the last
    pass.  ``results["backward"]["origins"]`` lists, per block, the origin of
    every reduce column (``fwd`` / ``read`` / ``pair``; block 0 has none).
    """

    def __init__(self, model: SpectralModel, config: RowColConfig) -> None:
        if config.chain not in ("mean_field", "exact", "first_order"):
            raise NotImplementedError(
                "RowColCNNTrainer: chain in {'mean_field', 'exact', 'first_order'}."
            )
        if config.column_scale != "per_channel":
            raise NotImplementedError(
                "RowColCNNTrainer: column_scale='per_channel' only."
            )
        if config.swap_rule != "importance":
            raise NotImplementedError("RowColCNNTrainer: swap_rule='importance' only.")
        super().__init__(model, config)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check(model: SpectralModel) -> None:
        if len(model.blocks) != 3:
            raise NotImplementedError("RowColCNNTrainer expects conv, conv+pool, fc.")
        b0, b1, b2 = (cast(_Block, b) for b in model.blocks)
        if not cast(_Reduce, b0.reduce).is_identity or not isinstance(
            b0.transform, DenseConvTransform
        ):
            raise NotImplementedError("Block 0 must be a dense conv with no reduce.")
        r1, r2 = cast(_Reduce, b1.reduce), cast(_Reduce, b2.reduce)
        if (
            r1.is_identity
            or not r1.is_conv
            or not isinstance(b1.transform, DenseConvTransform)
            or b1.pool is None
            or b1.pool.mode not in ("avg", "max")
        ):
            raise NotImplementedError(
                "Block 1 must be a dense conv with a channel reduce and a pool."
            )
        if (
            r2.is_identity
            or not r2.is_conv
            or not isinstance(b2.transform, DenseTransform)
            or not any(type(op).__name__ == "Flatten" for op in b2.pre)
        ):
            raise NotImplementedError(
                "Block 2 must be a dense FC with a channel reduce and a flatten."
            )
        if r1.whiten or r2.whiten:
            raise NotImplementedError("Whitened reduces are unsupported.")
        if not model.final_reduction.is_identity:
            raise NotImplementedError("The final reduction must be the identity.")

    @torch.no_grad()
    def fit(self, loader: DataLoader, **kwargs: object) -> tuple[Any, dict[str, Any]]:
        test_loader = cast("DataLoader | None", kwargs.get("test_loader"))
        pass_callback = cast("Any", kwargs.get("pass_callback"))
        config: RowColConfig = self.config  # type: ignore[assignment]
        dev = torch.device(config.device)

        is_vector, _n_out, _oh = self._peek_label_shape(loader)
        if is_vector:
            raise NotImplementedError("RowColCNNTrainer supports scalar labels.")
        source: SpectralModel = self.model  # type: ignore[assignment]
        self._check(source)
        model = copy.deepcopy(source).to(dev)
        self.model = model
        cfgs = model.block_config.blocks
        k_c = int(cast(int, cfgs[CONV2].reduce.k))
        k_3 = int(cast(int, cfgs[FC].reduce.k))
        w_c = int(cast(_Reduce, cast(_Block, model.blocks[CONV2]).reduce).V.shape[0])
        w_f = int(cast(_Reduce, cast(_Block, model.blocks[FC]).reduce).V.shape[0])
        r1 = min(max(1, round(config.r_frac * min(k_c, k_3))), w_c - 1, w_f - 1)
        r0 = min(max(1, round(config.r_frac * k_c)), w_c - 1)
        self._r = [r0, r1]
        self._origin = [[], [FWD] * w_c, [FWD] * w_f]
        self._gen = torch.Generator(device="cpu").manual_seed(int(config.lift_seed))
        self._m0: Tensor | None = None  # E[mean_x z⁰ z⁰ᵀ]: conv1 never changes
        self._init_block_state(model)

        X, y = self._materialise(loader, dev)
        results: dict[str, Any] = {}
        for p in range(1, config.n_passes + 1):
            self._pass = p
            entries = self._round(model, X, y)
            final = self._fit_blocked_readout(loader, test_loader)
            results = {
                "layers": entries,
                "final": final,
                "backward": {
                    "algorithm": "rowcol_cnn",
                    "pass": p,
                    "n_passes": config.n_passes,
                    "r_frac": config.r_frac,
                    "beta": config.beta,
                    "chain": config.chain,
                    "column_scale": config.column_scale,
                    "swap_rule": config.swap_rule,
                    "tau_over": config.tau_over,
                    "upper_update": config.upper_update,
                    "ranks": list(self._r),
                    "origins": [
                        {o: tags.count(o) for o in (FWD, READ, PAIR)}
                        for tags in self._origin
                    ],
                },
            }
            if pass_callback is not None:
                pass_callback(p, copy.deepcopy(results))
        return model, results

    # ------------------------------------------------------------------ #
    # Forward pieces (one chunk)
    # ------------------------------------------------------------------ #

    def _chunks(self, X: Tensor, y: Tensor):  # type: ignore[no-untyped-def]
        chunk = int(cast(RowColConfig, self.config).chunk_size)
        for i in range(0, X.shape[0], chunk):
            yield X[i : i + chunk], y[i : i + chunk], i

    @staticmethod
    def _conv1(model: SpectralModel, xc: Tensor) -> Tensor:
        return cast(_Block, model.blocks[CONV1])(xc)

    @staticmethod
    def _conv2(model: SpectralModel, z0: Tensor) -> tuple[Tensor, Tensor]:
        """``(h¹, z¹)``: pre-activation at full resolution and the pooled output."""
        blk = cast(_Block, model.blocks[CONV2])
        h1 = cast(Any, blk.transform).project(blk.expand_input(z0)) / blk.c
        z = ACTIVATIONS[blk.activation](h1) / math.sqrt(blk.out_features)
        return h1, cast(Any, blk.pool).apply(z)

    @staticmethod
    def _fc(model: SpectralModel, z1: Tensor) -> Tensor:
        """``h^FC`` of the FC block on the pooled conv2 output."""
        blk = cast(_Block, model.blocks[FC])
        return cast(Any, blk.transform).project(blk.expand_input(z1)) / blk.c

    def _g_fc(self, model: SpectralModel, h2: Tensor) -> Tensor:
        """``g^FC = a ⊙ σ'(h^FC)/√p_f``: the exact top boundary condition."""
        blk = cast(_Block, model.blocks[FC])
        gate = _sigma_prime(h2, str(blk.activation)) / math.sqrt(blk.out_features)
        return gate * self._readout_grad(model).unsqueeze(0)

    # ------------------------------------------------------------------ #
    # One round
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _round(
        self, model: SpectralModel, X: Tensor, y: Tensor
    ) -> list[dict[str, Any]]:
        config: RowColConfig = self.config  # type: ignore[assignment]
        beta = float(config.beta)
        conv2 = cast(_Block, model.blocks[CONV2])
        fc = cast(_Block, model.blocks[FC])
        red_c, red_f = cast(_Reduce, conv2.reduce), cast(_Reduce, fc.reduce)
        w_c2 = cast(Tensor, conv2.weight)  # (p_c, K_c, 3, 3)
        w_fc = cast(Tensor, fc.weight)  # (K_f·H'W', p_f)
        p_c, p_f = int(w_c2.shape[0]), int(w_fc.shape[1])
        taps = int(w_c2.shape[2] * w_c2.shape[3])
        r0, r1 = self._r
        entries: list[dict[str, Any]] = []
        # Phase-I network of this pass (first-order chain reads through it).
        self._phase1 = copy.deepcopy(model) if config.chain == "first_order" else None

        # ---- boundary conv2 | FC ------------------------------------------
        # S0: A₁ = E[y g^FC z̄⁰ᵀ], ḡ (and E[mean_x z⁰ z⁰ᵀ] once).
        a1, g_bar = self._stats_top(model, X, y)
        u1, s1, v_read = self._thin_svd(a1, r1)  # (p_f, r1), (r1,), (p_c, r1)
        self._u1, self._g_bar, self._v_read = u1, g_bar, v_read

        # S1: per-column moments of conv2's current columns + the read
        # directions on z⁰, the frozen scores, and conv2's (d, e).
        cols_c = _cols(red_c).double()  # (K_c, p_c)
        cand = torch.cat([cols_c, v_read.transpose(0, 1)])  # (K_c + r1, p_c)
        st = self._stats_conv_cols(model, X, y, cand, u1, g_bar)
        self._scores = st.pop("scores")  # (n, r1), frozen
        k_c = cols_c.shape[0]
        cur = {k: v[..., :k_c] if v.ndim else v for k, v in st.items()}
        new = {k: v[..., k_c:] if v.ndim else v for k, v in st.items()}
        imp = self._importance_conv(model, cur, cols_c.norm(dim=1))
        drop = torch.argsort(imp)[:r1]
        keep = torch.ones(k_c, dtype=torch.bool, device=imp.device)
        keep[drop] = False
        tau_c = _tau(cur["q2"], keep, self._origin[CONV2], config.tau_over)
        scale = beta * tau_c / new["q2"].sqrt().clamp_min(EPS)  # (r1,)
        v_new = v_read.transpose(0, 1) * scale.unsqueeze(1)  # (r1, p_c)
        _set_cols(red_c, drop, v_new)
        w_c2[:, drop] = torch.randn(
            p_c, r1, w_c2.shape[2], w_c2.shape[3], generator=self._gen
        ).to(w_c2)
        dropped_row = self._retag(CONV2, drop, READ)
        # Reassemble conv2's column statistics after the swap (z⁰ unchanged):
        # the new slots carry the read directions' moments, rescaled.
        for k in ("q2", "lam", "lam_s"):
            cur[k][..., drop] = new[k] * scale**2
        for k in ("mom", "mom_s"):
            cur[k][..., drop] = new[k] * scale
        self._conv_stats = cur

        # S2: the FC's statistics on the updated pooled map z̃¹.
        old_slots, old_scores = self._fc_old_pairs()
        fs = (
            self._stats_fc(model, X, y)
            if old_scores is None
            else self._stats_fc(model, X, y, old_scores)
        )
        # Algorithm 3: every feature of the FC on z̃¹, before its swap.
        refit = self._update_fc(model, fs, old_slots) if self._block_mode else None
        cols_f = _cols(red_f).double()  # (K_f, p_c)
        hw = int(w_fc.shape[0] // cols_f.shape[0])
        norms_f = cols_f.norm(dim=1).clamp_min(EPS)
        q2_f = torch.einsum("ap,pq,aq->a", cols_f, fs["m_loc"], cols_f)
        lam_f = torch.einsum("ap,pq,aq->a", cols_f, fs["c_y"], cols_f) / norms_f**2
        mom_f = (cols_f @ fs["m_y"]) / norms_f
        imp_f = self._importance_fc(model, lam_f, mom_f, norms_f, fs["d"], fs["e"])
        drop_up = torch.argsort(imp_f)[:r1]
        keep_f = torch.ones(cols_f.shape[0], dtype=torch.bool, device=imp_f.device)
        keep_f[drop_up] = False
        tau_f = _tau(q2_f, keep_f, self._origin[FC], config.tau_over)
        q_mat = fs["q"]  # (p_c, r1) = E[z̄̃¹ (y s)ᵀ]
        rms_q = torch.einsum("pa,pq,qa->a", q_mat, fs["m_bar"], q_mat).sqrt()
        target_f = beta * math.sqrt(p_f) * tau_f / math.sqrt(hw)
        q_mat = q_mat * (target_f / rms_q.clamp_min(EPS)).unsqueeze(0)
        self._m_bar, self._target_f = fs["m_bar"], target_f
        self._capture_ghosts(FC, _cols(red_f).transpose(0, 1), drop_up)
        _set_cols(red_f, drop_up, q_mat.transpose(0, 1))
        rows = drop_up.unsqueeze(1) * hw + torch.arange(hw, device=drop_up.device)
        w_fc[rows.reshape(-1)] = (
            u1.transpose(0, 1).unsqueeze(1).expand(-1, hw, -1).reshape(-1, p_f)
        ).to(w_fc)
        dropped_col = self._retag(FC, drop_up, PAIR)
        self._register(FC, drop_up, self._scores, boundary=1)
        self._q_fc = q_mat  # scaled paired channels, for the chain
        entries.append(
            {
                "boundary": 1,
                "rank": r1,
                "singular_values": [float(x) for x in s1[:8].tolist()],
                "tau_row": tau_c,
                "tau_col": tau_f,
                "eta": 1.0,
                "rms_q_mean": float(rms_q.mean()),
                "rms_q_max": float(rms_q.max()),
                "dropped_row": dropped_row,
                "dropped_col": dropped_col,
                **({"refit": refit} if refit is not None else {}),
            }
        )

        # ---- boundary conv1 | conv2 (column-only) ---------------------------
        # S3: A₀ = E[y g^conv2_x z⁰_xᵀ] with the mean-field chain through the
        # corrected FC.
        a0 = self._stats_boundary0(model, X, y)
        u0, s0, _v0 = self._thin_svd(a0, r0)  # (p_c, r0)
        q0 = a0.transpose(0, 1) @ u0  # (p_c, r0) = E[z⁰ (y s₀)ᵀ]
        cols_c = _cols(red_c).double()
        cur = self._conv_stats
        imp0 = self._importance_conv(model, cur, cols_c.norm(dim=1))
        drop0 = torch.argsort(imp0)[:r0]
        keep0 = torch.ones(cols_c.shape[0], dtype=torch.bool, device=imp0.device)
        keep0[drop0] = False
        tau_0 = _tau(cur["q2"], keep0, self._origin[CONV2], config.tau_over)
        m0 = cast(Tensor, self._m0)
        rms_q0 = torch.einsum("pa,pq,qa->a", q0, m0, q0).sqrt()
        target_0 = beta * math.sqrt(p_c) * math.sqrt(taps) * tau_0
        q0 = q0 * (target_0 / rms_q0.clamp_min(EPS)).unsqueeze(0)
        _set_cols(red_c, drop0, q0.transpose(0, 1))
        w_c2[:, drop0] = 0.0
        w_c2[:, drop0, w_c2.shape[2] // 2, w_c2.shape[3] // 2] = u0.to(w_c2)
        dropped_0 = self._retag(CONV2, drop0, PAIR)
        # Moments of the paired columns are not needed later in this round
        # (they are re-streamed at the next round's S1); keep q2 for tau.
        cur["q2"][drop0] = target_0**2
        entries.append(
            {
                "boundary": 0,
                "rank": r0,
                "singular_values": [float(x) for x in s0[:8].tolist()],
                "tau_col": tau_0,
                "eta": 1.0,
                "rms_q_mean": float(rms_q0.mean()),
                "rms_q_max": float(rms_q0.max()),
                "dropped_col": dropped_0,
            }
        )
        del self._conv_stats, self._scores, self._q_fc, self._phase1
        return entries

    # ------------------------------------------------------------------ #
    # Streamed statistics
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _stats_top(
        self, model: SpectralModel, X: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``A₁ = E[y g^FC z̄⁰ᵀ]`` and ``ḡ = E[g^FC]`` (float64); caches
        ``E[mean_x z⁰ z⁰ᵀ]`` on the first call."""
        n = X.shape[0]
        p_c = int(cast(Tensor, cast(_Block, model.blocks[CONV2]).weight).shape[0])
        p_f = int(cast(Tensor, cast(_Block, model.blocks[FC]).weight).shape[1])
        a1 = torch.zeros(p_f, p_c, dtype=torch.float64, device=X.device)
        g_bar = torch.zeros(p_f, dtype=torch.float64, device=X.device)
        m0 = (
            torch.zeros(p_c, p_c, dtype=torch.float64, device=X.device)
            if self._m0 is None
            else None
        )
        exact = cast(RowColConfig, self.config).chain == "exact"
        self._g_fc_frozen = (
            torch.zeros(n, p_f, dtype=X.dtype, device=X.device) if exact else None
        )
        hw = 0
        for xc, yc, i in self._chunks(X, y):
            z0 = self._conv1(model, xc)
            _h1, z1 = self._conv2(model, z0)
            g = self._g_fc(model, self._fc(model, z1))
            if self._g_fc_frozen is not None:
                self._g_fc_frozen[i : i + xc.shape[0]] = g
            z0m = z0.mean(dim=(2, 3))
            a1 += g.double().transpose(0, 1) @ (yc.double().unsqueeze(1) * z0m.double())
            g_bar += g.double().sum(dim=0)
            if m0 is not None:
                rows = _loc_rows(z0)
                hw = int(rows.shape[0] // z0.shape[0])
                m0 += (rows.transpose(0, 1) @ rows).double()
        if m0 is not None:
            self._m0 = m0 / (n * hw)
        return a1 / n, g_bar / n

    @torch.no_grad()
    def _stats_conv_cols(
        self,
        model: SpectralModel,
        X: Tensor,
        y: Tensor,
        cols: Tensor,
        u1: Tensor,
        g_bar: Tensor,
    ) -> dict[str, Tensor]:
        """Per-column moments of the channel directions ``cols`` (K, p_c) on
        conv1's output (locations as samples): ``q2 = E[mean_x h²]``,
        ``lam = E[y mean_x h²]``, ``mom = E[y mean_x h]``, ``lam_s = E[y š
        mean_x h²]``, ``mom_s = E[y š mean_x h]`` with ``š = Uᵀ(g^FC − ḡ)``
        (r, K); the frozen scores ``s = Uᵀ g^FC`` (n, r); conv2's ``(d, e)``."""
        n = X.shape[0]
        k = cols.shape[0]
        r = u1.shape[1]
        dev = X.device
        q2 = torch.zeros(k, dtype=torch.float64, device=dev)
        lam = torch.zeros_like(q2)
        mom = torch.zeros_like(q2)
        lam_s = torch.zeros(r, k, dtype=torch.float64, device=dev)
        mom_s = torch.zeros_like(lam_s)
        scores = torch.zeros(n, r, dtype=X.dtype, device=dev)
        d_sum = e_sum = 0.0
        count = 0
        conv2 = cast(_Block, model.blocks[CONV2])
        for xc, yc, i in self._chunks(X, y):
            z0 = self._conv1(model, xc)
            h1, z1 = self._conv2(model, z0)
            g = self._g_fc(model, self._fc(model, z1))
            gp = _sigma_prime(h1, str(conv2.activation))
            d_sum += float(gp.double().sum())
            e_sum += float((h1.double() * gp.double()).sum())
            count += h1.numel()
            h = _coords(z0, cols)  # (c, K, H, W)
            h2m = h.double().pow(2).mean(dim=(2, 3))  # (c, K)
            h1m = h.double().mean(dim=(2, 3))
            yd = yc.double().unsqueeze(1)
            q2 += h2m.sum(dim=0)
            lam += (yd * h2m).sum(dim=0)
            mom += (yd * h1m).sum(dim=0)
            sc = g.double() @ u1  # (c, r)
            scores[i : i + xc.shape[0]] = sc.to(X.dtype)
            ys_c = (yd * (sc - (g_bar @ u1).unsqueeze(0))).transpose(0, 1)  # (r, c)
            lam_s += ys_c @ h2m
            mom_s += ys_c @ h1m
        return {
            "q2": q2 / n,
            "lam": lam / n,
            "mom": mom / n,
            "lam_s": lam_s / n,
            "mom_s": mom_s / n,
            "scores": scores,
            "d": torch.tensor(d_sum / max(count, 1), dtype=torch.float64),
            "e": torch.tensor(e_sum / max(count, 1), dtype=torch.float64),
        }

    @torch.no_grad()
    def _stats_fc(
        self, model: SpectralModel, X: Tensor, y: Tensor, extra: Tensor | None = None
    ) -> dict[str, Tensor]:
        """On the (updated) pooled conv2 output ``z̃¹``: ``q = E[z̄̃¹ (y s)ᵀ]``,
        ``m_bar = E[z̄̃¹ z̄̃¹ᵀ]``, ``m_loc = E[mean_x z̃¹ z̃¹ᵀ]``, ``c_y = E[y
        mean_x z̃¹ z̃¹ᵀ]``, ``m_y = E[y z̄̃¹]`` (float64) and the FC's ``(d, e)``."""
        n = X.shape[0]
        p_c = int(cast(Tensor, cast(_Block, model.blocks[CONV2]).weight).shape[0])
        r = self._scores.shape[1]
        dev = X.device
        q = torch.zeros(p_c, r, dtype=torch.float64, device=dev)
        # ``extra`` (n, m): stored scores of surviving paired columns of earlier
        # passes, re-evaluated on the same stream (upper_update="block").
        q_x = torch.zeros(
            p_c, 0 if extra is None else extra.shape[1], dtype=torch.float64, device=dev
        )
        m_bar = torch.zeros(p_c, p_c, dtype=torch.float64, device=dev)
        m_loc = torch.zeros_like(m_bar)
        c_y = torch.zeros_like(m_bar)
        m_y = torch.zeros(p_c, dtype=torch.float64, device=dev)
        d_sum = e_sum = 0.0
        count = 0
        hw = 0
        fc = cast(_Block, model.blocks[FC])
        for xc, yc, i in self._chunks(X, y):
            z0 = self._conv1(model, xc)
            _h1, z1 = self._conv2(model, z0)
            h2 = self._fc(model, z1)
            gp = _sigma_prime(h2, str(fc.activation))
            d_sum += float(gp.double().sum())
            e_sum += float((h2.double() * gp.double()).sum())
            count += h2.numel()
            z1m = z1.mean(dim=(2, 3))  # (c, p_c)
            ys = yc.unsqueeze(1) * self._scores[i : i + xc.shape[0]]
            q += z1m.double().transpose(0, 1) @ ys.double()
            if extra is not None:
                yx = yc.unsqueeze(1) * extra[i : i + xc.shape[0]]
                q_x += z1m.double().transpose(0, 1) @ yx.double()
            m_bar += (z1m.transpose(0, 1) @ z1m).double()
            rows = _loc_rows(z1)
            hw = int(rows.shape[0] // z1.shape[0])
            m_loc += (rows.transpose(0, 1) @ rows).double()
            y_rows = yc.repeat_interleave(hw).unsqueeze(1)
            c_y += (rows.transpose(0, 1) @ (y_rows * rows)).double()
            m_y += (yc.double().unsqueeze(1) * z1m.double()).sum(dim=0)
        return {
            "q": q / n,
            "q_extra": q_x / n,
            "m_bar": m_bar / n,
            "m_loc": m_loc / (n * hw),
            "c_y": c_y / (n * hw),
            "m_y": m_y / n,
            "d": torch.tensor(d_sum / max(count, 1), dtype=torch.float64),
            "e": torch.tensor(e_sum / max(count, 1), dtype=torch.float64),
        }

    def _fc_old_pairs(self) -> tuple[list[int], Tensor | None]:
        """Surviving paired slots of the FC from earlier passes and their stored
        scores ``(n, m)`` (``upper_update="block"``; else nothing)."""
        if not self._block_mode:
            return [], None
        src = self._slot_src[FC]
        slots = [i for i, t in enumerate(self._origin[FC]) if t == PAIR and i in src]
        if not slots:
            return [], None
        scores = torch.stack(
            [self._score_store[src[i][0]][:, src[i][1]] for i in slots], dim=1
        )
        return slots, scores

    @torch.no_grad()
    def _update_fc(
        self, model: SpectralModel, fs: dict[str, Tensor], old_slots: list[int]
    ) -> dict[str, float]:
        """Algorithm 3 on the FC from the S2 statistics of ``z̃¹``."""
        config: RowColConfig = self.config  # type: ignore[assignment]
        fc = cast(_Block, model.blocks[FC])
        red_f = cast(_Reduce, fc.reduce)
        tags = self._origin[FC]
        dev = red_f.V.device
        out: dict[str, float] = {}
        surv = [i for i, t in enumerate(tags) if t == FWD]
        if surv:
            cols = _cols(red_f).double()
            old = torch.cat([cols[surv].transpose(0, 1), self._ghost[FC]], dim=1)
            rcfg = model.block_config.blocks[FC].reduce
            new, out = refit_aligned(fs["c_y"], fs["m_y"], fs["m_loc"], old, rcfg)
            moved = (new[:, : len(surv)] - old[:, : len(surv)]).pow(2).sum()
            out["moved"] = float(moved) / len(surv)
            _set_cols(
                red_f,
                torch.tensor(surv, device=dev),
                new[:, : len(surv)].transpose(0, 1),
            )
            self._ghost[FC] = new[:, len(surv) :]
        out["n_forward"] = float(len(surv))
        out["n_refreshed"] = float(len(old_slots))
        if not old_slots:
            return out
        cols = _cols(red_f).double()
        q2 = torch.einsum("ap,pq,aq->a", cols, fs["m_loc"], cols)
        keep = torch.ones(cols.shape[0], dtype=torch.bool, device=dev)
        tau = _tau(q2, keep, tags, config.tau_over)
        w_fc = cast(Tensor, fc.weight)
        hw = int(w_fc.shape[0] // cols.shape[0])
        target = float(config.beta) * math.sqrt(int(w_fc.shape[1])) * tau
        target /= math.sqrt(hw)
        q = fs["q_extra"]
        rms = torch.einsum("pa,pq,qa->a", q, fs["m_bar"], q).sqrt().clamp_min(EPS)
        _set_cols(
            red_f,
            torch.tensor(old_slots, device=dev),
            (q * (target / rms).unsqueeze(0)).transpose(0, 1),
        )
        return out

    @torch.no_grad()
    def _chain_random(
        self, model: SpectralModel, g: Tensor, *, all_columns: bool = False
    ) -> Tensor:
        """Random branch of the chain for the FC signals ``g`` (m, p_f):
        ``V̄_FC[non] reshape(W_FC[non]ᵀ g / c_FC)`` as ``(m, p_c, H', W')`` maps
        at the pooled resolution (unpooled by :meth:`_unpool`).  Mean-field
        passes ``ḡ`` (m = 1), the exact chain the frozen per-sample signals."""
        fc = cast(_Block, model.blocks[FC])
        red_f = cast(_Reduce, fc.reduce)
        w_fc = cast(Tensor, fc.weight)
        cols_f = _cols(red_f).to(g.dtype)  # (K_f, p_c)
        k_f = cols_f.shape[0]
        hw = int(w_fc.shape[0] // k_f)
        tags = self._origin[FC]
        non = torch.tensor(
            [i for i, t in enumerate(tags) if all_columns or t != PAIR],
            device=w_fc.device,
        )
        w_non = w_fc.to(g.dtype).reshape(k_f, hw, -1)[non]  # (K_non, H'W', p_f)
        t = torch.einsum("axi,mi->max", w_non, g) / fc.c.to(g.dtype)  # (m,K_non,H'W')
        rand = torch.einsum("ap,max->mpx", cols_f[non], t)  # (m, p_c, H'W')
        side = int(math.isqrt(hw))
        return rand.reshape(g.shape[0], -1, side, side)  # (m, p_c, H', W')

    @torch.no_grad()
    def _first_order_q0(self) -> Tensor:
        """``q⁰`` (p_c, r1) of Approx. 6': the read-directions of the conv2|FC
        boundary lifted through the Phase-I conv2 (channel reduce, kernel
        summed over taps) and scaled so that ``RMS_μ⟨q⁰_i, z̄̃^conv2_μ⟩`` equals
        the FC column step's per-channel target."""
        ph = cast(SpectralModel, self._phase1)
        conv2 = cast(_Block, ph.blocks[CONV2])
        cols = _cols(cast(_Reduce, conv2.reduce))  # (K_c, p_c)
        w = cast(Tensor, conv2.weight).sum(dim=(2, 3))  # (p_c, K_c): taps summed
        v = self._v_read.to(cols.dtype)  # (p_c, r1)
        q0 = ((w @ (cols @ v)) / conv2.c).double()  # (p_c, r1)
        rms = torch.einsum("pa,pq,qa->a", q0, self._m_bar, q0).sqrt()
        return q0 * (self._target_f / rms.clamp_min(EPS)).unsqueeze(0)

    @staticmethod
    def _unpool(model: SpectralModel, r: Tensor, z: Tensor) -> Tensor:
        """Adjoint of conv2's pool applied to the pooled-resolution maps ``r``
        (c, p_c, H', W') given the pre-pool activation map ``z`` (c, p_c, H, W):
        avg-pool spreads each value over its cell / (s·s); max-pool routes it
        to the cell's arg-max location (per sample and channel)."""
        pool = cast(Any, cast(_Block, model.blocks[CONV2]).pool)
        k, s = int(pool.kernel_size), int(pool.stride)
        if pool.mode.strip().lower() == "avg":
            return F.interpolate(r, scale_factor=s, mode="nearest") / float(s * s)
        _pooled, idx = F.max_pool2d(z, kernel_size=k, stride=s, return_indices=True)
        return F.max_unpool2d(r, idx, kernel_size=k, stride=s, output_size=z.shape[2:])

    @torch.no_grad()
    def _stats_boundary0(self, model: SpectralModel, X: Tensor, y: Tensor) -> Tensor:
        """``A₀ = E[y g^conv2_x z⁰_xᵀ]`` (locations as samples, float64) with the
        mean-field chained ``g^conv2``."""
        n = X.shape[0]
        conv2 = cast(_Block, model.blocks[CONV2])
        fc = cast(_Block, model.blocks[FC])
        p_c = int(cast(Tensor, conv2.weight).shape[0])
        frozen = self._g_fc_frozen  # (n, p_f) for the exact chain, else None
        first_order = self._phase1 is not None
        rand = (
            self._chain_random(
                cast(SpectralModel, self._phase1 if first_order else model),
                self._g_bar.to(X.dtype).unsqueeze(0),
                all_columns=first_order,
            )
            if frozen is None
            else None
        )  # mean-field / first-order: one (1, p_c, H', W') map, pooled resolution
        q_fc = (self._first_order_q0() if first_order else self._q_fc).to(X.dtype)
        c_fc = float(fc.c)
        a0 = torch.zeros(p_c, p_c, dtype=torch.float64, device=X.device)
        hw = 0
        for xc, yc, i in self._chunks(X, y):
            z0 = self._conv1(model, xc)
            h1, _z1 = self._conv2(model, z0)
            act = ACTIVATIONS[conv2.activation](h1) / math.sqrt(p_c)  # pre-pool map
            gate = _sigma_prime(h1, str(conv2.activation)) / math.sqrt(p_c)
            pair = (self._scores[i : i + xc.shape[0]] @ q_fc.transpose(0, 1)) / c_fc
            rand_c = (
                rand.expand(xc.shape[0], -1, -1, -1)
                if rand is not None
                else self._chain_random(model, frozen[i : i + xc.shape[0]])
            )
            r_pooled = rand_c + pair.unsqueeze(2).unsqueeze(3)  # (c, p_c, H', W')
            g2 = gate * self._unpool(model, r_pooled, act)
            g_rows = _loc_rows(g2)
            z_rows = _loc_rows(z0)
            hw = int(z_rows.shape[0] // z0.shape[0])
            y_rows = yc.repeat_interleave(hw).unsqueeze(1)
            a0 += (g_rows.transpose(0, 1) @ (y_rows * z_rows)).double()
        return a0 / (n * hw)

    # ------------------------------------------------------------------ #
    # Importance
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _omega_bar(self, model: SpectralModel) -> Tensor:
        """``ω̄`` (p_c, p_f): conv2 filter ``j``'s location-summed effective
        weight into the FC neurons, ``V_FCᵀ Σ_x W_FC[(·, x), :] / c_FC``."""
        fc = cast(_Block, model.blocks[FC])
        cols_f = _cols(cast(_Reduce, fc.reduce)).double()  # (K_f, p_c)
        w_fc = cast(Tensor, fc.weight).double()
        k_f = cols_f.shape[0]
        w_sum = w_fc.reshape(k_f, -1, w_fc.shape[1]).sum(dim=1)  # (K_f, p_f)
        return cols_f.transpose(0, 1) @ w_sum / fc.c.double()

    @torch.no_grad()
    def _importance_conv(
        self, model: SpectralModel, st: dict[str, Tensor], norms: Tensor
    ) -> Tensor:
        """Importance of conv2's reduce columns from their moments ``st`` (raw
        coordinates; normalised here) and the retained FC signal."""
        conv2 = cast(_Block, model.blocks[CONV2])
        w = cast(Tensor, conv2.weight).double()  # (p_c, K, kh, kw)
        norms = norms.clamp_min(EPS)
        lam = st["lam"] / norms**2
        mom = st["mom"] / norms
        lam_s = st["lam_s"] / norms.unsqueeze(0) ** 2
        mom_s = st["mom_s"] / norms.unsqueeze(0)
        om = self._omega_bar(model)  # (p_c, p_f)
        og = om @ self._g_bar  # (p_c,)
        ou = om @ self._u1.double()  # (p_c, r)
        t2 = og.unsqueeze(1) * lam.unsqueeze(0) + ou @ lam_s  # (p_c, K)
        t1 = og.unsqueeze(1) * mom.unsqueeze(0) + ou @ mom_s
        coef = w.flatten(2) * norms.reshape(1, -1, 1) / conv2.c.double()  # (p_c,K,T)
        rate = st["e"] * t2.unsqueeze(2) * coef + st["d"] * t1.unsqueeze(2)
        return rate.pow(2).mean(dim=(0, 2))

    @torch.no_grad()
    def _importance_fc(
        self,
        model: SpectralModel,
        lam: Tensor,
        mom: Tensor,
        norms: Tensor,
        d: Tensor,
        e: Tensor,
    ) -> Tensor:
        """Importance of the FC's reduce columns (top block: ``ω = a``, one lift
        coefficient per (neuron, location))."""
        fc = cast(_Block, model.blocks[FC])
        w = cast(Tensor, fc.weight).double()  # (K_f·H'W', p_f)
        k_f = norms.shape[0]
        coef = (
            w.reshape(k_f, -1, w.shape[1]).permute(2, 0, 1)
            * norms.reshape(1, -1, 1)
            / fc.c.double()
        )  # (p_f, K_f, H'W')
        a_out = self._readout_grad(model).double().reshape(-1, 1, 1)
        rate = a_out * (e * lam.reshape(1, -1, 1) * coef + d * mom.reshape(1, -1, 1))
        return rate.pow(2).mean(dim=(0, 2))
