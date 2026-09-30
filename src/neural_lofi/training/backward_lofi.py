"""Backward-corrected spectral training (backward Neural LoFi).

Extends the reduce-first :class:`~neural_lofi.training.spectral.SpectralTrainer`
with a *backward* correction pass, operating on an **already-fitted**
:class:`~neural_lofi.models.spectral.SpectralModel`.  Forward LoFi is greedy:
each block's reduction ``V`` keeps the directions whose (first- and
second-order) statistics correlate with the label in that block's own input
geometry, blind to whether the blocks above can transmit those directions to
the output.  In gradient descent that information arrives through the
backpropagated signal; backward LoFi computes its leading spectral surrogate
and feeds it back as *effective labels* for a corrected sweep.

Derivation (one paragraph)
--------------------------
Under the same approximations that yield forward LoFi, the one-step change of
the backpropagated signal reaching block ``j`` is governed by the singular
value decomposition of a small cross-block matrix::

    B_j = (1/N) ztilde_{j+1}^T (y ⊙ g_j)   ∈ R^{P_{j+1} × k_j},

where ``ztilde_{j+1} = pool(σ'(W_{j+1} r_{j+1} / c_{j+1}) / √P_{j+1})`` is the
*derivative feature map* of the block above (the forward recipe with ``σ'`` in
place of ``σ``, pooling applied identically) and ``g_j`` are block ``j``'s
reduced expand-input coordinates.  This is the compact matrix
``B^{*} = A_← V`` of the paper's *Full forward–backward Neural LoFi cycle*
(the cross-layer driver restricted through the forward spikes).  Each of the
``k_bwd`` retained left singular directions ``u_i`` (optionally led by the
constant direction ``u_0 ∝ E[y · ztilde]``) defines a per-sample score
``s_i(x) = ⟨ztilde(x), u_i⟩`` and an effective label ``ytilde_i = y ⊙ s_i`` —
the label modulated by a function that is simple in the derivative geometry of
the block above.  Re-fitting block ``j``'s reduction with these modulated
labels selects features whose label correlation is *transmissible* — visible
through derivative directions the next block actually uses.

Contract and phases
-------------------
The trainer takes a **fitted** model (Phase I is whatever
:class:`SpectralTrainer` already produced) and runs:

Phase II   Per boundary ``j`` (blocks ``0 … L-2`` whose reduce is not the
           identity): stream the frozen source model, accumulate ``B_j`` (and
           optionally the constant direction), thin-SVD, keep the top
           ``k_bwd`` left singular vectors as backward estimators.
Phase III  Corrected sweep on a **widened** copy of the model, following the
           paper's algorithm: block ``j``'s reduce grows from its fitted
           width ``keep`` to ``k_total`` (default: the full union width
           ``keep + k_bwd · k_per_estimator``), and its reduction is re-fit
           as the ordered Gram–Schmidt union ``orth_{k_total}[base | est_0 |
           est_1 | …]`` — the base block at its original rank followed by,
           per estimator, a full block of ``k_per_estimator``
           signed-covariance eigenvectors of ``E[ytilde_i z zᵀ]``
           (``k_per_estimator`` defaults to the block's forward cutoff, as in
           the paper; a linear column ``E[ytilde_i z]/‖·‖`` is prepended per
           estimator when ``backward_linear_spike`` is on).  Everything else
           (c-norms, whitening, vector dispatch, ridge readout) is the
           inherited streaming block fit.

Phases II+III may be iterated (``n_cycles``): the model is widened **once**,
then each extra cycle re-runs II+III on the previous cycle's corrected model,
propagating corrections one block further per cycle.

Streaming design
----------------
Nothing of size ``N`` is stored.  Phase II accumulates ``B_j`` batch-by-batch;
scores are **recomputed per batch from the frozen source model** during Phase
III (so shuffling loaders cannot misalign labels and scores).  For scalar
labels the base + backward label streams ride the per-class vector-covariance
accumulator as extra label columns (one accumulation pass); vector labels run
one pass per stream.  Conv boundary statistics treat **locations as samples**,
matching the forward covariance's per-location bias: ``g`` is average-pooled
onto ``ztilde``'s spatial grid and each location contributes
``y · ztilde(loc) ḡ(loc)ᵀ`` to ``B_j``.  Scores stay per-sample scalars via
the spatial mean of ``ztilde`` (the score is linear in ``ztilde``, so this is
the mean location score), and the corrected reduction fit itself uses the
standard patch-flattening — each spatial location inherits its sample's
modulated label, exactly like the forward vector path.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..models.block_config import BlockedModelConfig, ReduceConfig
from ..models.spectral import SpectralModel, _Block, _Reduce, _reduce_keep
from ..utils.helpers import ACTIVATION_DERIVATIVES
from ..utils.structured.hadamard import next_pow2
from ..utils.transforms import (
    DenseConvTransform,
    DenseTransform,
    SORFConvTransform,
    SORFTransform,
    StructuredTransform,
)
from .eigen import (
    alternating_vector_eigen_transverse_from_covariance,
    covariance_chunk_signed_vector_accumulate_,
    flatten_spatial_vector,
    linear_mean_prepend_block_from_covariance,
    welford_cov_accumulate_,
)
from .spectral import SpectralTrainer, SpectralTrainerConfig

log = logging.getLogger(__name__)

# Numerical floor for score RMS normalization / degenerate constant direction,
# and the Gram-Schmidt residual threshold for dropping dependent columns.
SCORE_RMS_EPS = 1e-12
GS_RESIDUAL_EPS = 1e-6

__all__ = [
    "BackwardBoundary",
    "BackwardSpectralConfig",
    "BackwardSpectralTrainer",
    "backward_directions",
    "block_derivative_features",
    "max_backward_directions",
    "orthonormal_union",
    "widen_spectral_model",
]


def max_backward_directions(
    k_fwd: int, n_outputs: int = 1, *, include_constant_direction: bool = False
) -> int:
    """Backward directions a boundary can supply above a reduce of ``k_fwd``.

    The boundary matrix ``B`` is ``(P_{j+1}, C·k_fwd)``, so its thin SVD has at
    most ``C·k_fwd`` left singular directions; the constant direction
    ``u_0 ∝ E[y·ztilde]`` is a separate rank-one object and adds one more when
    kept.  ``k_bwd`` is clamped to this cap per boundary.
    """
    return k_fwd * n_outputs + (1 if include_constant_direction else 0)


# ---------------------------------------------------------------------- #
# Pure helpers (unit-testable without a trainer)
# ---------------------------------------------------------------------- #


@torch.no_grad()
def block_derivative_features(block: _Block, a_prev: Tensor) -> Tensor:
    """Derivative feature map ``ztilde`` of one block.

    The block's forward recipe with ``σ'`` in place of ``σ``::

        ztilde = pool( σ'(W r / c) / √P ),   r = expand_input(a_prev)

    Pooling (max or avg) is applied to the derivative map exactly as the
    forward applies it to the activation map — the derivative features are
    treated as a feature map in their own right, mirroring how the forward
    reduction fit consumes the pooled block output.

    Parameters
    ----------
    block : _Block
        The block whose derivative map to compute (the block *above* the
        corrected boundary).
    a_prev : Tensor
        The block's wide input ``a_{ℓ-1}`` — ``(N, D)`` flat or
        ``(N, C, H, W)`` image features.

    Returns
    -------
    Tensor
        Derivative features, same shape family as the block's forward output.
    """
    try:
        sigma_prime = ACTIVATION_DERIVATIVES[block.activation]
    except KeyError:
        raise ValueError(
            f"Activation {block.activation!r} has no registered derivative; "
            f"backward LoFi supports {sorted(ACTIVATION_DERIVATIVES)}. "
            "('step' is excluded: σ' = 0 a.e. gives a degenerate backward "
            "signal; 'relu_normalized' is not elementwise.)"
        ) from None
    r = block.expand_input(a_prev)
    transform = cast(StructuredTransform, block.transform)
    z = sigma_prime(transform.project(r) / block.c)
    z = z / math.sqrt(block.out_features)
    if block.pool is not None:
        z = block.pool.apply(z)
    return z


def _per_sample(t: Tensor) -> Tensor:
    """Per-sample feature rows: spatial-mean a ``(N, P, H, W)`` map to ``(N, P)``.

    Flat features pass through.  Used for the per-sample *scores* only
    (scores must be scalars to modulate labels; the score is linear in
    ``ztilde``, so the mean-map score equals the mean location score).
    """
    return t.mean(dim=(2, 3)) if t.ndim == 4 else t


def _boundary_rows(zt: Tensor, g: Tensor, yb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Row-aligned ``(ztilde, g, y)`` for the boundary accumulation.

    Conv derivative maps treat **locations as samples** — the same bias as the
    forward covariance: ``g`` is average-pooled onto ``ztilde``'s spatial grid
    (each coarse cell is the mean of the fine ``g`` cells under it; a flat
    ``g`` broadcasts to every location) and both maps flatten locations into
    rows in the :func:`flatten_spatial_vector` order, each row inheriting its
    sample's label.  A flat ``ztilde`` with a conv ``g`` spatial-means ``g``.
    """
    if zt.ndim == 4:
        n, p, h, w = zt.shape
        zt_flat = zt.permute(0, 2, 3, 1).reshape(n * h * w, p)
        if g.ndim == 4:
            g_al = nn.functional.adaptive_avg_pool2d(g, (h, w))
        else:
            g_al = g[:, :, None, None].expand(n, g.shape[1], h, w)
        g_flat = g_al.permute(0, 2, 3, 1).reshape(n * h * w, g_al.shape[1])
        y_flat = yb.unsqueeze(1).expand(n, h * w, yb.shape[1]).reshape(n * h * w, -1)
        return zt_flat, g_flat, y_flat
    return zt, (g.mean(dim=(2, 3)) if g.ndim == 4 else g), yb


@torch.no_grad()
def left_singular(b: Tensor) -> tuple[Tensor, Tensor]:
    """Left singular vectors and singular values of ``b`` (``m × n``).

    Wide matrices (``n`` beyond ``max(4m, 4096)``) go through the ``m × m``
    Gram ``b bᵀ`` — one GEMM plus a small ``eigh`` — instead of a thin SVD,
    which cuSOLVER rejects past ~2³¹ elements (a flattened conv→fc boundary
    at large ``k`` is ``P × k·H·W``).  Only the leading directions are
    consumed downstream, so the squared conditioning is immaterial; narrow
    matrices keep the exact SVD path.
    """
    m, n = b.shape
    if n <= max(4 * m, 4096):
        u, s, _ = torch.linalg.svd(b, full_matrices=False)
        return u, s
    evals, evecs = torch.linalg.eigh(b @ b.transpose(0, 1))
    order = torch.argsort(evals, descending=True)
    return evecs[:, order], evals[order].clamp_min(0).sqrt()


@torch.no_grad()
def backward_directions(
    m_hat: Tensor,
    b_mat: Tensor,
    k_bwd: int,
    *,
    include_constant_direction: bool = False,
    block_idx: int = -1,
) -> tuple[Tensor, float, Tensor]:
    """Backward directions from the accumulated boundary moments.

    Implements the paper's compact backward spectral step on pre-accumulated
    statistics (both divided by the sample count):

    * ``m_hat`` — the constant-direction moment ``E[ztilde yᵀ]``, shape
      ``(P, C)`` (``C = 1`` for scalar labels).  Its leading left singular
      vector is the constant direction ``u_0`` (for scalar labels exactly
      ``E[y·ztilde]/‖·‖``) and its leading singular value is ``rho0``.
    * ``b_mat`` — the boundary matrix, per-class blocks stacked along columns:
      ``B = [B_0 | … | B_{C-1}]`` with ``B_c = E[y_c · ztilde gᵀ]``, shape
      ``(P, C·k)``.  Its top left singular vectors are the backward directions
      (the "stacked SVD" vector-label generalization).

    ``k_bwd`` counts the total directions kept: with
    ``include_constant_direction`` that is ``u_0`` plus the top ``k_bwd - 1``
    left singular vectors, else the top ``k_bwd`` left singular vectors.

    Returns ``(directions (P, k_bwd), rho0, singular_values)``.  When the
    constant moment is degenerate it is replaced by an extra SVD direction
    (warning).
    """
    if k_bwd < 1:
        raise ValueError(f"k_bwd must be >= 1, got {k_bwd}")

    u_m, s_m, _ = torch.linalg.svd(m_hat, full_matrices=False)
    rho0 = float(s_m[0].item()) if s_m.numel() else 0.0

    columns: list[Tensor] = []
    if include_constant_direction:
        if rho0 > SCORE_RMS_EPS:
            columns.append(u_m[:, 0])
        else:
            log.warning(
                "Boundary %d: ‖E[y·ztilde]‖=%.2e is degenerate; replacing the "
                "constant backward direction with an extra SVD direction.",
                block_idx,
                rho0,
            )

    u_svd, s_svd = left_singular(b_mat)
    n_svd = k_bwd - len(columns)
    n_avail = u_svd.shape[1]
    if n_svd > n_avail:
        log.warning(
            "Boundary %d: requested %d SVD directions but B has rank <= %d; "
            "keeping %d.",
            block_idx,
            n_svd,
            n_avail,
            n_avail,
        )
        n_svd = n_avail
    columns.extend(u_svd[:, :n_svd].T.unbind(dim=0))

    return torch.stack(columns, dim=1), rho0, s_svd


@torch.no_grad()
def _informativeness(cols: Tensor, covs: list[Tensor]) -> Tensor:
    """``max_s |vᵀ C_s v|`` per column of ``cols`` over the stream covariances.

    The label streams are the base label and the RMS-normalized effective
    labels, so the quadratic forms are on a common scale; a direction scores
    by the most label-correlated energy it captures under any of them.
    """
    if not covs:
        raise ValueError("no stream covariances available for scoring")
    best: Tensor | None = None
    for cov in covs:
        c = cols.to(device=cov.device, dtype=cov.dtype)
        q = ((c.transpose(0, 1) @ cov) * c.transpose(0, 1)).sum(dim=1).abs()
        best = q if best is None else torch.maximum(best, q)
    assert best is not None
    return best.to(cols.device)


@torch.no_grad()
def orthonormal_union(
    blocks: list[Tensor],
    total: int,
    *,
    eps: float = GS_RESIDUAL_EPS,
) -> tuple[Tensor, list[int]]:
    """Ordered Gram-Schmidt union of column blocks, truncated to ``total``.

    Concatenates the blocks *in priority order* and orthonormalizes column by
    column (with re-orthogonalization for float stability), dropping any
    column whose residual norm falls below ``eps`` — i.e. a column already
    (numerically) spanned by higher-priority columns contributes nothing.
    Stops once ``total`` columns are collected; zero-pads (with a warning) if
    the union has smaller rank, so downstream shape contracts hold.

    Returns ``(V, kept_per_block)`` where ``V`` is ``(D, total)`` and
    ``kept_per_block[j]`` counts the columns of block ``j`` that made it in.
    """
    if not blocks:
        raise ValueError("orthonormal_union requires at least one block")
    d = blocks[0].shape[0]
    kept: list[Tensor] = []
    kept_per_block = [0 for _ in blocks]

    for j, block in enumerate(blocks):
        if block.numel() == 0:
            continue
        if block.shape[0] != d:
            raise ValueError(f"Block {j} has {block.shape[0]} rows, expected {d}")
        for col in block.T:
            if len(kept) >= total:
                break
            v = col.clone()
            # Two rounds of classical Gram-Schmidt ≈ modified GS stability.
            for _ in range(2):
                for q in kept:
                    v = v - q * (q @ v)
            v_norm = v.norm()
            if float(v_norm) <= eps:
                continue
            kept.append(v / v_norm)
            kept_per_block[j] += 1
        if len(kept) >= total:
            break

    if len(kept) < total:
        log.warning(
            "orthonormal_union: only %d independent columns available for a "
            "%d-column basis; zero-padding the remainder.",
            len(kept),
            total,
        )
        pad = [blocks[0].new_zeros(d) for _ in range(total - len(kept))]
        kept = kept + pad

    return torch.stack(kept, dim=1), kept_per_block


# ---------------------------------------------------------------------- #
# Model widening
# ---------------------------------------------------------------------- #


@torch.no_grad()
def widen_spectral_model(
    model: SpectralModel, extra: Sequence[int], *, seed: int = 0
) -> SpectralModel:
    """Deepcopy ``model`` and widen block ``j``'s reduce by ``extra[j]`` columns.

    The input model is never touched.  Per widened block the reduce ``V``
    grows from its fitted width ``keep`` to ``keep + extra[j]`` (fresh
    placeholder — the trainer re-fits it) and the block's expand grows on the
    input side to read the new coordinates:

    * dense (FC / conv): the **trained columns are kept** and fresh
      ``randn`` rows / input-channels are appended, drawn from a
      ``seed``-seeded generator.  The append is exact also across a
      ``flatten`` pre-op (channel-major layout puts new channels in a
      trailing contiguous chunk).
    * SORF: the effective transform pads the input to ``d_pad``, so as long
      as the widened width still fits the same ``d_pad`` the trained
      transform is **exactly preserved** (the new coordinates occupy former
      zero-pad slots); only when the widened width crosses the next power of
      two is the transform rebuilt with a fresh seed (warning — trained
      random features not preserved for that block).

    ``model.block_config`` is updated (``reduce.k += extra[j]``) so
    checkpointing and shape validation stay consistent.
    """
    if len(extra) != len(model.blocks):
        raise ValueError(
            f"extra must have one entry per block ({len(model.blocks)}), "
            f"got {len(extra)}"
        )
    widened = copy.deepcopy(model)
    gen = torch.Generator()
    gen.manual_seed(seed)

    def _fresh_seed() -> int:
        return int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())

    new_blocks_cfg = list(widened.block_config.blocks)
    for j, delta in enumerate(extra):
        if delta == 0:
            continue
        if delta < 0:
            raise ValueError(f"extra[{j}] must be >= 0, got {delta}")
        bcfg = new_blocks_cfg[j]
        if bcfg.reduce.k is None:
            raise ValueError(
                f"Block {j} has an identity reduce (k=null) — there is no "
                "reduction to widen/correct at this block."
            )
        block = cast(_Block, widened.blocks[j])
        old_reduce = cast(_Reduce, block.reduce)
        old_keep = cast(int, _reduce_keep(bcfg.reduce))
        v = old_reduce.V
        in_width = int(v.shape[1] if old_reduce.is_conv else v.shape[0])
        new_keep = old_keep + delta
        if new_keep > in_width:
            raise ValueError(
                f"Block {j}: widened reduction would keep {new_keep} directions "
                f"but the incoming features have only {in_width}."
            )
        device = v.device
        block.reduce = _Reduce(
            in_width=in_width,
            keep=new_keep,
            is_conv=old_reduce.is_conv,
            whiten=old_reduce.whiten,
        ).to(device)

        transform = block.transform
        old_in = int(cast(StructuredTransform, transform).in_features)
        if old_in % old_keep != 0:
            raise RuntimeError(
                f"Block {j}: expand input width {old_in} is not a multiple of "
                f"the reduce width {old_keep} — cannot infer the widening layout."
            )
        if isinstance(transform, DenseTransform):
            # factor = h·w across a flatten pre-op, else 1.
            factor = old_in // old_keep
            w_old = cast(Tensor, transform.weight_view).detach()
            fresh = torch.randn(delta * factor, w_old.shape[1], generator=gen)
            new_w = nn.Parameter(
                torch.cat([w_old, fresh.to(w_old)], dim=0), requires_grad=False
            )
            block.weight = new_w
            block.transform = DenseTransform(new_w)
        elif isinstance(transform, DenseConvTransform):
            w_old = cast(Tensor, transform.weight_view).detach()
            fresh = torch.randn(
                w_old.shape[0], delta, w_old.shape[2], w_old.shape[3], generator=gen
            )
            new_w = nn.Parameter(
                torch.cat([w_old, fresh.to(w_old)], dim=1), requires_grad=False
            )
            block.weight = new_w
            block.transform = DenseConvTransform(
                new_w, padding=transform.padding, stride=transform.stride
            )
        elif isinstance(transform, SORFConvTransform):
            patch_new = new_keep * transform.kernel_size**2
            if next_pow2(patch_new) == transform.sorf.d_pad:
                # Same padded width: the effective transform is unchanged; the
                # new channels simply occupy former zero-pad slots.
                transform.in_channels = new_keep
                transform.in_features = patch_new
                transform.sorf.in_features = patch_new
            else:
                log.warning(
                    "Block %d: SORF conv expand rebuilt at widened patch width "
                    "%d (d_pad grows) — trained random features of this block "
                    "are not preserved.",
                    j,
                    patch_new,
                )
                block.transform = SORFConvTransform(
                    new_keep,
                    transform.out_channels,
                    kernel_size=transform.kernel_size,
                    padding=transform.padding,
                    stride=transform.stride,
                    seed=_fresh_seed(),
                ).to(device)
        elif isinstance(transform, SORFTransform):
            factor = old_in // old_keep
            new_in = new_keep * factor
            if next_pow2(new_in) == transform.d_pad:
                transform.in_features = new_in
            else:
                log.warning(
                    "Block %d: SORF expand rebuilt at widened width %d (d_pad "
                    "grows) — trained random features of this block are not "
                    "preserved.",
                    j,
                    new_in,
                )
                block.transform = SORFTransform(
                    new_in, transform.out_features, seed=_fresh_seed()
                ).to(device)
        else:
            raise TypeError(
                f"Block {j}: unsupported transform {type(transform).__name__}"
            )
        new_blocks_cfg[j] = replace(
            bcfg, reduce=replace(bcfg.reduce, k=bcfg.reduce.k + delta)
        )

    widened.block_config = BlockedModelConfig(
        blocks=tuple(new_blocks_cfg),
        final_reduce=widened.block_config.final_reduce,
    )
    return widened


# ---------------------------------------------------------------------- #
# Config + boundary record + trainer
# ---------------------------------------------------------------------- #


@dataclass
class BackwardBoundary:
    """Backward correction extracted at one adjacent-block boundary.

    ``block_idx`` is the block ``j`` whose reduce is corrected (the derivative
    features come from block ``j+1``).  ``directions`` is ``(P_{j+1}, m)``;
    ``inv_rms`` the per-direction score normalizers (ones when normalization
    is off); ``rho0`` / ``singular_values`` / ``score_rms`` are diagnostics.
    """

    block_idx: int
    directions: Tensor
    inv_rms: Tensor
    rho0: float
    singular_values: Tensor
    score_rms: Tensor


@dataclass
class BackwardSpectralConfig(SpectralTrainerConfig):
    """Configuration for :class:`BackwardSpectralTrainer`.

    Parameters
    ----------
    k_bwd
        The paper's backward cutoff ``k_ℓ^←`` — how many backward
        directions/estimators each boundary keeps: the top left singular
        vectors of the boundary matrix ``B = E[y ztilde gᵀ]`` (led by the
        constant direction ``u_0`` when ``include_constant_direction`` is on;
        ``u_0`` counts toward ``k_bwd``).  ``0`` deactivates the boundary.
        Scalar (broadcast to every boundary) or a per-boundary list of length
        ``n_blocks - 1`` (boundary ``j`` corrects block ``j``'s reduce using
        block ``j+1``'s derivative features).  Clamped per boundary to
        :func:`max_backward_directions`.  With a scalar value, boundaries
        whose reduce is the identity (``k=null``) are skipped with a log
        message; with an explicit list a nonzero entry on an identity reduce
        raises.
    k_per_estimator
        Columns each estimator's effective label contributes to the corrected
        reduction — eigenvectors of ``E[ytilde_i z zᵀ]`` (plus one linear
        column when ``backward_linear_spike`` is on).  ``None`` (default)
        follows the paper: the block's own forward width (its base reduce
        keep).  Scalar or per-boundary list.
    k_total
        The paper's optional total rank budget ``k̄_ℓ`` for the corrected
        reduce.  ``None`` (default) keeps the full union width
        ``keep + k_bwd · (k_per_estimator + spike)``; an integer widens the
        reduce to exactly ``k_total`` and lets the ordered Gram–Schmidt union
        fill it greedily (base block first).  Must be ``>= `` the base keep;
        values above the full union are clamped down.  Scalar or per-boundary
        list.
    combine_mode
        ``"add"`` (default): the corrected reduce **widens** from its base
        keep to ``k_total`` — backward columns are strictly additional.
        ``"swap"``: the corrected reduce keeps its **original width** ``k``;
        ``swap_fraction`` of its columns are backward, the rest the leading
        forward eigenvectors.  Per boundary the budget is
        ``round(swap_fraction · k)`` columns, divided as evenly as possible
        over ``min(k_bwd, budget)`` estimators (per-estimator quotas), and
        the assembly order is ``[top (k − budget) base | est_0 | … |
        remaining base as backfill]`` truncated to exactly ``k`` — a
        Gram-Schmidt-dropped stream column backfills with the next base
        eigenvector, so the corrected model degrades gracefully toward the
        plain forward fit.  ``k_per_estimator`` / ``k_total`` are ignored
        (and must be left ``None``); ``backward_linear_spike`` is
        unsupported in swap mode.
    swap_fraction
        Fraction of each corrected reduce's columns replaced by backward
        directions (swap mode only).  Scalar in ``(0, 1]`` or a per-boundary
        list.
    cycle_rule
        How a corrected reduce is assembled across cycles (swap mode, scalar
        labels).  ``"refresh"`` (default, the paper): every cycle recomputes
        the forward eigenvectors on the corrected prefix and replaces the
        backward block with directions from the previous cycle's model.
        ``"accumulate"``: from cycle 2 on, the previous cycle's basis is
        carried, its ``budget`` least-informative columns are dropped to make
        room for the new backward block, so backward columns stack across
        cycles.  ``"merged"``: every cycle pools the forward eigenvectors, the
        new backward candidates and (from cycle 2) the carried basis, ranks
        them all by informativeness and keeps the top ``k`` — the backward
        fraction is emergent (at most ``swap_fraction``).  Informativeness of
        a direction ``v`` is ``max_s |vᵀ C_s v|`` over the current cycle's
        label streams (base + effective labels; comparable because scores
        are RMS-normalized).  Every layer entry reports ``origin_counts``.
    include_constant_direction
        Lead the estimator set with ``u_0`` from ``E[ztilde yᵀ]`` (for scalar
        labels ``E[y·ztilde]/‖·‖``) — the backward analogue of the linear
        spike.  Off by default (quadratic-only convention).  In swap mode
        ``u_0`` counts as the first estimator (co-author convention).
    backward_linear_spike
        Prepend each estimator's normalized first moment ``E[ytilde_i z]/‖·‖``
        to its eigenvector block (package prepend convention: on top of
        ``k_per_estimator``, mirroring ``include_linear_mean``).  Off by
        default (quadratic-only convention).
    normalize_scores
        RMS-normalize the score columns before forming effective labels (the
        downstream eigenproblems are scale-invariant; this stabilizes
        degenerate-moment thresholds and makes diagnostics comparable).
    n_cycles
        Backward + corrected-sweep iterations.  The model is widened once;
        cycle ``c`` uses cycle ``c-1``'s corrected model as the source, so
        depth-``c`` credit assignment needs ``c`` cycles.
    store_backward_diagnostics
        Record per-cycle boundary diagnostics under ``results["backward"]``.
    gs_eps
        Gram-Schmidt residual threshold for dropping dependent columns.
    widen_seed
        Seed for the fresh random expand columns drawn while widening.
    """

    k_bwd: int | list[int] = 1
    k_per_estimator: int | list[int] | None = None
    k_total: int | list[int] | None = None
    combine_mode: str = "add"
    swap_fraction: float | list[float] | None = None
    cycle_rule: str = "refresh"
    include_constant_direction: bool = False
    backward_linear_spike: bool = False
    normalize_scores: bool = True
    n_cycles: int = 1
    store_backward_diagnostics: bool = True
    gs_eps: float = GS_RESIDUAL_EPS
    widen_seed: int = 0
    # Scalar-label corrected fits accumulate base + backward streams as label
    # columns of the per-class covariance accumulator; at large m the
    # (1+m, P, P) buffer dominates memory, so streams are processed in groups
    # of at most this many per data pass.  The per-class covariances are exactly
    # grouping-independent; the cross-moment is one GEMM whose column count
    # changes with the group, so results agree to float32 rounding (~1e-7).
    max_streams_per_pass: int = 16


class BackwardSpectralTrainer(SpectralTrainer):
    """Backward-corrected sweep on an already-fitted :class:`SpectralModel`.

    ``fit`` widens a deepcopy of the input model (the input is never
    modified), extracts the backward directions per boundary from the frozen
    source model (Phase II), and re-runs the inherited streaming block fit
    with the extra label streams injected into each corrected reduction
    (Phase III).  Returns ``(corrected_model, results)`` where ``results``
    keeps the plain trainer's ``{"layers", "final"}`` shape (describing the
    corrected model) plus a ``"backward"`` diagnostics entry.

    With ``k_bwd=0`` (or a model with no correctable boundary) the fit
    degenerates to a plain re-fit of an unwidened copy — numerically identical
    to the input model, which doubles as a consistency check.
    """

    def __init__(
        self,
        model: SpectralModel,
        config: BackwardSpectralConfig,
    ) -> None:
        super().__init__(model, config)
        if not isinstance(config, BackwardSpectralConfig):
            raise TypeError(
                "BackwardSpectralTrainer requires a BackwardSpectralConfig, "
                f"got {type(config).__name__}"
            )
        if config.n_cycles < 1:
            raise ValueError(f"n_cycles must be >= 1, got {config.n_cycles}")
        if config.combine_mode not in ("add", "swap"):
            raise ValueError(
                f"combine_mode must be 'add' or 'swap', got {config.combine_mode!r}"
            )
        if config.combine_mode == "swap":
            if config.swap_fraction is None:
                raise ValueError("swap mode requires swap_fraction")
            if config.k_per_estimator is not None or config.k_total is not None:
                raise ValueError(
                    "swap mode sets the budget from swap_fraction — leave "
                    "k_per_estimator and k_total as None"
                )
            if config.backward_linear_spike:
                raise ValueError("backward_linear_spike is unsupported in swap mode")
        elif config.swap_fraction is not None:
            raise ValueError("swap_fraction requires combine_mode='swap'")
        if config.cycle_rule not in ("refresh", "accumulate", "merged"):
            raise ValueError(
                "cycle_rule must be 'refresh', 'accumulate' or 'merged', "
                f"got {config.cycle_rule!r}"
            )
        if config.cycle_rule != "refresh" and config.combine_mode != "swap":
            raise ValueError(f"cycle_rule={config.cycle_rule!r} requires swap mode")
        # Per-fit state: boundary records + the frozen source model whose
        # derivative features define the scores of the current cycle.
        self._boundaries: dict[int, BackwardBoundary] = {}
        self._source_model: SpectralModel | None = None
        self._orig_reduce_cfgs: dict[int, ReduceConfig] = {}
        self._k_bwd: dict[int, int] = {}
        self._k_pe: dict[int, int] = {}
        self._k_total: dict[int, int] = {}
        # Swap mode: forward columns kept up front + per-estimator quotas.
        self._swap_keep: dict[int, int] = {}
        self._quotas: dict[int, list[int]] = {}
        # Cycle state: current cycle index and, per boundary, the previous
        # cycle's final basis with its per-column lineage ("forward"/"backward").
        self._cycle = 0
        self._prev_basis: dict[int, Tensor] = {}
        self._prev_origin: dict[int, list[str]] = {}

    # ------------------------------------------------------------------ #
    # Budget bookkeeping
    # ------------------------------------------------------------------ #

    def _resolve_backward_budget(
        self, n_outputs: int
    ) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
        """Resolve the per-boundary backward budget.

        Returns ``(kbwd_map, kpe_map, ktot_map)`` over *active* boundaries
        only.  Boundary ``j`` (``0 … L-2``) corrects block ``j``'s reduce; it
        is active when ``k_bwd[j] > 0`` and the reduce is not the identity.
        ``k_bwd`` is clamped to :func:`max_backward_directions`;
        ``k_per_estimator`` defaults to the block's base keep (the paper's
        ``k_ℓ``); ``k_total`` defaults to the full union width and is bounded
        by ``[base keep, full union]``.  Fails fast when block ``j+1``'s
        activation has no registered derivative.
        """
        model: SpectralModel = self.model  # type: ignore[assignment]
        config: BackwardSpectralConfig = self.config  # type: ignore[assignment]
        n_boundaries = len(model.blocks) - 1

        def _resolve(
            v: int | list[int] | None, name: str
        ) -> tuple[list[int | None], bool]:
            if v is None:
                return [None] * n_boundaries, False
            if isinstance(v, int):
                return [v] * n_boundaries, False
            vals: list[int | None] = [int(x) for x in v]
            if len(vals) != n_boundaries:
                raise ValueError(
                    f"{name} list must have n_blocks-1={n_boundaries} entries "
                    f"(one per boundary), got {len(vals)}"
                )
            return vals, True

        kbwds, kbwd_explicit = _resolve(config.k_bwd, "k_bwd")
        kpes, _ = _resolve(config.k_per_estimator, "k_per_estimator")
        ktots, _ = _resolve(config.k_total, "k_total")
        spike = 1 if config.backward_linear_spike else 0
        swap = config.combine_mode == "swap"
        fracs: list[float | None] = [None] * n_boundaries
        if swap:
            f = config.swap_fraction
            fracs = (
                [float(x) for x in f]  # type: ignore[union-attr]
                if isinstance(f, (list, tuple))
                else [float(cast(float, f))] * n_boundaries
            )
            if len(fracs) != n_boundaries:
                raise ValueError(
                    f"swap_fraction list must have n_blocks-1={n_boundaries} "
                    f"entries, got {len(fracs)}"
                )
            for j, fj in enumerate(fracs):
                if fj is not None and not 0.0 < fj <= 1.0:
                    raise ValueError(f"swap_fraction[{j}] must be in (0, 1], got {fj}")
        self._swap_keep = {}
        self._quotas = {}

        kbwd_map: dict[int, int] = {}
        kpe_map: dict[int, int] = {}
        ktot_map: dict[int, int] = {}
        for j in range(n_boundaries):
            kb = cast(int, kbwds[j])
            if kb < 0:
                raise ValueError(f"k_bwd[{j}] must be >= 0, got {kb}")
            if kb == 0:
                continue
            # Width of g_j (the boundary matrix's column index) — the reduce's
            # keep, which for a linear_svd reduce is k + linear_k.  A `pre`
            # flatten can only widen g further, so the cap stays conservative.
            k_fwd = _reduce_keep(model.block_config.blocks[j].reduce)
            if k_fwd is None:
                if kbwd_explicit:
                    raise ValueError(
                        f"Boundary {j}: block {j} has an identity reduce "
                        "(k=null) — there is no reduction to correct.  Set "
                        f"k_bwd[{j}]=0."
                    )
                log.info("Block %d reduce is the identity — boundary %d skipped.", j, j)
                continue
            act = model.block_config.blocks[j + 1].expand.activation
            if act not in ACTIVATION_DERIVATIVES:
                raise ValueError(
                    f"Boundary {j}: block {j + 1} activation {act!r} has no "
                    "registered derivative; backward LoFi supports "
                    f"{sorted(ACTIVATION_DERIVATIVES)}."
                )
            cap = max_backward_directions(
                k_fwd,
                n_outputs,
                include_constant_direction=config.include_constant_direction,
            )
            if kb > cap:
                log.warning(
                    "Boundary %d: k_bwd=%d exceeds the %d directions the "
                    "boundary matrix can supply — clamping to %d.",
                    j,
                    kb,
                    cap,
                    cap,
                )
                kb = cap
            if swap:
                budget = int(round(cast(float, fracs[j]) * k_fwd))
                budget = max(1, min(budget, k_fwd))
                m_eff = min(kb, budget)
                base_q, rem = divmod(budget, m_eff)
                quotas = [base_q + 1] * rem + [base_q] * (m_eff - rem)
                kbwd_map[j] = m_eff
                kpe_map[j] = base_q  # nominal; per-stream quotas rule
                ktot_map[j] = k_fwd
                self._swap_keep[j] = k_fwd - budget
                self._quotas[j] = quotas
                continue
            kpe = kpes[j] if kpes[j] is not None else k_fwd
            kpe = cast(int, kpe)
            if kpe < 1:
                raise ValueError(f"k_per_estimator[{j}] must be >= 1, got {kpe}")
            full = k_fwd + kb * (kpe + spike)
            kt = ktots[j] if ktots[j] is not None else full
            kt = cast(int, kt)
            if kt < k_fwd:
                raise ValueError(
                    f"Boundary {j}: k_total={kt} is below the base keep "
                    f"{k_fwd} — the corrected reduce cannot shrink."
                )
            if kt > full:
                log.warning(
                    "Boundary %d: k_total=%d exceeds the full union width %d "
                    "— clamping to %d.",
                    j,
                    kt,
                    full,
                    full,
                )
                kt = full
            kbwd_map[j] = kb
            kpe_map[j] = kpe
            ktot_map[j] = kt
        return kbwd_map, kpe_map, ktot_map

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Run the backward correction; return ``(corrected_model, results)``.

        ``self.model`` is the **fitted input model** on entry and the widened
        corrected model on exit (the input model itself is never modified).
        """
        test_loader = cast("DataLoader | None", kwargs.get("test_loader"))
        input_model: SpectralModel = self.model  # type: ignore[assignment]
        config: BackwardSpectralConfig = self.config  # type: ignore[assignment]

        if float(input_model.readout_weight.detach().abs().sum().item()) == 0.0:
            log.warning(
                "Input model has an all-zero readout — it looks unfitted.  "
                "BackwardSpectralTrainer expects a model already fitted by "
                "SpectralTrainer (its reductions define the Phase-I features)."
            )

        # The direction cap depends on the label width (B has C·k_fwd columns).
        _is_vec, n_outputs, _one_hot = self._peek_label_shape(loader)
        kbwd_map, kpe_map, ktot_map = self._resolve_backward_budget(n_outputs)
        if not kbwd_map:
            log.warning(
                "No active backward boundaries (zero budget, single-block "
                "model, or identity reduces only) — re-running the plain "
                "spectral fit on an unwidened copy."
            )
            self.model = copy.deepcopy(input_model)
            model, plain_results = self._fit_blocked(loader, test_loader=test_loader)
            plain_results["backward"] = {
                "k_bwd": [],
                "k_per_estimator": [],
                "k_total": [],
                "n_cycles": 0,
                "cycles": [],
            }
            return model, plain_results

        # Per boundary the reduce grows from its base keep to k_total.
        extra = [
            (
                ktot_map[j]
                - cast(int, _reduce_keep(input_model.block_config.blocks[j].reduce))
                if j in ktot_map
                else 0
            )
            for j in range(len(input_model.blocks))
        ]
        corrected = widen_spectral_model(input_model, extra, seed=config.widen_seed)
        self._orig_reduce_cfgs = {
            j: input_model.block_config.blocks[j].reduce for j in kbwd_map
        }
        self._k_bwd, self._k_pe, self._k_total = kbwd_map, kpe_map, ktot_map
        self.model = corrected

        if config.verbose:
            log.info(
                "Backward spectral fit: %d blocks, boundaries %s "
                "(k_bwd=%s, k_per_estimator=%s, k_total=%s), cycles=%d",
                len(input_model.blocks),
                sorted(kbwd_map),
                kbwd_map,
                kpe_map,
                ktot_map,
                config.n_cycles,
            )

        cycles_log: list[dict[str, Any]] = []
        results: dict[str, Any] = {}
        self._prev_basis, self._prev_origin = {}, {}
        try:
            for cycle in range(config.n_cycles):
                self._cycle = cycle
                source = input_model if cycle == 0 else copy.deepcopy(corrected)
                self._source_model = source
                self._boundaries = self._backward_sweep(source, loader)
                last = cycle == config.n_cycles - 1
                _, results = self._fit_blocked(
                    loader, test_loader=test_loader if last else None
                )
                if config.store_backward_diagnostics:
                    cycles_log.append(self._cycle_diagnostics(cycle, results))
        finally:
            self._boundaries = {}
            self._source_model = None

        n_boundaries = len(input_model.blocks) - 1
        results["backward"] = {
            "k_bwd": [kbwd_map.get(j, 0) for j in range(n_boundaries)],
            "k_per_estimator": [kpe_map.get(j, 0) for j in range(n_boundaries)],
            "k_total": [ktot_map.get(j, 0) for j in range(n_boundaries)],
            "combine_mode": config.combine_mode,
            "swap_fraction": config.swap_fraction,
            "cycle_rule": config.cycle_rule,
            "swap_keep_forward": [self._swap_keep.get(j) for j in range(n_boundaries)],
            "estimator_quotas": [self._quotas.get(j) for j in range(n_boundaries)],
            "include_constant_direction": config.include_constant_direction,
            "backward_linear_spike": config.backward_linear_spike,
            "n_cycles": config.n_cycles,
            "cycles": cycles_log,
        }
        return corrected, results

    # ------------------------------------------------------------------ #
    # Phase II: backward sweep over boundaries (streaming)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _backward_sweep(
        self, source: SpectralModel, loader: DataLoader
    ) -> dict[int, BackwardBoundary]:
        """Extract backward directions at every active boundary.

        Boundary ``j`` couples only blocks ``j`` and ``j+1`` of the frozen
        source model, so the iterations are mutually independent.  Two
        streaming passes per boundary: one accumulating ``E[ztilde yᵀ]`` and
        the stacked boundary matrix, one measuring the score RMS of the
        retained directions.
        """
        config: BackwardSpectralConfig = self.config  # type: ignore[assignment]
        dev = torch.device(config.device)

        out: dict[int, BackwardBoundary] = {}
        for j in sorted(self._k_bwd):
            k_bwd = self._k_bwd[j]
            block_j = cast(_Block, source.blocks[j])
            block_above = cast(_Block, source.blocks[j + 1])

            m_hat: Tensor | None = None
            b_acc: Tensor | None = None
            count = 0
            for x, y in loader:
                x_dev = x.to(dev, non_blocking=True)
                a_prev = source.forward_to_block(x_dev, stop=j)
                g_map = block_j.expand_input(a_prev)
                zt_map = block_derivative_features(block_above, block_j(a_prev))
                yb = y.to(dev, non_blocking=True).to(torch.float32)
                yb = yb if yb.ndim == 2 else yb.unsqueeze(1)
                # Conv boundaries: locations as samples (g pooled to zt's grid).
                zt, g, yr = _boundary_rows(zt_map, g_map, yb)
                if m_hat is None:
                    p, k, c = zt.shape[1], g.shape[1], yr.shape[1]
                    m_hat = torch.zeros(p, c, dtype=torch.float64, device=dev)
                    b_acc = torch.zeros(p, c * k, dtype=torch.float64, device=dev)
                assert b_acc is not None
                n_b = zt.shape[0]
                # Per-class blocks y_c ⊙ g, stacked along columns → (n, C·k).
                yg = (yr.unsqueeze(2) * g.unsqueeze(1)).reshape(n_b, -1)
                m_hat += (zt.transpose(0, 1) @ yr).double()
                b_acc += (zt.transpose(0, 1) @ yg).double()
                count += n_b
            if m_hat is None or b_acc is None or count == 0:
                raise RuntimeError(f"Boundary {j}: empty loader, cannot fit.")

            directions, rho0, svals = backward_directions(
                m_hat / count,
                b_acc / count,
                k_bwd,
                include_constant_direction=config.include_constant_direction,
                block_idx=j,
            )
            directions = directions.to(torch.float32)

            # Second pass: score RMS (normalization + diagnostics).
            ssq = torch.zeros(directions.shape[1], dtype=torch.float64, device=dev)
            n_scores = 0
            for x, _y in loader:
                a_j = source.forward_to_block(x.to(dev, non_blocking=True), stop=j + 1)
                zt = _per_sample(block_derivative_features(block_above, a_j))
                s = zt @ directions
                ssq += s.double().pow(2).sum(dim=0)
                n_scores += s.shape[0]
            rms = (ssq / n_scores).sqrt().clamp_min(SCORE_RMS_EPS).to(torch.float32)
            inv_rms = 1.0 / rms if config.normalize_scores else torch.ones_like(rms)

            out[j] = BackwardBoundary(
                block_idx=j,
                directions=directions,
                inv_rms=inv_rms,
                rho0=rho0,
                singular_values=svals.to(torch.float32).cpu(),
                score_rms=rms.cpu(),
            )
            if config.verbose:
                top_sv = svals[: min(3, svals.numel())].cpu().numpy()
                log.info(
                    "  boundary %d→%d: rho0=%.4e, top singular values=%s",
                    j + 1,
                    j,
                    rho0,
                    top_sv,
                )
        return out

    def _boundary_scores(self, boundary: BackwardBoundary, x_dev: Tensor) -> Tensor:
        """Per-sample backward scores ``(n, m)`` from the frozen source model.

        Recomputed per batch (never stored per sample), so a shuffling loader
        can never misalign scores and labels.
        """
        source = self._source_model
        assert source is not None
        j = boundary.block_idx
        a_j = source.forward_to_block(x_dev, stop=j + 1)
        zt = _per_sample(
            block_derivative_features(cast(_Block, source.blocks[j + 1]), a_j)
        )
        return (zt @ boundary.directions) * boundary.inv_rms

    # ------------------------------------------------------------------ #
    # Phase III: corrected reduction fit (hooks into the inherited block fit)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _fit_reduction_on_block(
        self, ell: int, rcfg: Any, rmod: _Reduce, loader: DataLoader
    ) -> dict[str, Any]:
        """Inject backward label streams when the fitted reduce has a boundary.

        ``_fit_blocked`` fits block ``j``'s reduce in the call with
        ``ell = j - 1`` (and the final reduction with ``ell = L - 1``, which
        never has a boundary entry) — everything without an active boundary
        falls through to the plain inherited fit.
        """
        boundary = self._boundaries.get(ell + 1)
        if boundary is None:
            return super()._fit_reduction_on_block(ell, rcfg, rmod, loader)
        return self._fit_corrected_reduction(ell, rcfg, rmod, loader, boundary)

    @torch.no_grad()
    def _fit_corrected_reduction(
        self,
        ell: int,
        rcfg: Any,
        rmod: _Reduce,
        loader: DataLoader,
        boundary: BackwardBoundary,
    ) -> dict[str, Any]:
        """Fit one corrected reduction from base labels + backward streams.

        Streams block ``ell``'s wide output through the (partially re-fitted)
        corrected model while the estimator scores stream from the frozen source
        model; fits the base reduction at its original rank, then per estimator
        a block of ``k_per_estimator`` eigenvectors of ``E[ytilde_i z zᵀ]``
        (led by the linear column ``E[ytilde_i z]/‖·‖`` when
        ``backward_linear_spike`` is on), and assembles the corrected
        projection by ordered Gram–Schmidt ``[base | est_0 | est_1 | …]``
        truncated to ``k_total``.
        """
        model: SpectralModel = self.model  # type: ignore[assignment]
        config: BackwardSpectralConfig = self.config  # type: ignore[assignment]
        dev = torch.device(config.device)
        j = boundary.block_idx
        base_rcfg = self._orig_reduce_cfgs[j]
        assert base_rcfg.k is not None  # active boundaries never have identity reduces
        base_k: int = base_rcfg.k
        m = int(boundary.directions.shape[1])
        n_lin = 1 if config.backward_linear_spike else 0
        n_quad = self._k_pe[j]
        # Swap mode: per-estimator column quotas + forward columns kept up front.
        quotas = self._quotas.get(j)
        swap_keep = self._swap_keep.get(j)
        total_keep = cast(int, _reduce_keep(rcfg))

        def _stream_quota(s: int) -> int:
            """Columns stream ``s`` (1-based estimator index) contributes."""
            return quotas[s - 1] if quotas is not None else n_quad

        probe: Tensor | None = None
        for x, _y in loader:
            probe = model.forward_to_block(x.to(dev, non_blocking=True), stop=ell + 1)
            break
        if probe is None:
            raise RuntimeError(f"Block {ell}: empty loader, cannot fit.")
        is_conv = probe.ndim == 4
        p = probe.shape[1]
        out_shape = tuple(probe.shape[1:])

        # One accumulator device decision for covariances + whitening moments,
        # mirroring the parent fit (they always share a device there).
        n_class_slots = (
            self._n_outputs
            if self._is_vector
            else min(1 + m, max(1, int(config.max_streams_per_pass)))
        )
        accum_dev = self._accumulator_device(p, n_class_slots)
        s_cov = mu_z = None
        if rcfg.whiten:
            s_cov = torch.zeros(p, p, dtype=torch.float32, device=accum_dev)
            mu_z = torch.zeros(p, dtype=torch.float32, device=accum_dev)

        def _batch(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            """→ (wide features a, base labels (n, C), scores (n, m))."""
            x_dev = x.to(dev, non_blocking=True)
            a = model.forward_to_block(x_dev, stop=ell + 1)
            yb = y.to(dev, non_blocking=True).to(torch.float32)
            yb = yb if yb.ndim == 2 else yb.unsqueeze(1)
            return a, yb, self._boundary_scores(boundary, x_dev)

        accum_dt = self._accum_dtype()
        blocks_cols: list[Tensor]
        lin_svals: Tensor | None = None
        linear_k = 0
        # accumulate/merged score candidate directions on the current cycle's
        # stream covariances (base + effective labels), kept from the last pass.
        rule = config.cycle_rule if swap_keep is not None else "refresh"
        stream_covs: list[Tensor] = []

        if self._is_vector:
            if rule != "refresh":
                raise NotImplementedError(
                    f"cycle_rule={rule!r} is implemented for scalar labels only"
                )
            # One accumulation pass per stream (base + m backward streams):
            # each stream's effective label is itself a (n, C) vector.
            c_out = self._n_outputs
            c_stack = torch.zeros(c_out, p, p, dtype=accum_dt, device=accum_dev)
            m_acc = torch.zeros(p, c_out, dtype=accum_dt, device=accum_dev)
            blocks_cols = []
            eigvals = torch.empty(0)
            for stream in range(1 + m):
                c_stack.zero_()
                m_acc.zero_()
                total = 0
                for x, y in loader:
                    a, yb, scores = _batch(x, y)
                    labels = yb if stream == 0 else yb * scores[:, stream - 1 : stream]
                    if is_conv:
                        zf, lf, _ = flatten_spatial_vector(a, labels)
                    else:
                        zf, lf = a, labels
                    if stream == 0 and s_cov is not None and mu_z is not None:
                        welford_cov_accumulate_(s_cov, mu_z, total, zf)
                    # Effective labels are soft even for one-hot bases.
                    total = covariance_chunk_signed_vector_accumulate_(
                        c_stack,
                        m_acc,
                        zf,
                        lf,
                        total,
                        assume_one_hot=self._is_one_hot if stream == 0 else False,
                    )
                if total == 0:
                    raise RuntimeError(f"Block {ell}: no data accumulated.")
                if stream == 0:
                    linear_k = base_rcfg.linear_k or c_out
                    _a_star, _u, lin_svals, eigvals, v_cols, _traj = (
                        alternating_vector_eigen_transverse_from_covariance(
                            c_stack,
                            m_acc,
                            base_k,
                            linear_k=linear_k,
                            max_iter=config.inner_max_iter,
                            tol=config.inner_tol,
                            patience=config.inner_patience,
                        )
                    )
                    blocks_cols.append(v_cols.to(dtype=torch.float32, device=dev))
                else:
                    blocks_cols.append(
                        self._stream_columns_vector(
                            c_stack, m_acc, n_lin, _stream_quota(stream)
                        ).to(dev)
                    )
        else:
            # Scalar labels: base + backward streams ride the per-class vector
            # accumulator as extra label columns.  Streams are processed in
            # groups of ``max_streams_per_pass`` (one data pass per group) so
            # the accumulator memory stays bounded at large m.  Per-class
            # covariances are exactly grouping-independent (one GEMM each); the
            # cross-moment is a single GEMM whose column count is the group
            # size, so it agrees only to float32 rounding (~1e-7).
            n_streams = 1 + m
            group = max(1, int(config.max_streams_per_pass))
            if rule != "refresh" and group < n_streams:
                raise ValueError(
                    f"cycle_rule={rule!r} needs every stream covariance at once: "
                    f"set max_streams_per_pass >= {n_streams}"
                )
            n_slots = min(n_streams, group)
            c_stack = torch.zeros(n_slots, p, p, dtype=accum_dt, device=accum_dev)
            m_acc = torch.zeros(p, n_slots, dtype=accum_dt, device=accum_dev)
            blocks_cols = []
            eigvals = torch.empty(0)
            total = 0
            for g0 in range(0, n_streams, group):
                g1 = min(g0 + group, n_streams)
                c_grp = c_stack[: g1 - g0]
                m_grp = m_acc[:, : g1 - g0]
                c_grp.zero_()
                m_grp.zero_()
                total = 0
                for x, y in loader:
                    a, yb, scores = _batch(x, y)
                    cols = [
                        yb if s == 0 else yb * scores[:, s - 1 : s]
                        for s in range(g0, g1)
                    ]
                    labels = torch.cat(cols, dim=1)
                    if is_conv:
                        zf, lf, _ = flatten_spatial_vector(a, labels)
                    else:
                        zf, lf = a, labels
                    if g0 == 0 and s_cov is not None and mu_z is not None:
                        welford_cov_accumulate_(s_cov, mu_z, total, zf)
                    total = covariance_chunk_signed_vector_accumulate_(
                        c_grp, m_grp, zf, lf, total, assume_one_hot=False
                    )
                if total == 0:
                    raise RuntimeError(f"Block {ell}: no data accumulated.")
                for s in range(g0, g1):
                    cov_s = 0.5 * (c_grp[s - g0] + c_grp[s - g0].transpose(0, 1))
                    u_s = m_grp[:, s - g0] / total
                    if s == 0:
                        eigvals, base_cols = linear_mean_prepend_block_from_covariance(
                            cov_s,
                            u_s,
                            k=base_k,
                            include_linear_mean=base_rcfg.include_linear_mean,
                            orthogonalize=base_rcfg.orthogonalize_to_mean,
                            target_dtype=torch.float32,
                            target_device=dev,
                            log_prefix=f"Block {ell} corrected reduce (base): ",
                        )
                        blocks_cols.append(base_cols)
                    else:
                        blocks_cols.append(
                            self._stream_columns_scalar(
                                cov_s, u_s, n_lin, _stream_quota(s)
                            )
                        )
                    if rule != "refresh":
                        stream_covs.append(cov_s)

        origins: list[str] | None = None
        if swap_keep is not None:
            base_all = blocks_cols[0]
            est_cols = blocks_cols[1:]
            budget = total_keep - swap_keep
            prev = self._prev_basis.get(j) if self._cycle > 0 else None
            prev_origin = self._prev_origin.get(j, [])
            if rule == "refresh" or (rule == "accumulate" and prev is None):
                # Top base eigenvectors first, then the estimator blocks, then
                # the remaining base columns as backfill — width stays exactly
                # ``total_keep`` (= the uncorrected reduce width).
                blocks_cols = [
                    base_all[:, :swap_keep],
                    *est_cols,
                    base_all[:, swap_keep:],
                ]
                tags = ["forward", *(["backward"] * len(est_cols)), "forward"]
            elif rule == "accumulate":
                # Carry the previous basis; its ``budget`` least-informative
                # columns make room for the new backward block and backfill.
                order = torch.argsort(
                    _informativeness(prev, stream_covs), descending=True
                )
                n_keep = max(total_keep - budget, 0)
                keep_idx, drop_idx = order[:n_keep], order[n_keep:]
                blocks_cols = [
                    prev[:, keep_idx],
                    *est_cols,
                    prev[:, drop_idx],
                    base_all,
                ]
                tags = [None, *(["backward"] * len(est_cols)), None, "forward"]
                per_col = [
                    [prev_origin[int(i)] for i in keep_idx],
                    [prev_origin[int(i)] for i in drop_idx],
                ]
            else:  # merged: rank every candidate column, one block each.
                cands = [base_all, *est_cols] + ([prev] if prev is not None else [])
                cand_tags = (
                    ["forward"] * base_all.shape[1]
                    + ["backward"] * sum(c.shape[1] for c in est_cols)
                    + (list(prev_origin) if prev is not None else [])
                )
                all_cols = torch.cat(cands, dim=1)
                order = torch.argsort(
                    _informativeness(all_cols, stream_covs), descending=True
                )
                blocks_cols = [all_cols[:, i : i + 1] for i in order.tolist()]
                tags = [cand_tags[i] for i in order.tolist()]
        v_union, kept = orthonormal_union(blocks_cols, total_keep, eps=config.gs_eps)
        if swap_keep is not None:
            # Lineage of every final column (Gram-Schmidt preserves block order).
            origins = []
            if rule == "accumulate" and prev is not None:
                blk_origin = [per_col[0]] + [None] * len(est_cols) + [per_col[1], None]
                for b_i, (n_kept, tag) in enumerate(zip(kept, tags, strict=True)):
                    src = blk_origin[b_i]
                    origins += src[:n_kept] if src is not None else [tag] * n_kept
            else:
                for n_kept, tag in zip(kept, tags, strict=True):
                    origins += [cast(str, tag)] * n_kept
            origins += ["forward"] * (total_keep - len(origins))  # zero-pad columns
            self._prev_basis[j] = v_union.detach().clone()
            self._prev_origin[j] = origins
        self._store_reduction(rmod, rcfg, v_union, s_cov, mu_z, total)

        entry: dict[str, Any] = {
            "block": ell,
            "p": p,
            "k": total_keep,
            "corrected": True,
            "boundary_block": j,
            "k_bwd": m,
            "k_per_estimator": n_quad,
            "backward_linear_spike": bool(n_lin),
            "basis_columns_per_stream": kept,
            "eigenvalues": eigvals.cpu().numpy().tolist(),
            "train_output_shape": out_shape,
        }
        if swap_keep is not None:
            entry["combine_mode"] = "swap"
            entry["swap_keep_forward"] = swap_keep
            entry["estimator_quotas"] = list(cast(list[int], quotas))
            entry["cycle"] = self._cycle
            entry["cycle_rule"] = rule
            assert origins is not None
            n_bwd = origins.count("backward")
            entry["origin_counts"] = {
                "forward": len(origins) - n_bwd,
                "backward": n_bwd,
            }
            entry["backward_fraction"] = n_bwd / total_keep
        if lin_svals is not None:
            entry["linear_svd"] = True
            entry["linear_k"] = linear_k
            entry["linear_singular_values"] = lin_svals.cpu().numpy().tolist()
        if config.verbose:
            log.info(
                "  block %d reduce corrected: k=%d, streams kept %s",
                j,
                total_keep,
                kept,
            )
        return entry

    def _stream_columns_scalar(
        self, cov: Tensor, u: Tensor, n_lin: int, n_quad: int
    ) -> Tensor:
        """One estimator's columns, scalar labels: ``n_lin`` + ``n_quad``.

        The linear term is the estimator's normalized first moment
        ``E[ytilde z]/‖·‖`` (``n_lin`` is 0 or 1); the quadratic terms are the
        top ``n_quad`` eigenvectors of ``E[ytilde z zᵀ]``, deflated against the
        linear direction when it is kept.
        """
        dev = torch.device(self.config.device)
        if n_quad == 0:
            # Linear only (n_lin == 1; the 0/0 case never reaches here).
            u_norm = float(u.norm().item())
            if u_norm > SCORE_RMS_EPS:
                return (u / u_norm).to(dtype=torch.float32, device=dev).unsqueeze(1)
            # Degenerate moment: fall back to the top eigenvector.
            _, cols = linear_mean_prepend_block_from_covariance(
                cov,
                u,
                k=1,
                include_linear_mean=False,
                orthogonalize=False,
                target_dtype=torch.float32,
                target_device=dev,
            )
            return cols
        _, cols = linear_mean_prepend_block_from_covariance(
            cov,
            u,
            k=n_quad,
            include_linear_mean=n_lin > 0,
            orthogonalize=n_lin > 0,
            target_dtype=torch.float32,
            target_device=dev,
            log_prefix="Backward estimator: ",
        )
        return cols

    def _stream_columns_vector(
        self, c_stack: Tensor, m_raw: Tensor, n_lin: int, n_quad: int
    ) -> Tensor:
        """One estimator's columns, vector labels: ``n_lin`` + ``n_quad``.

        The effective label ``ytilde_i = y ⊙ s_i`` is itself a ``C``-vector, so
        its cross-moment admits up to ``C`` linear directions; this convention
        keeps at most the leading one, so each estimator contributes exactly
        ``n_lin + n_quad`` columns in both the scalar and vector paths.  Add
        estimators, not per-estimator linear directions.

        The alternating solver defines its transverse eigenvectors by
        deflation against the linear block (``linear_k >= 1`` structurally),
        so with the spike off (``n_lin = 0``) it still runs at ``linear_k=1``
        and the linear column is dropped from the returned block.
        """
        config: BackwardSpectralConfig = self.config  # type: ignore[assignment]
        if n_quad == 0:
            u_svd, _, _ = torch.linalg.svd(m_raw, full_matrices=False)
            return cast(Tensor, u_svd[:, :1]).to(torch.float32)
        _a, _u, _sv, _ev, v_cols, _traj = (
            alternating_vector_eigen_transverse_from_covariance(
                c_stack,
                m_raw,
                n_quad,
                linear_k=max(n_lin, 1),
                max_iter=config.inner_max_iter,
                tol=config.inner_tol,
                patience=config.inner_patience,
            )
        )
        if n_lin == 0:
            v_cols = v_cols[:, 1:]
        return v_cols.to(torch.float32)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def _cycle_diagnostics(self, cycle: int, results: dict[str, Any]) -> dict[str, Any]:
        """JSON-serializable summary of one backward + corrected cycle."""
        return {
            "cycle": cycle,
            "boundaries": [
                {
                    "block": b.block_idx,
                    "rho0": b.rho0,
                    "singular_values": b.singular_values.numpy().tolist(),
                    "score_rms": b.score_rms.numpy().tolist(),
                    "k_bwd": int(b.directions.shape[1]),
                }
                for _, b in sorted(self._boundaries.items())
            ],
            "basis_columns_per_stream": [
                entry.get("basis_columns_per_stream")
                for entry in results.get("layers", [])
                if entry.get("corrected")
            ],
            "final": {
                k: v
                for k, v in results.get("final", {}).items()
                if isinstance(v, (int, float))
            },
        }
