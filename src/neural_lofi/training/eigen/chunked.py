"""Spatial reshaping and chunked signed-covariance accumulation.

Conv layers form the signed covariance over spatial patches without ever
materialising the ``(N·H·W, P)`` patch matrix: each mini-batch contributes an
unnormalised ``(P, P)`` partial covariance that the trainer accumulates and
normalises once.  This module holds the patch-flattening layouts (scalar and
vector labels), the per-chunk accumulators, and the chunk-size heuristic.
Leaf module: depends only on the covariance primitive.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ...utils.covariance import signed_covariance_matrix


def flatten_spatial(z: Tensor, y: Tensor) -> tuple[Tensor, Tensor, int]:
    """Collapse conv-output spatial dims into the sample dim.

    Input ``z`` has shape ``(N, P, H, W)`` and ``y`` shape ``(N,)`` or
    ``(N, 1)``. Returns ``(z_flat, y_flat, spatial)`` where ``z_flat`` has
    shape ``(N*H*W, P)``, ``y_flat`` broadcasts each sample's label across
    its ``H*W`` patches, and ``spatial = H*W``.

    The permutation order ``(N, P, H, W) → (N, H, W, P) → (N*H*W, P)`` is
    fixed: it mirrors the layout assumed by the legacy CNN trainer when
    forming the signed covariance over spatial patches.
    """
    n, p, h_out, w_out = z.shape
    spatial = h_out * w_out
    y_f = y.to(z.dtype).view(-1)
    z_flat = z.permute(0, 2, 3, 1).reshape(n * spatial, p)
    y_flat = y_f.unsqueeze(1).expand(n, spatial).reshape(n * spatial)
    return z_flat, y_flat, spatial


@torch.no_grad()
def covariance_chunk_signed(z: Tensor, y: Tensor) -> tuple[Tensor, Tensor, int]:
    """Unnormalised signed covariance + label-weighted mean from one chunk.

    Used by the CNN trainer's chunked-covariance path to accumulate
    ``Σ y_i Z_i^T Z_i`` across mini-batches without materialising the full
    ``(N, P, H, W)`` tensor. Inputs and layout match :func:`flatten_spatial`.

    Returns ``(C_part, u_part, n_patches)`` where ``C_part`` has shape
    ``(P, P)``, ``u_part`` has shape ``(P,)``, both unnormalised; the
    caller divides by the total accumulated ``n_patches`` to obtain the
    per-patch covariance and label-weighted mean.
    """
    z_flat, y_flat, _ = flatten_spatial(z, y)
    c_part = signed_covariance_matrix(z_flat, y_flat, normalize=False)
    u_part = (z_flat * y_flat.unsqueeze(1)).sum(dim=0)
    return c_part, u_part, z_flat.shape[0]


@torch.no_grad()
def covariance_chunk_signed_accumulate_(
    cov: Tensor, u: Tensor, z: Tensor, y: Tensor, total_old: int
) -> int:
    """Fold one chunk into the **running mean** signed moments (float64).

    Streaming, allocation-free analog of :func:`covariance_chunk_signed`, but
    the accumulators ``cov (P, P)`` and ``u (P,)`` hold the running *mean*
    ``(1/total) Σ y z zᵀ`` / ``(1/total) Σ y z`` — divide-as-you-go rather than
    sum-then-divide-once, so the buffers never hold a large raw sum.  Given the
    prior sample count ``total_old`` and this chunk's ``new = z.shape[0]`` rows,
    the update is the standard incremental mean

        total_new = total_old + new
        mean      ← mean · (total_old / total_new) + S / total_new

    with ``S`` the raw chunk sum.  Returns ``total_new`` (the caller threads it
    back in as the next ``total_old``).

    ``z`` is a **2-D row block** ``(M, P)`` — FC features directly, or conv
    patches already flattened via :func:`flatten_spatial` — and ``y`` is
    ``(M,)`` / ``(M, 1)``.  The per-chunk product ``zᵀ diag(y) z`` and the
    label-weighted sum ``Σ y z`` are formed in ``z``'s dtype (cheap ``float32``
    GPU GEMM) and folded into the accumulators via ``.to(cov)`` / ``.to(u)``,
    which casts to the accumulator dtype **and** moves to its device (so the
    accumulator may live on CPU while the GEMM runs on GPU).

    The contribution is **not** symmetrised (matching the linear summation of
    ``Σ y z zᵀ``); the caller symmetrises the final ``cov`` once before the
    eigensolve, exactly as :func:`signed_covariance_matrix` does by default.
    """
    new = z.shape[0]
    total_new = total_old + new
    scale_old = total_old / total_new
    inv_new = 1.0 / total_new
    y_f = y.to(z.dtype).view(-1)
    zy = z * y_f.unsqueeze(1)
    cov.mul_(scale_old).add_((zy.transpose(0, 1) @ z).to(cov), alpha=inv_new)
    u.mul_(scale_old).add_(zy.sum(dim=0).to(u), alpha=inv_new)
    return total_new


@torch.no_grad()
def welford_cov_accumulate_(s: Tensor, mean: Tensor, count: int, z: Tensor) -> int:
    """Fold a ``(m, P)`` block into the running centered co-moment and mean.

    Single-pass Welford/Chan parallel-merge of the **unnormalised** centered
    co-moment ``S = Σ (z−μ)(z−μ)ᵀ`` (``s``, ``(P, P)``) and the running mean
    ``μ`` (``mean``, ``(P,)``).  Unlike the raw second-moment ``Σ z zᵀ``, this
    never forms a large raw Gram, so the per-direction variance
    ``var_j = v_jᵀ (S/(count−1)) v_j`` is a PSD quadratic form with **no
    subtraction** — stable in **float32** (no float64 ``(P, P)`` buffer needed).

    Per batch (with the prior ``count`` rows already folded and ``m = z.shape[0]``
    new rows)::

        μ_b   = mean(z)                       # batch mean (P,)
        S_b   = (z − μ_b)ᵀ (z − μ_b)          # one centered GEMM (P, P)
        δ     = μ_b − μ
        count_new = count + m
        S    += S_b + (count·m / count_new) · δ δᵀ   # Chan merge correction
        μ    += (m / count_new) · δ

    The ``count == 0`` first batch falls out of the same formula (the correction
    scales by ``0`` and ``μ`` jumps to ``μ_b``), so no special case is needed.
    Returns ``count_new`` (the caller threads it back in as the next ``count``).

    Mirrors the ``covariance_chunk_signed_*_accumulate_`` style: ``z`` is a 2-D
    row block ``(m, P)`` (FC features directly, or conv patches already flattened
    via :func:`flatten_spatial`), the batch GEMM runs in ``z``'s dtype/device
    (cheap float32 GPU), and the result folds into the accumulators via ``.to(s)``
    / ``.to(mean)`` (casts to the accumulator dtype **and** moves to its device,
    so the accumulators may live on CPU while the GEMM runs on GPU).  ``s`` is the
    **unnormalised** ``S``; the caller divides by ``count − 1`` once after the pass.
    """
    m = z.shape[0]
    count_new = count + m
    mu_b = z.mean(dim=0)  # (P,) in z's dtype/device
    zc = z - mu_b
    s_b = zc.transpose(0, 1) @ zc  # (P, P) centered batch co-moment
    delta = mu_b - mean.to(z)  # running mean brought to z's device for the merge
    alpha = count * m / count_new  # 0 on the first batch
    s.add_((s_b + alpha * torch.outer(delta, delta)).to(s))
    mean.add_((delta * (m / count_new)).to(mean))
    return count_new


def flatten_spatial_vector(z: Tensor, y: Tensor) -> tuple[Tensor, Tensor, int]:
    """Collapse conv-output spatial dims into the sample dim (vector labels).

    Vector-label analog of :func:`flatten_spatial`.  ``z`` has shape
    ``(N, P, H, W)`` and ``y`` shape ``(N, C)`` (one-hot / centered one-hot).
    Returns ``(z_flat, y_flat, spatial)`` where ``z_flat`` is ``(N*H*W, P)``,
    ``y_flat`` broadcasts each sample's length-``C`` label across its ``H*W``
    patches → ``(N*H*W, C)``, and ``spatial = H*W``.  The permutation order
    ``(N, P, H, W) → (N, H, W, P) → (N*H*W, P)`` matches :func:`flatten_spatial`
    so the labels line up with the patch features.
    """
    if y.ndim != 2:
        raise ValueError(
            f"y must be a 2-D (N, C) tensor for vector labels, got {tuple(y.shape)}"
        )
    n, p, h_out, w_out = z.shape
    spatial = h_out * w_out
    c = y.shape[1]
    z_flat = z.permute(0, 2, 3, 1).reshape(n * spatial, p)
    y_flat = y.to(z.dtype).unsqueeze(1).expand(n, spatial, c).reshape(n * spatial, c)
    return z_flat, y_flat, spatial


@torch.no_grad()
def covariance_chunk_signed_vector(z: Tensor, y: Tensor) -> tuple[Tensor, Tensor, int]:
    """Per-class signed covariances + cross-moment from one chunk (vector labels).

    Vector-label analog of :func:`covariance_chunk_signed`.  Inputs and layout
    match :func:`flatten_spatial_vector` (``z`` is ``(N, P, H, W)``, ``y`` is
    ``(N, C)``).  Returns ``(c_stack_part, m_part, n_patches)`` where

    * ``c_stack_part[c] = Σ_patch y_{·,c} z zᵀ`` is the *unnormalised* per-class
      signed covariance, shape ``(C, P, P)``;
    * ``m_part = Σ_patch z yᵀ`` is the *unnormalised* cross-moment ``(P, C)``;
    * ``n_patches = N*H*W``.

    The caller accumulates these across chunks and divides ``c_stack`` by the
    total ``n_patches`` exactly once (one global denominator — never normalise
    a class by its own patch count).  Each per-class covariance is built with a
    single BLAS-3 GEMM via :func:`signed_covariance_matrix` (``normalize=False,
    symmetrize=False``) rather than a fused einsum, keeping intermediate memory
    bounded.
    """
    z_flat, y_flat, _ = flatten_spatial_vector(z, y)
    p = z_flat.shape[1]
    c = y_flat.shape[1]
    c_stack = z_flat.new_empty((c, p, p))
    for cls in range(c):
        c_stack[cls] = signed_covariance_matrix(
            z_flat, y_flat[:, cls], normalize=False, symmetrize=False
        )
    m_part = z_flat.transpose(0, 1) @ y_flat
    return c_stack, m_part, z_flat.shape[0]


def _is_hard_one_hot(y: Tensor) -> bool:
    """True iff every row of ``y`` is a 0/1 one-hot vector (exactly one ``1``).

    Cheap ``O(M·C)`` gate (one host sync) for the class-segmented fast path in
    :func:`covariance_chunk_signed_vector_accumulate_`.  Centered one-hot
    (``one_hot - 1/C``) and soft labels fail this and take the weighted loop.
    """
    is_01 = ((y == 0) | (y == 1)).all()
    one_per_row = ((y == 1).sum(dim=1) == 1).all()
    return bool(is_01 & one_per_row)


@torch.no_grad()
def covariance_chunk_signed_vector_accumulate_(
    c_stack: Tensor,
    m: Tensor,
    z: Tensor,
    y: Tensor,
    total_old: int,
    assume_one_hot: bool | None = None,
) -> int:
    """Fold one chunk into the running-mean vector-label moments (float64).

    Streaming, allocation-free analog of :func:`covariance_chunk_signed_vector`.
    ``c_stack (C, P, P)`` holds the running *mean* per-class signed covariances
    ``(1/total) Σ y_{·,c} z zᵀ`` (divide-as-you-go: per chunk all classes are
    rescaled by ``total_old / total_new`` and the new contribution added scaled
    by ``1/total_new`` — every class decays even when it gets no rows this chunk,
    since the global denominator grew).  ``m (P, C)`` holds the *raw* cross-moment
    ``Σ z yᵀ`` (scale-free: it only feeds the linear-SVD directions).  Given the
    prior count ``total_old`` and ``new = z.shape[0]``, returns ``total_new``.

    ``z`` is a **2-D row block** ``(M, P)`` (FC features, or conv patches already
    flattened via :func:`flatten_spatial_vector`) and ``y`` is ``(M, C)``.

    **Class-segmented fast path (1A).**  For HARD one-hot labels the per-class
    covariance is the Gram of the class-``c`` row subset,
    ``C_c = Σ_{m: class=c} z_m z_mᵀ = Z_cᵀ Z_c``: select each class's rows once
    and form a single ``(P, P)`` GEMM per class, total ``Σ_c P²·M_c = P²·M``
    FLOPs — a ``C×`` cut over the weighted loop's ``C·P²·M``, exact and PSD.
    Soft / centered labels fall back to the weighted product
    ``zᵀ diag(y_{·,c}) z`` per class.  Per-class covariances are **not**
    symmetrised, matching :func:`covariance_chunk_signed_vector`.

    ``assume_one_hot`` short-circuits the per-chunk one-hot detection (a
    GPU→CPU ``bool`` sync that serialises dispatch against compute): pass
    ``True``/``False`` when the caller already knows the label encoding (it is
    constant across a fit); ``None`` auto-detects per chunk.
    """
    if y.ndim != 2:
        raise ValueError(
            f"y must be a 2-D (M, C) tensor for vector labels, got {tuple(y.shape)}"
        )
    new = z.shape[0]
    total_new = total_old + new
    scale_old = total_old / total_new
    inv_new = 1.0 / total_new
    y_f = y.to(z.dtype)
    c = y_f.shape[1]
    c_stack.mul_(scale_old)  # decay every class's running mean (global denominator)
    one_hot = _is_hard_one_hot(y_f) if assume_one_hot is None else assume_one_hot
    if one_hot:
        cls = y_f.argmax(dim=1)
        for k in range(c):
            z_k = z[cls == k]
            if z_k.shape[0] == 0:
                continue
            c_stack[k].add_((z_k.transpose(0, 1) @ z_k).to(c_stack), alpha=inv_new)
    else:
        for k in range(c):
            zy = z * y_f[:, k].unsqueeze(1)
            c_stack[k].add_((zy.transpose(0, 1) @ z).to(c_stack), alpha=inv_new)
    m.add_((z.transpose(0, 1) @ y_f).to(m))
    return total_new


def auto_batch_size(n: int, p: int, mem_limit_bytes: int) -> tuple[bool, int]:
    """Decide whether to chunk a feature tensor and pick a chunk size.

    Returns ``(need_batch, batch_size)``. If the ``(N, P)`` float32 tensor
    would exceed ``mem_limit_bytes``, ``need_batch`` is ``True`` and
    ``batch_size`` is the largest chunk whose float32 footprint fits.
    Otherwise ``need_batch`` is ``False`` and ``batch_size = n``.

    When ``p <= 0`` chunking is meaningless; returns ``(False, n)`` so the
    caller can proceed (and fail on the downstream "p > 0" check with a
    clearer message).
    """
    if p <= 0:
        return False, n
    need_batch = (n * p * 4) > mem_limit_bytes
    batch_size = max(1, mem_limit_bytes // (p * 4)) if need_batch else n
    return need_batch, batch_size
