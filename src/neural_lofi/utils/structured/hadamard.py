"""Fast Walsh–Hadamard transform and power-of-two helpers.

The fast Walsh–Hadamard transform (FWHT) is the core primitive of the
structured random transforms (SORF, Fastfood, ...) implemented in
:mod:`neural_lofi.utils.transforms`.  It applies the (Sylvester /
natural-ordered) Hadamard matrix ``H_D`` to the last axis of a tensor in
``O(D log D)`` time and ``O(1)`` extra storage, never materialising the
``(D, D)`` matrix.

``H_D`` is the recursive Sylvester construction
``H_{2n} = [[H_n, H_n], [H_n, -H_n]]`` with ``H_1 = [1]`` — the same matrix
ordering returned by :func:`scipy.linalg.hadamard`.  With ``normalize=True``
the transform is scaled by ``1/sqrt(D)`` so it is orthonormal
(``H_norm @ H_norm = I``), which makes :func:`fwht` its own inverse.

The functions are *mechanical*: they operate on whatever device and dtype the
input carries.  Power-of-two padding policy lives with the callers.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# Optional fused CUDA kernel (Dao-AILab/fast-hadamard-transform).  When present,
# it runs the FWHT at memory-bandwidth in a single fused pass — 12-37x faster
# than the pure-PyTorch butterfly below (which is memory-bound by its log2(D)
# strided stages).  It is the path that makes SORF actually pay off on wide /
# large-d layers.  Optional: requires a source build against the local torch +
# CUDA toolkit (see the fast-hadamard-transform README).  Absent → the
# pure-PyTorch fallback is used, so CPU and kernel-less environments still work.
try:  # pragma: no cover - import guard
    from fast_hadamard_transform import (  # type: ignore[import-untyped,import-not-found]
        hadamard_transform as _fused_hadamard,
    )
except Exception:  # pragma: no cover - kernel not installed
    _fused_hadamard = None

# Largest D for the cached-matrix fallback: build the dense (D, D) Hadamard once
# and apply it as a cuBLAS matmul (far faster than the butterfly for small D).
# Above this the (D, D) matrix is too big — fall back to the butterfly.
_MATMUL_MAX_D = 4096
# (D, device, dtype) -> normalised Sylvester Hadamard matrix H_D / sqrt(D).
# FIFO-bounded (see _hadamard_matrix) so it cannot pin unbounded GPU memory.
_HMAT_CACHE: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}
_HMAT_CACHE_MAX_ENTRIES = 4
# dtypes the fused kernel supports.
_FUSED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FUSED_MAX_D = 1 << 15  # kernel supports power-of-two D up to 2**15


def is_pow2(d: int) -> bool:
    """Return ``True`` iff ``d`` is a positive power of two."""
    return d > 0 and (d & (d - 1)) == 0


def next_pow2(d: int) -> int:
    """Smallest power of two ``>= d`` (and ``>= 1``).

    ``next_pow2(1) == 1``, ``next_pow2(5) == 8``, ``next_pow2(8) == 8``.
    """
    if d <= 1:
        return 1
    return 1 << (d - 1).bit_length()


def pad_to(x: Tensor, d_pad: int) -> Tensor:
    """Zero-pad the last axis of ``x`` up to width ``d_pad``.

    Returns ``x`` unchanged when its last dim already equals ``d_pad``.
    Raises ``ValueError`` if ``d_pad`` is smaller than the current width.
    """
    d_in = x.shape[-1]
    if d_pad == d_in:
        return x
    if d_pad < d_in:
        raise ValueError(f"d_pad={d_pad} is smaller than the last dim {d_in}")
    return F.pad(x, (0, d_pad - d_in))


def _hadamard_matrix(d: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Cached normalised Sylvester Hadamard matrix ``H_D / sqrt(D)`` (``D`` pow2).

    The cache is FIFO-bounded to ``_HMAT_CACHE_MAX_ENTRIES``: a real run uses
    only a handful of distinct ``(D, device, dtype)`` keys, but an unbounded
    dict could pin many large matrices (67 MB at D=4096 float32) on the GPU.
    """
    key = (d, device, dtype)
    h = _HMAT_CACHE.get(key)
    if h is None:
        m = torch.ones((1, 1), device=device, dtype=dtype)
        while m.shape[0] < d:  # Sylvester recursion H_{2n} = [[H,H],[H,-H]]
            m = torch.cat([torch.cat([m, m], 1), torch.cat([m, -m], 1)], 0)
        h = m / math.sqrt(d)
        while len(_HMAT_CACHE) >= _HMAT_CACHE_MAX_ENTRIES:
            _HMAT_CACHE.pop(next(iter(_HMAT_CACHE)))  # dicts iterate FIFO
        _HMAT_CACHE[key] = h
    return h


def _fwht_butterfly(x: Tensor, normalize: bool) -> Tensor:
    """Pure-PyTorch FWHT via ``log2(D)`` butterfly stages (fallback for large D)."""
    d = x.shape[-1]
    lead = x.shape[:-1]
    y = x.clone()
    h = 1
    while h < d:
        # Split each length-2h block into [first h | second h] and apply the
        # butterfly (a, b) -> (a + b, a - b) element-wise.
        y = y.reshape(*lead, d // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*lead, d)
        h *= 2
    return y / math.sqrt(d) if normalize else y


def fwht(x: Tensor, *, normalize: bool = True) -> Tensor:
    """Fast Walsh–Hadamard transform over the last axis of ``x``.

    Computes ``y = H_D @ x`` per row (``H_D`` the Sylvester-ordered Hadamard
    matrix; symmetric so ``H_D = H_D^T``), vectorised over all leading
    dimensions.  Dispatches to the fastest available backend, all numerically
    identical (same ordering, atol ~1e-5):

    1. **fused CUDA kernel** (``fast_hadamard_transform``) when installed and
       ``x`` is CUDA, a supported dtype, and ``D <= 2**15`` — one fused
       bandwidth-bound pass;
    2. **cached-matrix matmul** for ``D <= _MATMUL_MAX_D`` — build ``H_D`` once
       and apply it as a cuBLAS GEMM (far faster than the butterfly for small D);
    3. **butterfly** otherwise — pure-PyTorch ``log2(D)`` stages.

    Parameters
    ----------
    x : Tensor
        ``(..., D)`` with ``D`` a power of two.
    normalize : bool, default True
        Divide by ``sqrt(D)`` so the transform is orthonormal (self-inverse).
        ``False`` applies the raw ``±1`` Hadamard matrix.
    """
    d = x.shape[-1]
    if not is_pow2(d):
        raise ValueError(f"fwht requires the last dim to be a power of two, got {d}")

    if (
        _fused_hadamard is not None
        and x.is_cuda
        and x.dtype in _FUSED_DTYPES
        and d <= _FUSED_MAX_D
    ):
        # hadamard_transform(x, scale) = scale * H_raw @ x  (raw ±1 Hadamard).
        scale = (1.0 / math.sqrt(d)) if normalize else 1.0
        out: Tensor = _fused_hadamard(x.contiguous(), scale=scale)
        return out

    if d <= _MATMUL_MAX_D:
        # x @ (H_D / sqrt(D)) = normalised FWHT; undo the scale when not normalising.
        h = _hadamard_matrix(d, x.device, x.dtype)
        y = x @ h
        return y if normalize else y * math.sqrt(d)
    return _fwht_butterfly(x, normalize)
