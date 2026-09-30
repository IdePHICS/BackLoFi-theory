"""Fused SORF operator: ``scale · H(s₂ ⊙ H(s₁ ⊙ H(s₀ ⊙ x)))``.

This is the compute core of :class:`neural_lofi.utils.transforms.SORFTransform`.
One *block* applies the three sign-diagonals ``s₀, s₁, s₂`` interleaved with the
raw (un-normalised) Walsh–Hadamard transform ``H``; ``scale = 1/d_pad`` folds the
``√d_pad`` SORF normalisation together with the three ``1/√d_pad`` Hadamard norms.

The bottleneck of the naive implementation is *memory traffic*, not FLOPs: done as
six separate tensor ops (three Hadamards + three sign-multiplies) it makes ~6 HBM
passes over the full ``(B, M, d_pad)`` working set.  :func:`sorf_project` dispatches
to a fused first-party Triton kernel (``triton_sorf``) that does the whole chain in
registers — one read, one write — when it is available and the layer width is in
range, and otherwise to the pure-PyTorch reference (which itself routes each
Hadamard through the optional Dao fused kernel).  The two paths are numerically
identical to ``atol ~1e-4``; the reference is the golden definition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .hadamard import fwht, is_pow2, pad_to

# Largest padded layer width handled by the in-register Triton kernel.  Above
# this the ``(rows, d_pad)`` register tile spills (re-introducing the HBM traffic
# the fusion removes), so we fall back to the reference — whose Hadamards route
# through the Dao kernel, which is tuned for large D.  One-line-movable; the
# 2048 boundary is confirmed empirically by ``benchmark_sorf_triton.py``.
_TRITON_MAX_D = 2048

# Optional fused Triton backend.  Absent (no triton / not yet built) -> the
# reference path is always used, so CPU and kernel-less environments still work.
try:  # pragma: no cover - import guard
    from .triton_sorf import TRITON_OK as _TRITON_OK
    from .triton_sorf import triton_sorf as _triton_sorf
    from .triton_sorf import triton_sorf_fc as _triton_sorf_fc
except Exception:  # pragma: no cover - triton backend unavailable
    _TRITON_OK = False
    _triton_sorf = None  # type: ignore[assignment]
    _triton_sorf_fc = None  # type: ignore[assignment]

# Optional fused implicit-im2col conv-SORF kernel (removes the explicit F.unfold).
try:  # pragma: no cover - import guard
    from .triton_conv_sorf import TRITON_CONV_OK as _TRITON_CONV_OK
    from .triton_conv_sorf import triton_conv_sorf as _triton_conv_sorf
except Exception:  # pragma: no cover - triton backend unavailable
    _TRITON_CONV_OK = False
    _triton_conv_sorf = None  # type: ignore[assignment]


def _sorf_reference(x_pad: Tensor, signs: Tensor, scale: float) -> Tensor:
    """Golden SORF: ``(M, d_pad)`` × ``(B, 3, d_pad)`` → ``(B, M, d_pad)``.

    Six-pass form (three sign-multiplies + three raw Hadamards), vectorised over
    the ``B`` blocks.  ``signs`` is expected already on ``x_pad``'s device/dtype.
    """
    t = x_pad.unsqueeze(0) * signs[:, 0].unsqueeze(1)  # (B, M, d_pad)
    t = fwht(t, normalize=False)
    t = t * signs[:, 1].unsqueeze(1)
    t = fwht(t, normalize=False)
    t = t * signs[:, 2].unsqueeze(1)
    t = fwht(t, normalize=False)
    return t * scale


def _use_triton(x_pad: Tensor) -> bool:
    """True iff the fused Triton kernel applies to this input."""
    d_pad = x_pad.shape[-1]
    return (
        _TRITON_OK
        and _triton_sorf is not None
        and x_pad.is_cuda
        and x_pad.dtype == torch.float32
        and is_pow2(d_pad)
        and d_pad <= _TRITON_MAX_D
    )


def sorf_project(x_pad: Tensor, signs: Tensor, *, scale: float) -> Tensor:
    """Apply the SORF chain to every block, returning ``(B, M, d_pad)``.

    Parameters
    ----------
    x_pad : Tensor
        ``(M, d_pad)`` padded input rows, ``d_pad`` a power of two.
    signs : Tensor
        ``(B, 3, d_pad)`` Rademacher ±1 diagonals, on ``x_pad``'s device/dtype.
    scale : float
        Folded normalisation (``1/d_pad`` for the standard SORF recipe).

    The result is ``scale · H(s₂ ⊙ H(s₁ ⊙ H(s₀ ⊙ x)))`` per block.  The caller
    is responsible for the final ``permute``/``reshape``/truncation to width ``P``.
    """
    if _use_triton(x_pad):
        out: Tensor = _triton_sorf(x_pad, signs, scale)
        return out
    return _sorf_reference(x_pad, signs, scale)


def fc_sorf_project(
    x_pad: Tensor, signs: Tensor, *, scale: float, out_features: int
) -> Tensor:
    """SORF chain + truncation to width ``P``: ``(M, d_pad)`` → ``(M, P)``.

    The FC layer's full projection in one call.  Triton path: a single fused
    kernel that loads each input tile once, reuses it across all ``B`` blocks
    (the conv kernel's reuse trick), and writes straight into the truncated
    ``(M, P)`` output — no ``(B, M, d_pad)`` intermediate and no separate
    permute/truncate pass.  Fallback: the reference chain followed by the
    permute/reshape/truncate.
    """
    if _use_triton(x_pad):
        out: Tensor = _triton_sorf_fc(x_pad, signs, scale, out_features)
        return out
    t = _sorf_reference(x_pad, signs, scale)
    m = x_pad.shape[0]
    return t.permute(1, 0, 2).reshape(m, -1)[:, :out_features]


def _use_triton_conv(h: Tensor, d_pad: int) -> bool:
    """True iff the fused implicit-im2col conv kernel applies to this input."""
    return (
        _TRITON_CONV_OK
        and _triton_conv_sorf is not None
        and h.is_cuda
        and h.dtype == torch.float32
        and is_pow2(d_pad)
        and d_pad <= _TRITON_MAX_D
    )


def conv_sorf_project(
    h: Tensor,
    signs: Tensor,
    *,
    kernel_size: int,
    padding: int,
    stride: int,
    d_pad: int,
    out_features: int,
) -> Tensor:
    """SORF over conv patches of ``h`` → ``(N·L, out_features)``.

    Dispatches to the fused implicit-im2col Triton kernel (one cache-friendly
    input gather + one write, no materialised ``F.unfold``) when available, else
    the reference: explicit ``F.unfold`` + the (Triton-accelerated) FC
    :func:`sorf_project`.  ``signs`` is ``(B, 3, d_pad)`` on ``h``'s device/dtype.
    The caller reshapes the ``(N·L, P)`` result to ``(N, P, H_out, W_out)``.
    """
    scale = 1.0 / d_pad
    if _use_triton_conv(h, d_pad):
        out: Tensor = _triton_conv_sorf(
            h,
            signs,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            scale=scale,
            out_features=out_features,
        )  # (N·L, P) — already truncated
        return out

    # Reference: explicit im2col then the FC SORF over flattened patches.
    patches = F.unfold(h, kernel_size, padding=padding, stride=stride)
    n = h.shape[0]
    spatial = patches.shape[-1]
    pt = patches.transpose(1, 2).reshape(n * spatial, -1)
    x = pad_to(pt, d_pad)
    t = sorf_project(x, signs, scale=scale)  # (B, N·L, d_pad)
    n_blocks = t.shape[0]
    return t.permute(1, 0, 2).reshape(n * spatial, n_blocks * d_pad)[:, :out_features]
