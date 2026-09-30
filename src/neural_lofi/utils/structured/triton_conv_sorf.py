"""Fused implicit-im2col + SORF convolution kernel.

``SORFConvTransform`` is a random conv: extract every ``C·k²`` patch (im2col)
then apply the SORF projection to the flattened patch.  The pure-PyTorch path
does an **explicit** ``F.unfold`` — materialising a ``k²``×-larger patch tensor
in HBM — before the SORF.  That im2col traffic, not the Hadamards, is what keeps
conv-SORF behind cuDNN.

This kernel removes it: each program gathers its patch **implicitly** straight
from the input feature map (computing the input offsets on the fly, masking
conv-padding and the ``d_pad`` zero-pad), applies the SORF chain in registers
(reusing :func:`neural_lofi.utils.structured.triton_sorf._hadamard`), and writes
the result once.  Net traffic ≈ one (cache-friendly, overlapping) input gather +
one output write, vs the explicit path's input-read + ``k²``×-write + SORF.

Patch ordering matches ``torch.nn.functional.unfold`` (channel-major
``idx = c·k² + kh·k + kw``) so the output is bit-compatible with the reference.
Only power-of-two ``d_pad`` and CUDA float32 are supported; the caller gates this.
"""

from __future__ import annotations

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

from .triton_sorf import _configs, _hadamard, _prune

TRITON_CONV_OK = True


@triton.autotune(
    configs=_configs(),
    key=["BLOCK_D", "C", "H", "W", "K", "M"],
    prune_configs_by={"early_config_prune": _prune},
)
@triton.jit
def _conv_sorf_kernel(  # noqa: N803
    h_ptr,
    signs_ptr,
    out_ptr,
    M,
    P,
    N_BLOCKS,
    scale,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    W_OUT: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG2_D: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """One program owns a tile of output pixels (rows) and ALL ``N_BLOCKS`` blocks.

    The patch is gathered from the input feature map **once** and reused across
    every block (each block has its own signs but the same input patch), so the
    ``B``-fold gather redundancy of a per-block grid is removed — decisive when
    ``d_pad`` is small and ``B = ceil(P/d_pad)`` is large (e.g. a C=3 conv1 with
    wide ``P``).  Row tiles are the (1-D) grid dim; ``M`` can be huge (N·L).
    """
    pid_m = tl.program_id(0)

    # int64 row index: the output offset rows*P and the gather offset
    # (n*C+c)*H*W can exceed 2**31 for large M (= N·L) or large P.
    rows = (pid_m * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)).to(tl.int64)
    row_mask = rows < M  # output pixel
    # Decode output pixel -> (n, oh, ow).
    n = rows // L
    rem_l = rows % L
    oh = rem_l // W_OUT
    ow = rem_l % W_OUT

    # Decode patch element d_idx -> (c, kh, kw), matching F.unfold's c-major order.
    d_idx = tl.arange(0, BLOCK_D)
    c = d_idx // (K * K)
    rem_k = d_idx % (K * K)
    kh = rem_k // K
    kw = rem_k % K

    # Implicit im2col: gather the patch ONCE (mask conv padding + the d_pad
    # zero-pad d_idx >= C*K*K).
    ih = oh[:, None] * STRIDE - PAD + kh[None, :]
    iw = ow[:, None] * STRIDE - PAD + kw[None, :]
    valid = (d_idx[None, :] < C * K * K) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    offset = ((n[:, None] * C + c[None, :]) * H + ih) * W + iw
    patch = tl.load(h_ptr + offset, mask=valid & row_mask[:, None], other=0.0)

    # Each block: its own signs over the shared patch, written to its P-slice.
    for b in tl.range(N_BLOCKS):
        s_base = b * 3 * BLOCK_D
        s0 = tl.load(signs_ptr + s_base + d_idx)
        s1 = tl.load(signs_ptr + s_base + BLOCK_D + d_idx)
        s2 = tl.load(signs_ptr + s_base + 2 * BLOCK_D + d_idx)

        v = patch * s0[None, :]
        v = _hadamard(v, ROWS_PER_PROGRAM, BLOCK_D, LOG2_D)
        v = v * s1[None, :]
        v = _hadamard(v, ROWS_PER_PROGRAM, BLOCK_D, LOG2_D)
        v = v * s2[None, :]
        v = _hadamard(v, ROWS_PER_PROGRAM, BLOCK_D, LOG2_D)
        v = v * scale

        # Block b owns output columns [b·d_pad, b·d_pad + d_pad); keep only < P
        # (writing the full d_pad and truncating later would waste the write when
        # P < B·d_pad).
        col0 = b * BLOCK_D
        out_off = rows[:, None] * P + col0 + d_idx[None, :]
        col_mask = (col0 + d_idx)[None, :] < P
        tl.store(out_ptr + out_off, v, mask=row_mask[:, None] & col_mask)


def triton_conv_sorf(
    h: Tensor,
    signs: Tensor,
    *,
    kernel_size: int,
    padding: int,
    stride: int,
    scale: float,
    out_features: int,
) -> Tensor:
    """Fused conv-SORF: ``(N,C,H,W)`` × ``(B,3,d_pad)`` → ``(N·L, P)``.

    Each block writes its slice of the ``P`` output channels directly, so the
    result is already truncated to ``P``; the caller reshapes to
    ``(N, P, H_out, W_out)``.
    """
    h = h.contiguous()
    signs = signs.contiguous()
    n, c, h_in, w_in = h.shape
    k = kernel_size
    h_out = (h_in + 2 * padding - k) // stride + 1
    w_out = (w_in + 2 * padding - k) // stride + 1
    spatial = h_out * w_out
    m = n * spatial
    d_pad = signs.shape[-1]
    n_blocks = signs.shape[0]
    log2_d = d_pad.bit_length() - 1

    out = torch.empty((m, out_features), device=h.device, dtype=h.dtype)

    def grid(meta: dict) -> tuple[int]:
        return (triton.cdiv(m, meta["ROWS_PER_PROGRAM"]),)

    # Launch under the input's device context: unlike torch ops, the eager
    # Triton launcher (and its autotuner) binds the CURRENT device's context,
    # and on a non-zero device that mismatch makes the driver reject the
    # pointer args ("cannot be accessed from Triton (cpu tensor?)").
    with torch.cuda.device(h.device):
        _conv_sorf_kernel[grid](
            h,
            signs,
            out,
            m,
            out_features,
            n_blocks,
            scale,
            C=c,
            H=h_in,
            W=w_in,
            W_OUT=w_out,
            L=spatial,
            K=k,
            STRIDE=stride,
            PAD=padding,
            BLOCK_D=d_pad,
            LOG2_D=log2_d,
        )
    return out
