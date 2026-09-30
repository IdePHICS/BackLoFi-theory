"""Fused Triton kernel for the SORF operator ``scale·H(s₂⊙H(s₁⊙H(s₀⊙x)))``.

The naive SORF is memory-bound: three Hadamards + three sign-multiplies done as
separate tensor ops make ~6 HBM passes over the ``(B, M, d_pad)`` working set.
This kernel fuses the whole chain — each program loads a ``(ROWS, d_pad)`` tile
of input rows for one block, applies all three sign-diagonals and Hadamards
*in registers*, and writes the result once.  One read, one write.

The in-register Hadamard is the standard Sylvester butterfly (``log₂(d_pad)``
stages of ``(a, b) → (a+b, a-b)``), expressed with ``tl.reshape`` / ``tl.permute``
/ ``tl.split`` / ``tl.join`` so it never leaves registers between stages.  The
butterfly stages commute (each acts on a distinct index bit), so the natural-order
result matches :func:`scipy.linalg.hadamard` and the reference
:func:`neural_lofi.utils.structured.hadamard.fwht` (``normalize=False``).

Only a power-of-two ``d_pad`` is supported; the caller
(:func:`neural_lofi.utils.structured.sorf.sorf_project`) gates on ``d_pad`` range,
CUDA, dtype and falls back to the pure-PyTorch reference otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

TRITON_OK = True


def _configs() -> list[triton.Config]:
    """Autotune space: rows-per-program × warps × pipeline stages."""
    cfgs = []
    for rows in (1, 2, 4, 8, 16):
        for warps in (4, 8, 16):
            for stages in (2, 3):
                cfgs.append(
                    triton.Config(
                        {"ROWS_PER_PROGRAM": rows},
                        num_warps=warps,
                        num_stages=stages,
                    )
                )
    return cfgs


def _prune(
    configs: Sequence[triton.Config], named_args: dict, **kwargs: object
) -> list:
    """Drop configs whose ``(ROWS, d_pad)`` register tile would blow up.

    Keeps the tile (and its transient butterfly buffer) within a sane register
    budget so large ``d_pad`` does not spill to local memory — which would
    reintroduce the HBM traffic the fusion removes.  ``BLOCK_D`` is a constexpr,
    so it arrives via ``kwargs`` (not ``named_args``).
    """
    d = cast(int, kwargs.get("BLOCK_D", named_args.get("BLOCK_D")))
    kept = [c for c in configs if d * c.kwargs["ROWS_PER_PROGRAM"] <= 8192]
    return kept or [min(configs, key=lambda c: c.kwargs["ROWS_PER_PROGRAM"])]


@triton.jit
def _hadamard(v, R: tl.constexpr, D: tl.constexpr, LOG2_D: tl.constexpr):  # noqa: N803
    """Raw (un-normalised) Walsh–Hadamard over the last axis of a ``(R, D)`` tile.

    ``log₂(D)`` butterfly stages, all in registers.  Stage ``k`` (stride
    ``h = 2**k``) reshapes to ``(R, G, 2, h)``, combines the size-2 axis as
    ``(a, b) → (a+b, a-b)`` (matching ``stack(..., dim=-2)`` in the reference
    butterfly), and reshapes back.
    """
    for k in tl.static_range(LOG2_D):
        # Stage k: stride h = 2**k, group count g = D // (2*h).  Inlined into the
        # shape tuples so they stay compile-time ints (an intermediate `h = 1<<k`
        # binding would be treated as a runtime tensor, and a constexpr binding
        # cannot be rebound across static_range iterations).
        y = tl.reshape(v, (R, D // (1 << (k + 1)), 2, 1 << k))
        y = tl.permute(y, (0, 1, 3, 2))  # (R, g, h, 2): size-2 axis last
        a, b = tl.split(y)  # (R, g, h) each
        y = tl.join(a + b, a - b)  # (R, g, h, 2)
        y = tl.permute(y, (0, 1, 3, 2))  # (R, g, 2, h)
        v = tl.reshape(y, (R, D))
    return v


@triton.autotune(
    configs=_configs(),
    key=["BLOCK_D", "M"],
    prune_configs_by={"early_config_prune": _prune},
)
@triton.jit
def _sorf_kernel(  # noqa: N803
    x_ptr,
    signs_ptr,
    out_ptr,
    M,
    scale,
    BLOCK_D: tl.constexpr,
    LOG2_D: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """One program owns block ``pid_b`` and a tile of ``ROWS_PER_PROGRAM`` rows.

    Row tiles are the X grid dim (cap 2**31) and blocks the Y dim (cap 65535):
    ``M`` (= N·L for conv after im2col) can far exceed 65535, so it must not sit
    on a Y/Z axis.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    # int64 row index: the (B, M, d_pad) output offset pid_b*M*BLOCK_D + ... can
    # exceed 2**31 for large M (= N·L on the conv reference path) or large B.
    rows = (pid_m * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)).to(tl.int64)
    row_mask = rows < M
    d_idx = tl.arange(0, BLOCK_D)

    # Input tile (ROWS, d_pad), coalesced; masked rows read as 0.
    x_off = rows[:, None] * BLOCK_D + d_idx[None, :]
    v = tl.load(x_ptr + x_off, mask=row_mask[:, None], other=0.0)

    # Signs for this block: (d_pad,) each, loaded once, reused across the row tile.
    s_base = pid_b * 3 * BLOCK_D
    s0 = tl.load(signs_ptr + s_base + d_idx)
    s1 = tl.load(signs_ptr + s_base + BLOCK_D + d_idx)
    s2 = tl.load(signs_ptr + s_base + 2 * BLOCK_D + d_idx)

    # The fused chain, entirely in registers.
    v = v * s0[None, :]
    v = _hadamard(v, ROWS_PER_PROGRAM, BLOCK_D, LOG2_D)
    v = v * s1[None, :]
    v = _hadamard(v, ROWS_PER_PROGRAM, BLOCK_D, LOG2_D)
    v = v * s2[None, :]
    v = _hadamard(v, ROWS_PER_PROGRAM, BLOCK_D, LOG2_D)
    v = v * scale

    # Output (B, M, d_pad): one coalesced write per (block, row).  pid_b cast to
    # int64 so the per-block stride pid_b*M*BLOCK_D does not overflow int32.
    out_off = (
        pid_b.to(tl.int64) * M * BLOCK_D + rows[:, None] * BLOCK_D + d_idx[None, :]
    )
    tl.store(out_ptr + out_off, v, mask=row_mask[:, None])


@triton.autotune(
    configs=_configs(),
    key=["BLOCK_D", "M", "P"],
    prune_configs_by={"early_config_prune": _prune},
)
@triton.jit
def _sorf_fc_kernel(  # noqa: N803
    x_ptr,
    signs_ptr,
    out_ptr,
    M,
    P,
    N_BLOCKS,
    scale,
    BLOCK_D: tl.constexpr,
    LOG2_D: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """One program owns a tile of rows and ALL ``N_BLOCKS`` blocks.

    The conv kernel's reuse trick applied to the FC case: the input tile is
    loaded from HBM **once** and reused across every block (each block has its
    own signs but the same input), and each block's result is written straight
    into its slice of the truncated ``(M, P)`` output — no ``(B, M, d_pad)``
    intermediate and no separate permute/truncate pass.  This removes the
    ``B``-fold input re-read of the per-block grid, which is what made the
    per-element cost climb once ``B = ceil(P/d_pad)`` grew past L2 reuse.
    """
    pid_m = tl.program_id(0)

    # int64 row index: rows*P can exceed 2**31 for large M or wide P.
    rows = (pid_m * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)).to(tl.int64)
    row_mask = rows < M
    d_idx = tl.arange(0, BLOCK_D)

    # Input tile (ROWS, d_pad), loaded once; masked rows read as 0.
    x_off = rows[:, None] * BLOCK_D + d_idx[None, :]
    x0 = tl.load(x_ptr + x_off, mask=row_mask[:, None], other=0.0)

    for b in tl.range(N_BLOCKS):
        s_base = b * 3 * BLOCK_D
        s0 = tl.load(signs_ptr + s_base + d_idx)
        s1 = tl.load(signs_ptr + s_base + BLOCK_D + d_idx)
        s2 = tl.load(signs_ptr + s_base + 2 * BLOCK_D + d_idx)

        v = x0 * s0[None, :]
        v = _hadamard(v, ROWS_PER_PROGRAM, BLOCK_D, LOG2_D)
        v = v * s1[None, :]
        v = _hadamard(v, ROWS_PER_PROGRAM, BLOCK_D, LOG2_D)
        v = v * s2[None, :]
        v = _hadamard(v, ROWS_PER_PROGRAM, BLOCK_D, LOG2_D)
        v = v * scale

        # Block b owns output columns [b·d_pad, b·d_pad + d_pad); keep only < P.
        col0 = b * BLOCK_D
        out_off = rows[:, None] * P + col0 + d_idx[None, :]
        col_mask = (col0 + d_idx)[None, :] < P
        tl.store(out_ptr + out_off, v, mask=row_mask[:, None] & col_mask)


def triton_sorf_fc(
    x_pad: Tensor, signs: Tensor, scale: float, out_features: int
) -> Tensor:
    """Fused FC SORF: ``(M, d_pad)`` × ``(B, 3, d_pad)`` → ``(M, out_features)``.

    Same chain as :func:`triton_sorf` but with the input tile reused across all
    blocks inside one program and the result written directly into the
    truncated ``(M, out_features)`` layout.
    """
    x_pad = x_pad.contiguous()
    signs = signs.contiguous()
    m, d_pad = x_pad.shape
    n_blocks = signs.shape[0]
    log2_d = d_pad.bit_length() - 1  # d_pad is a power of two

    out = torch.empty((m, out_features), device=x_pad.device, dtype=x_pad.dtype)

    def grid(meta: dict) -> tuple[int]:
        return (triton.cdiv(m, meta["ROWS_PER_PROGRAM"]),)

    with torch.cuda.device(x_pad.device):
        _sorf_fc_kernel[grid](
            x_pad,
            signs,
            out,
            m,
            out_features,
            n_blocks,
            scale,
            BLOCK_D=d_pad,
            LOG2_D=log2_d,
        )
    return out


def triton_sorf(x_pad: Tensor, signs: Tensor, scale: float) -> Tensor:
    """Fused SORF: ``(M, d_pad)`` × ``(B, 3, d_pad)`` → ``(B, M, d_pad)``.

    Computes ``scale · H(s₂ ⊙ H(s₁ ⊙ H(s₀ ⊙ x)))`` per block.  ``d_pad`` must be
    a power of two; ``x_pad`` and ``signs`` must be CUDA float32 (the caller
    guards this).
    """
    x_pad = x_pad.contiguous()
    signs = signs.contiguous()
    m, d_pad = x_pad.shape
    n_blocks = signs.shape[0]
    log2_d = d_pad.bit_length() - 1  # d_pad is a power of two

    out = torch.empty((n_blocks, m, d_pad), device=x_pad.device, dtype=x_pad.dtype)

    def grid(meta: dict) -> tuple[int, int]:
        # (row-tiles, blocks): row tiles on the X axis (cap 2**31) since M can be
        # huge for conv (N·L); blocks on Y (cap 65535, and n_blocks is small).
        return (triton.cdiv(m, meta["ROWS_PER_PROGRAM"]), n_blocks)

    # Launch under the input's device context: unlike torch ops, the eager
    # Triton launcher (and its autotuner) binds the CURRENT device's context,
    # and on a non-zero device that mismatch makes the driver reject the
    # pointer args ("cannot be accessed from Triton (cpu tensor?)").
    with torch.cuda.device(x_pad.device):
        _sorf_kernel[grid](
            x_pad,
            signs,
            out,
            m,
            scale,
            BLOCK_D=d_pad,
            LOG2_D=log2_d,
        )
    return out
