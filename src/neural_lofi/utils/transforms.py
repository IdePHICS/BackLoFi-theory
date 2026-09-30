"""Structured random transforms for the spectral models.

Each spectral random-features layer applies a random projection ``h · W`` to
its input.  The default is a dense Gaussian ``W`` (``O(D·P)`` time and
storage).  This module provides a small, extensible abstraction that lets a
layer swap that dense projection for a *structured* operator that is much
cheaper while behaving like a Gaussian projection in its first two moments.

Currently implemented:

- :class:`DenseTransform` / :class:`DenseConvTransform` — thin wrappers around
  the existing dense weight (kept bit-identical; the matrix still lives in the
  block's ``blocks[ℓ].weight`` so serialization is unchanged).
- :class:`SORFTransform` / :class:`SORFConvTransform` — **SORF** (Structured
  Orthogonal Random Features, Yu et al. 2016): ``W = √D_pad · H D₃ H D₂ H D₁``
  with three Rademacher sign diagonals and the Walsh–Hadamard transform ``H``.
  ``O(D log D)`` time, ``O(P)`` storage.

Adding Fastfood / ORF / ACDC later means adding a subclass and a branch in
:func:`make_transform`.

Each transform exposes ``project(h)`` returning the **pre-activation**
projection only — the activation ``σ``, the per-layer norm ``c`` and the
``/√P`` factor (FFN) stay at the call sites.  CNN-mode bakes the ``1/√p``
scaling into the projection (mirroring the dense conv weight, which is
pre-scaled at init), so :class:`SORFConvTransform` applies it internally.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .structured import conv_sorf_project, fc_sorf_project, next_pow2, pad_to


def _draw_signs(n_blocks: int, d_pad: int, seed: int) -> Tensor:
    """Draw a ``(n_blocks, 3, d_pad)`` tensor of i.i.d. Rademacher ±1 values."""
    gen = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n_blocks, 3, d_pad), generator=gen)
    return (bits * 2 - 1).to(torch.float32)


class StructuredTransform(nn.Module):
    """Base class for a random projection ``h ↦ h · W`` (pre-activation).

    Subclasses set ``in_features`` / ``out_features`` and implement
    :meth:`project`.  ``forward`` delegates to :meth:`project` so a transform
    is also callable as a module.
    """

    in_features: int
    out_features: int

    def project(self, h: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, h: Tensor) -> Tensor:
        return self.project(h)

    @property
    def weight_view(self) -> Tensor | None:
        """Dense ``(in, out)`` matrix for introspection, or ``None``."""
        return None


class DenseTransform(StructuredTransform):
    """Dense Gaussian FFN projection ``h @ W`` with ``W`` of shape ``(D, P)``.

    Holds a *reference* to the block's ``blocks[ℓ].weight`` Parameter (boxed
    in a list so ``nn.Module`` does not re-register it) — the matrix stays the
    single serialized source of truth, so the dense path is bit-identical to a
    materialised ``h @ W``.
    """

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"DenseTransform expects a 2-D weight, got {weight.shape}")
        self.in_features = weight.shape[0]
        self.out_features = weight.shape[1]
        # List-box avoids nn.Module Parameter/Module auto-registration.
        self._weight_box = [weight]

    def project(self, h: Tensor) -> Tensor:
        w = self._weight_box[0]
        if h.shape[-1] != self.in_features:
            raise ValueError(
                f"DenseTransform expected {self.in_features} input features, "
                f"got {h.shape[-1]}"
            )
        return h @ w

    @property
    def weight_view(self) -> Tensor | None:
        return self._weight_box[0]


class ConvStructuredTransform(StructuredTransform):
    """Base for conv-style transforms: ``(N, C, H, W) → (N, P, H', W')``.

    Declares ``in_channels`` / ``out_channels`` so callers can validate the
    input channel count without knowing the concrete (dense vs SORF) class.
    """

    in_channels: int
    out_channels: int


class DenseConvTransform(ConvStructuredTransform):
    """Dense random conv projection ``conv2d(h, W)`` with ``W=(P, C, k, k)``.

    Like :class:`DenseTransform`, aliases the model's pre-scaled conv weight
    (``randn(P,C,k,k)/√P``) so the path is bit-identical to ``F.conv2d``.
    """

    def __init__(self, weight: Tensor, *, padding: int, stride: int) -> None:
        super().__init__()
        if weight.ndim != 4:
            raise ValueError(
                f"DenseConvTransform expects a 4-D weight, got {weight.shape}"
            )
        out_channels, in_channels, kh, kw = weight.shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_channels * kh * kw
        self.out_features = out_channels
        self.padding = padding
        self.stride = stride
        self._weight_box = [weight]

    def project(self, h: Tensor) -> Tensor:
        return F.conv2d(
            h, self._weight_box[0], padding=self.padding, stride=self.stride
        )

    @property
    def weight_view(self) -> Tensor | None:
        return self._weight_box[0]


class SORFTransform(StructuredTransform):
    """Structured Orthogonal Random Features projection (FFN).

    One block applies ``W = √D_pad · H D₃ H D₂ H D₁`` (orthonormal ``H``,
    Rademacher diagonals) to a ``D_pad``-padded input, matching an ``N(0,1)``
    Gaussian projection in its first two moments.  For width ``P`` we stack
    ``⌈P/D_pad⌉`` independent blocks and truncate to ``P``.

    Storage is the ``(n_blocks, 3, D_pad)`` sign buffer only (``O(P)``).
    """

    def __init__(self, in_features: int, out_features: int, *, seed: int) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("SORFTransform requires positive in/out features")
        self.in_features = in_features
        self.out_features = out_features
        self.d_pad = next_pow2(in_features)
        self.n_blocks = math.ceil(out_features / self.d_pad)
        self.register_buffer("signs", _draw_signs(self.n_blocks, self.d_pad, seed))

    def project(self, h: Tensor) -> Tensor:
        if h.shape[-1] != self.in_features:
            raise ValueError(
                f"SORFTransform expected {self.in_features} input features, "
                f"got {h.shape[-1]}"
            )
        x = pad_to(h, self.d_pad)  # (N, d_pad)
        # Buffers follow the input device/dtype (cheap no-op when matching).
        s = cast(Tensor, self.signs).to(device=h.device, dtype=h.dtype)  # (B,3,d_pad)

        # Per block: scale·H(s₂ ⊙ H(s₁ ⊙ H(s₀ ⊙ x))) with the folded scale
        # √D_pad·(1/√D_pad)³ = 1/D_pad, truncated to out_features.  Dispatched
        # to the fused Triton kernel when available, else the reference.
        return fc_sorf_project(
            x, s, scale=1.0 / self.d_pad, out_features=self.out_features
        )


class SORFConvTransform(ConvStructuredTransform):
    """SORF random conv: SORF over the flattened ``C·k·k`` patch dimension.

    A random conv layer is patch extraction (``im2col``) followed by a dense
    projection of the flattened patch.  This replaces that projection with a
    :class:`SORFTransform` over the patch dimension.  Like the FFN transforms,
    it returns the *raw* projection (no ``1/√p`` baked in); the ``1/√p`` scaling
    is applied by the model forward after the nonlinearity, matching the FFN
    recipe ``z = σ(project(h)) / √P``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int,
        stride: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        patch_dim = in_channels * kernel_size * kernel_size
        self.in_features = patch_dim
        self.out_features = out_channels
        self.sorf = SORFTransform(patch_dim, out_channels, seed=seed)

    def project(self, h: Tensor) -> Tensor:
        if h.ndim != 4:
            raise ValueError(
                f"SORFConvTransform expects 4-D (N,C,H,W) input, got {tuple(h.shape)}"
            )
        n, c, h_in, w_in = h.shape
        if c != self.in_channels:
            raise ValueError(
                f"SORFConvTransform expected {self.in_channels} input channels, got {c}"
            )
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        # SORF over the C·k·k patches.  Dispatches to the fused implicit-im2col
        # Triton kernel (no materialised unfold) or the F.unfold reference.
        s = cast(Tensor, self.sorf.signs).to(device=h.device, dtype=h.dtype)
        proj = conv_sorf_project(  # (N*L, P) — raw; /√P applied by the model
            h,
            s,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            d_pad=self.sorf.d_pad,
            out_features=self.out_channels,
        )
        spatial = h_out * w_out
        return (
            proj.reshape(n, spatial, self.out_channels)
            .permute(0, 2, 1)
            .reshape(n, self.out_channels, h_out, w_out)
        )


def make_transform(
    kind: str,
    *,
    in_features: int,
    out_features: int,
    seed: int,
    dense_weight: Tensor | None = None,
) -> StructuredTransform:
    """Factory for FFN-layer transforms (the extensibility seam).

    ``kind="dense"`` wraps ``dense_weight`` (the block's ``blocks[ℓ].weight``);
    ``kind="sorf"`` builds a :class:`SORFTransform`.  Conv-layer transforms are
    built directly (they need conv geometry) — see
    :class:`DenseConvTransform` / :class:`SORFConvTransform`.
    """
    if kind == "dense":
        if dense_weight is None:
            raise ValueError("make_transform('dense') requires dense_weight")
        return DenseTransform(dense_weight)
    if kind == "sorf":
        return SORFTransform(in_features, out_features, seed=seed)
    raise ValueError(f"unknown transform {kind!r}; expected 'dense' or 'sorf'")
