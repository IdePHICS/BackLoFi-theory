"""Stateless block ops (``pre`` flatten / l2norm, end-of-block ``pool``) + the
FFN/CNN mode label.

The weight-free ops applied inside a reduce-first block: the ``pre`` operations
between a block's reduce and its expand (flatten / l2norm) and the end-of-block
pool on the wide post-σ features.  They carry no learned or random state, so they
apply themselves on a tensor via ``apply``.  Extracted here (out of the retired
``layer_specs`` vocabulary) so the block model depends only on the block schema.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

InitStyle = Literal["ffn", "cnn"]


class Flatten:
    """Flatten ``(N, C, H, W)`` image features to ``(N, C*H*W)``."""

    def apply(self, h: Tensor) -> Tensor:
        return h.flatten(1)


class L2Norm:
    """Per-location L2 normalisation across channels: ``x / ‖x‖₂`` along dim 1."""

    def apply(self, h: Tensor) -> Tensor:
        norm = torch.sqrt(torch.sum(h**2, dim=1, keepdim=True) + 1e-8)
        return h / norm


class Pool:
    """End-of-block spatial pool (max or average) on the wide post-σ features."""

    def __init__(
        self,
        *,
        mode: str = "max",
        kernel_size: int = 2,
        stride: int | None = None,
        padding: int = 0,
    ) -> None:
        self.mode = mode
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def apply(self, h: Tensor) -> Tensor:
        # Dispatch on the (picklable) string mode rather than storing the pooling
        # functional as an attribute — F.max_pool2d is a ``boolean_dispatch`` local
        # that can't be pickled, which would make any pooling model un-torch.save-able.
        pool_fn = (
            torch.nn.functional.avg_pool2d
            if self.mode.strip().lower() == "avg"
            else torch.nn.functional.max_pool2d
        )
        return pool_fn(
            h, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )


# Block ``pre`` op name → stateless op class (reuse their ``apply``).
PRE_OPS: dict[str, type[Flatten] | type[L2Norm]] = {
    "flatten": Flatten,
    "l2norm": L2Norm,
}
