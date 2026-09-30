"""Reduce-first spectral model.

A single ``SpectralModel`` covering both the feedforward and convolutional
NLoFi architectures, built via :meth:`SpectralModel.from_block_config`.  A model
is ``L`` :class:`_Block`s + a final :class:`_Reduce` → ridge readout, applying
per block::

    a_l = pool(sigma(W_l @ (V_{l-1} @ a_{l-1}) / c_l) / sqrt(P_l))

— *reduce* the wide incoming features ``V_{l-1}``, then random-*expand* ``W_l``,
then sigma, then (CNN) pool.  The wide post-sigma ``a_l`` is carried between
blocks; each block opens by reducing it.  Random expand weights are stored raw
(the ``/sqrt(P)`` scaling lives at forward time).  ``_init_style`` is a read-only
``{"ffn", "cnn"}`` label derived from the ``input_shape`` rank.
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast

import torch
from torch import Tensor, nn

from ..utils.helpers import ACTIVATIONS
from ..utils.transforms import (
    DenseConvTransform,
    DenseTransform,
    SORFConvTransform,
    SORFTransform,
    StructuredTransform,
)
from .block_config import (
    BlockedModelConfig,
    ReduceConfig,
    validate_block_config,
)
from .block_ops import PRE_OPS, Flatten, InitStyle, L2Norm, Pool

log = logging.getLogger(__name__)


__all__ = [
    "InitStyle",
    "SpectralModel",
]


# ====================================================================== #
# Reduce-first runtime blocks.
#
# These back ``SpectralModel.from_block_config``.  A model is ``L`` ``_Block``s
# + a final ``_Reduce`` → ridge readout, applying per block:
#     reduce (V_{l-1}) -> pre (flatten/l2norm) -> expand (W_l)+sigma -> pool
# The fitted tensors (``V`` / whitening / ``c``) are filled by the trainer; an
# unfitted model is a numerical placeholder (V=0, identity whitening, c=1).
# ====================================================================== #


def _reduce_keep(r: ReduceConfig) -> int | None:
    """Output width of a reduction (``None`` ⇒ identity), incl. SVD/mean prepend."""
    if r.k is None:
        return None
    if r.linear_svd:
        return r.k + r.linear_k
    return r.k + 1 if r.include_linear_mean else r.k


class _Reduce(nn.Module):
    """A block's input reduction ``V_{ℓ-1}`` (+ optional centered whitening).

    ``keep is None`` ⇒ identity (``V_{-1}=I`` / no eigenreduction).  Applied as a
    matmul on flat features ``(N, D)`` or a 1×1 conv on image features
    ``(N, C, H, W)`` (``is_conv``).  ``V`` / ``mean`` / ``scale`` are filled by the
    trainer; the init (``V=0``, μ=0, σ=1) makes an unfitted reduce a no-op-shaped
    placeholder.
    """

    def __init__(
        self, *, in_width: int, keep: int | None, is_conv: bool, whiten: bool
    ) -> None:
        super().__init__()
        self.is_identity = keep is None
        self.is_conv = is_conv
        self.whiten = whiten
        if keep is None:
            self.V = nn.Parameter(torch.empty(0), requires_grad=False)
            self.mean = nn.Parameter(torch.empty(0), requires_grad=False)
            self.scale = nn.Parameter(torch.empty(0), requires_grad=False)
            return
        v = (
            torch.zeros(keep, in_width, 1, 1)
            if is_conv
            else torch.zeros(in_width, keep)
        )
        self.V = nn.Parameter(v, requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(keep), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(keep), requires_grad=False)

    def forward(self, a: Tensor) -> Tensor:
        if self.is_identity:
            return a
        r = torch.nn.functional.conv2d(a, self.V) if self.is_conv else a @ self.V
        if self.whiten:
            if r.ndim == 4:
                r = (r - self.mean.view(1, -1, 1, 1)) / self.scale.view(1, -1, 1, 1)
            else:
                r = (r - self.mean) / self.scale
        return r


class _Block(nn.Module):
    """One ``reduce → [pre] → expand+σ → [pool]`` unit of a reduce-first model.

    ``pre`` holds the stateless ops applied to the reduced features
    (:class:`~.block_ops.Flatten` / :class:`~.block_ops.L2Norm`), reusing their
    ``apply`` so the GD twin shares a single source of truth for the op math.
    """

    def __init__(
        self,
        *,
        reduce: _Reduce,
        pre: tuple[Flatten | L2Norm, ...],
        transform: nn.Module,
        weight: nn.Parameter | None,
        activation: str,
        normalize: bool,
        pool: Pool | None,
    ) -> None:
        super().__init__()
        self.reduce = reduce
        self.pre = pre
        # Dense transforms box their weight (no auto-registration), so register
        # it here to keep it in the state_dict; SORF transforms self-register.
        if weight is not None:
            self.weight = weight
        self.transform = transform
        self.c = nn.Parameter(torch.ones(1), requires_grad=False)
        self.activation = activation
        self.normalize = normalize
        self.pool = pool
        self.out_features = int(transform.out_features)  # type: ignore[attr-defined]

    def expand_input(self, a: Tensor) -> Tensor:
        """The reduced + pre-processed features fed to the expand (W_ℓ's input)."""
        r = self.reduce(a)
        for op in self.pre:
            r = op.apply(r)
        return r

    def forward(self, a: Tensor) -> Tensor:
        # z = σ(W_ℓ r / c) / √P  — the random-feature lift, then end-of-block pool.
        r = self.expand_input(a)
        transform = cast(StructuredTransform, self.transform)
        z = ACTIVATIONS[self.activation](transform.project(r) / self.c)
        z = z / math.sqrt(self.out_features)
        if self.pool is not None:
            z = self.pool.apply(z)
        return z


class SpectralModel(nn.Module):
    """Reduce-first random-features + eigenreduction + ridge-readout model.

    Built via :meth:`from_block_config` from a validated
    :class:`~neural_lofi.models.block_config.BlockedModelConfig` and an
    ``input_shape`` (``(D,)`` for FFN-mode or ``(C, H, W)`` for CNN-mode).  The
    model is ``L`` :class:`_Block`s (each ``reduce → [pre] → expand+σ → [pool]``)
    plus a final :class:`_Reduce` feeding the ridge readout; the wide post-σ
    output ``a_ℓ`` is carried between blocks.  The fitted tensors (``V`` /
    whitening / ``c``) are populated by
    :class:`~neural_lofi.training.spectral.SpectralTrainer`.
    """

    # Block-path attributes, populated by ``from_block_config``.  Declared here so
    # type-checkers know their concrete types through nn.Module's ``__getattr__``
    # (which otherwise widens every attribute to ``Tensor | Module``).
    blocks: nn.ModuleList
    final_reduction: _Reduce
    block_config: BlockedModelConfig
    readout_weight: nn.Parameter
    readout_bias: nn.Parameter
    _readout_dim: int

    @property
    def _init_style(self) -> InitStyle:
        """Architecture label derived from the input rank: 1-D → ffn, 3-D → cnn.

        Read-only and always consistent with ``input_shape`` — there is no
        stored flag to drift.  Used by the visualizer's FFN-mode guard.
        """
        return "ffn" if len(self.input_shape) == 1 else "cnn"

    @torch.no_grad()
    def forward_features(self, x: Tensor) -> list[Tensor]:
        """Return each block's wide post-σ output ``a_ℓ``.

        Symmetric to ``BackpropModel.forward_features``: runs the same forward
        as :meth:`forward` but collects the per-block intermediate outputs and
        stops before the final reduction + readout (conv features stay
        ``(N, C, H, W)``; FC features are ``(N, P)``).
        """
        feats: list[Tensor] = []
        a = x
        for block in self.blocks:
            a = cast(_Block, block)(a)
            feats.append(a)
        return feats

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        feats = self._final_features(self.forward_to_block(x))
        w = self.readout_weight
        if w.ndim == 2:
            return feats @ w.T + self.readout_bias
        return feats @ w + self.readout_bias

    @property
    def readout_input_dim(self) -> int:
        """Flattened width of the final-reduced features entering the readout."""
        return self._readout_dim

    # ------------------------------------------------------------------
    # Reduce-first block path (Stage 2): build / forward
    # ------------------------------------------------------------------

    @classmethod
    @torch.no_grad()
    def from_block_config(
        cls,
        config: BlockedModelConfig,
        *,
        input_shape: tuple[int, ...],
        seed: int = 0,
    ) -> SpectralModel:
        """Build a reduce-first ``SpectralModel`` from a validated block config.

        Runs :func:`validate_block_config` first (the checker, before any tensor
        is allocated), then allocates per-block storage by dry-running the shape
        flow.  Random expand weights are drawn from a ``seed``-seeded generator
        (same draw order as the expand-first path).  The fitted tensors (``V`` /
        whitening / ``c``) start as placeholders and are filled by
        :class:`~neural_lofi.training.spectral.SpectralTrainer`.
        """
        input_shape = tuple(input_shape)
        if len(input_shape) not in (1, 3):
            raise ValueError(
                f"input_shape must be 1-D (FFN) or 3-D (CNN); got {input_shape}"
            )
        validate_block_config(config, input_shape)

        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.input_shape = input_shape
        self.block_config = config

        gen = torch.Generator()
        gen.manual_seed(seed)

        def _layer_seed() -> int:
            return int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())

        flat = len(input_shape) == 1
        if flat:
            width, h, w = input_shape[0], 0, 0
        else:
            width, h, w = input_shape

        blocks = nn.ModuleList()
        for bcfg in config.blocks:
            keep = _reduce_keep(bcfg.reduce)
            reduce = _Reduce(
                in_width=width, keep=keep, is_conv=not flat, whiten=bcfg.reduce.whiten
            )
            if keep is not None:
                width = keep
            for op in bcfg.pre:
                if op == "flatten":
                    width, h, w, flat = width * h * w, 0, 0, True
            e = bcfg.expand
            weight: nn.Parameter | None = None
            if e.is_conv:
                ks = e.kernel_size
                assert ks is not None  # is_conv ⇔ kernel_size set (checker-enforced)
                if e.transform == "sorf":
                    transform: nn.Module = SORFConvTransform(
                        width,
                        e.p,
                        kernel_size=ks,
                        padding=e.padding,
                        stride=e.stride,
                        seed=_layer_seed(),
                    )
                else:
                    wt = torch.randn(e.p, width, ks, ks, generator=gen)
                    weight = nn.Parameter(wt, requires_grad=False)
                    transform = DenseConvTransform(
                        weight, padding=e.padding, stride=e.stride
                    )
                h = (h + 2 * e.padding - ks) // e.stride + 1
                w = (w + 2 * e.padding - ks) // e.stride + 1
                width = e.p
            else:
                if e.transform == "sorf":
                    transform = SORFTransform(width, e.p, seed=_layer_seed())
                else:
                    wt = torch.randn(width, e.p, generator=gen)
                    weight = nn.Parameter(wt, requires_grad=False)
                    transform = DenseTransform(weight)
                width = e.p
            pool: Pool | None = None
            if bcfg.pool is not None:
                pc = bcfg.pool
                pool = Pool(
                    mode=pc.mode,
                    kernel_size=pc.kernel_size,
                    stride=pc.stride,
                    padding=pc.padding,
                )
                s = pc.effective_stride
                h = (h + 2 * pc.padding - pc.kernel_size) // s + 1
                w = (w + 2 * pc.padding - pc.kernel_size) // s + 1
            blocks.append(
                _Block(
                    reduce=reduce,
                    pre=tuple(PRE_OPS[op]() for op in bcfg.pre),
                    transform=transform,
                    weight=weight,
                    activation=e.activation,
                    normalize=e.normalize,
                    pool=pool,
                )
            )

        self.blocks = blocks
        fkeep = _reduce_keep(config.final_reduce)
        self.final_reduction = _Reduce(
            in_width=width,
            keep=fkeep,
            is_conv=not flat,
            whiten=config.final_reduce.whiten,
        )
        out_width = fkeep if fkeep is not None else width
        self._readout_dim = out_width * (1 if flat else h * w)
        self.readout_weight = nn.Parameter(
            torch.zeros(self._readout_dim), requires_grad=False
        )
        self.readout_bias = nn.Parameter(torch.zeros(1), requires_grad=False)
        return self

    def forward_to_block(self, x: Tensor, stop: int | None = None) -> Tensor:
        """Apply blocks ``[0, stop)`` and return the wide output ``a_{stop-1}``.

        ``stop`` is exclusive (defaults to all blocks), so ``forward_to_block(x,
        stop=ℓ)`` is the *input* to block ``ℓ`` and ``forward_to_block(x,
        stop=ℓ+1)`` is block ``ℓ``'s output (the tensor block ℓ+1's reduction is
        fit on).  Read-only but autograd-capable.
        """
        stop = len(self.blocks) if stop is None else stop
        a = x
        for i in range(stop):
            a = self.blocks[i](a)
        return a

    def _final_features(self, a: Tensor) -> Tensor:
        """The flattened final-reduced features the ridge readout reads."""
        r = self.final_reduction(a)
        return r.flatten(1) if r.ndim > 2 else r
