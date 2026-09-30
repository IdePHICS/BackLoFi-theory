"""Backprop-trained **folded GD twin** of a reduce-first :class:`SpectralModel`.

The unified backprop counterpart to :class:`SpectralModel`: a single class
covering both feedforward (FFN) and convolutional (CNN) architectures, with the
mode auto-detected from the rank of ``input_shape`` (1-D -> FFN, 3-D -> CNN) and
stored as ``_init_style``.  Unlike :class:`SpectralModel`, every layer here is a
standard trainable layer (``nn.Linear`` / ``nn.Conv2d``) rather than a random
projection + eigenvector projection.

Built exclusively from a reduce-first :class:`BlockedModelConfig` via
:meth:`BackpropModel.from_block_config` — one trainable layer per block of width
``expand.p`` (the block's ``reduce`` is *folded* into that single weight;
``pre`` flatten/l2norm before, ``pool`` after).  ``input_shape=(D,)`` gives a
fully-connected stack; ``input_shape=(C, H, W)`` gives a mixed conv/pool/flatten
stack.  Trained via :class:`neural_lofi.training.BackpropTrainer`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn

from ..utils.helpers import ACTIVATION_MODULES
from .block_config import BlockedModelConfig, parse_block_config, validate_block_config
from .block_ops import InitStyle, L2Norm


class ChannelL2Norm(nn.Module):
    """Per-location L2 normalization across channels for 4D feature maps.

    The ``nn.Module`` wrapper of the stateless :class:`~.block_ops.L2Norm` op so
    it can live in the GD twin's ``nn.ModuleList``.  It delegates the norm math to
    ``L2Norm.apply`` so the GD twin and the spectral model share a single
    definition (identical epsilon convention).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op = L2Norm()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"ChannelL2Norm expects input shape (N, C, H, W), got {tuple(x.shape)}"
            )
        return self._op.apply(x)


class BackpropModel(nn.Module):
    """Trainable network (FFN or CNN) — the folded GD twin of a block config.

    The backprop counterpart to :class:`SpectralModel`, built only via
    :meth:`from_block_config`.  The mode is auto-detected from ``input_shape``
    (1-D -> ``"ffn"``, 3-D -> ``"cnn"``) and stored as ``_init_style``.  Each
    block contributes one trainable layer of width ``expand.p``; the block's
    ``reduce`` is folded into that weight, so there is no per-layer
    spec list.
    """

    # Instance attributes (annotated at class level so external access is typed
    # past ``nn.Module.__getattr__``'s ``Tensor | Module`` fallback).
    _init_style: InitStyle
    input_shape: tuple[int, ...]
    block_config: BlockedModelConfig
    layers: nn.ModuleList
    classifier: nn.Linear

    @classmethod
    def from_block_config(
        cls,
        config: BlockedModelConfig,
        *,
        input_shape: tuple[int, ...],
        output_dim: int = 1,
        classifier_bias: bool = True,
    ) -> BackpropModel:
        """Build the **folded GD twin** of a reduce-first block config.

        Per block, one trainable layer (``Conv2d``/``Linear``) of width
        ``expand.p`` with the expand's geometry + activation, the ``pre``
        (flatten/l2norm) applied before and the ``pool`` after; the classifier
        maps the last block's flat width to ``output_dim``.  The block's
        ``reduce`` is **folded** — it is not a separate trainable layer (the
        warm-start initialises the single weight to ``W·V``; the final reduction
        folds into the classifier).  Trainable layers are ``nn.ModuleDict``
        (``conv``/``linear`` + ``activation``), matching the ``__init__`` layout.
        """
        input_shape = tuple(input_shape)
        validate_block_config(config, input_shape)
        if len(input_shape) == 1:
            init_style: InitStyle = "ffn"
        elif len(input_shape) == 3:
            init_style = "cnn"
        else:
            raise ValueError(
                "input_shape must be 1-D (FFN) or 3-D (C, H, W) (CNN); "
                f"got shape={input_shape}"
            )

        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self._init_style = init_style
        self.input_shape = input_shape
        self.block_config = config
        self.layers = nn.ModuleList()

        flat = len(input_shape) == 1
        if flat:
            feature_dim, c_in, h, w = input_shape[0], 0, 0, 0
        else:
            c_in, h, w = input_shape
            feature_dim = c_in * h * w

        for bcfg in config.blocks:
            for op in bcfg.pre:
                if op == "flatten":
                    self.layers.append(nn.Flatten())
                    flat, feature_dim = True, c_in * h * w
                else:  # l2norm
                    self.layers.append(ChannelL2Norm())
            e = bcfg.expand
            if e.is_conv:
                assert e.kernel_size is not None
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "conv": nn.Conv2d(
                                c_in,
                                e.p,
                                kernel_size=e.kernel_size,
                                stride=e.stride,
                                padding=e.padding,
                                bias=False,
                            ),
                            "activation": _make_activation(e.activation),
                        }
                    )
                )
                h = (h + 2 * e.padding - e.kernel_size) // e.stride + 1
                w = (w + 2 * e.padding - e.kernel_size) // e.stride + 1
                c_in, feature_dim = e.p, e.p * h * w
            else:
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "linear": nn.Linear(feature_dim, e.p, bias=False),
                            "activation": _make_activation(e.activation),
                        }
                    )
                )
                feature_dim = e.p
            if bcfg.pool is not None:
                pc = bcfg.pool
                stride = pc.kernel_size if pc.stride is None else pc.stride
                pool_kw = {
                    "kernel_size": pc.kernel_size,
                    "stride": stride,
                    "padding": pc.padding,
                }
                self.layers.append(
                    nn.AvgPool2d(**pool_kw)
                    if pc.mode.strip().lower() == "avg"
                    else nn.MaxPool2d(**pool_kw)
                )
                h = (h + 2 * pc.padding - pc.kernel_size) // stride + 1
                w = (w + 2 * pc.padding - pc.kernel_size) // stride + 1
                feature_dim = c_in * h * w

        self.classifier = nn.Linear(feature_dim, output_dim, bias=classifier_bias)
        return self

    def forward_features(self, x: Tensor) -> list[Tensor]:
        """Return feature tensors after each configured layer.

        Type-driven (no spec lookup): a trainable layer is an ``nn.ModuleDict``
        (``conv``/``linear`` + ``activation``); every other entry is a stateless
        ``nn.Module`` (pool / flatten / l2norm) applied directly.
        """
        h = x
        outputs: list[Tensor] = []
        for layer in self.layers:
            if isinstance(layer, nn.ModuleDict):
                op = layer["conv"] if "conv" in layer else layer["linear"]
                h = layer["activation"](op(h))
            else:
                h = layer(h)
            outputs.append(h)
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self.forward_features(x)
        h = outputs[-1] if outputs else x
        if h.ndim > 2:
            h = h.flatten(1)
        return self.classifier(h)

    # ------------------------------------------------------------------ #
    # Self-describing checkpoint
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: str | Path) -> None:
        """Save weights plus the block config needed to rebuild this model.

        Bundles ``state_dict`` with ``input_shape`` / ``output_dim`` /
        ``classifier_bias`` and the reduce-first ``block_config`` as a nested
        dict (``asdict`` round-trips through :func:`parse_block_config`);
        :meth:`from_checkpoint` rebuilds via :meth:`from_block_config`.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_shape": tuple(self.input_shape),
                "output_dim": int(self.classifier.out_features),
                "classifier_bias": self.classifier.bias is not None,
                "block_config": asdict(self.block_config),
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> BackpropModel:
        """Rebuild a :class:`BackpropModel` saved by :meth:`save_checkpoint`."""
        ckpt = torch.load(path, weights_only=False)
        model = cls.from_block_config(
            parse_block_config(ckpt["block_config"]),
            input_shape=tuple(ckpt["input_shape"]),
            output_dim=int(ckpt["output_dim"]),
            classifier_bias=bool(ckpt["classifier_bias"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model


def _make_activation(name: str) -> nn.Module:
    """Deep-copy the named activation module so each layer owns its own instance."""
    base = ACTIVATION_MODULES.get(name)
    if base is None:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Available: {sorted(ACTIVATION_MODULES.keys())}"
        )
    return copy.deepcopy(base)
