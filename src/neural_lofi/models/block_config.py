"""Reduce-first block configuration schema + pre-build validation.

A reduce-first model is configured as a list of *blocks* plus a final reduction::

    a_ℓ = pool( σ( W_ℓ · ( V_{ℓ-1} · a_{ℓ-1} ) ) )
          └pool┘  └σ┘   └expand W_ℓ┘  └reduce V_{ℓ-1}┘

Each block declares, in application order:

* ``reduce`` — the input reduction ``V_{ℓ-1}`` of the *wide* incoming features
  (``k=null`` ⇒ identity, i.e. no reduction; also the old "keep full width" /
  ``eigenreduction: false`` case).
* ``pre`` — stateless ops applied to the *reduced* features (``flatten`` /
  ``l2norm``).  ``flatten`` is how a conv→fc transition is expressed: the reduce
  channel-reduces the 4-D map, then ``flatten`` makes it a vector for the fc
  ``expand`` (this order keeps a pool-free CNN bit-identical to the legacy model).
* ``expand`` — the random expansion ``W_ℓ`` + activation (conv when
  ``kernel_size`` is given, else fully-connected).
* ``pool`` — an optional **end-of-block** pool on the wide post-σ features.

``final_reduce`` maps the last block's wide output ``a_{L-1}`` to the features the
ridge readout reads.

YAML (nested) form::

    blocks:
      - reduce: {k: null}                    # V_{-1} = identity
        expand: {p: 500, activation: relu, kernel_size: 3, padding: 1}
      - reduce: {k: 10, whiten: true, linear_svd: true, linear_k: 10}
        expand: {p: 500, activation: relu, kernel_size: 3, padding: 1}
        pool:   {mode: max, kernel_size: 2}
      - reduce: {k: 10, whiten: true, linear_svd: true, linear_k: 10}
        pre:    [flatten]                     # conv→fc: flatten the reduced 4-D map
        expand: {p: 200, activation: relu}
    final_reduce: {k: 100, whiten: true, linear_svd: true, linear_k: 10}

:func:`parse_block_config` turns the nested dict into typed dataclasses;
:func:`validate_block_config` dry-runs the shape flow against the model's input
shape *before* any tensor is allocated, raising clear structural / dimensional /
vector errors instead of failing deep inside the build.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..utils.helpers import ACTIVATIONS

# Stateless ops legal in a block's ``pre`` list (applied to the reduced features,
# between the reduce and the expand).  ``pool`` is NOT here — it is an
# end-of-block op carried in ``BlockConfig.pool``.
_PRE_OPS: frozenset[str] = frozenset({"flatten", "l2norm"})
_POOL_MODES: frozenset[str] = frozenset({"max", "avg"})
_TRANSFORMS: frozenset[str] = frozenset({"dense", "sorf"})


class BlockConfigError(ValueError):
    """Raised when a block configuration is structurally or dimensionally invalid."""


# ---------------------------------------------------------------------- #
# Schema dataclasses
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReduceConfig:
    """Input reduction ``V_{ℓ-1}`` of a block (or the final reduction).

    ``k=None`` is the identity reduction (no eigenreduction — keep the full
    incoming width).  ``whiten`` / ``linear_svd`` / ``include_linear_mean`` are
    only meaningful when ``k`` is set; they describe how ``V`` is *fit* (validated
    in :func:`validate_block_config`).
    """

    k: int | None = None
    whiten: bool = False
    whiten_mode: str = "std"
    linear_svd: bool = False
    linear_k: int = 0
    include_linear_mean: bool = False
    orthogonalize_to_mean: bool = True


@dataclass(frozen=True)
class ExpandConfig:
    """Random expansion ``W_ℓ`` + activation of a block.

    Convolutional when ``kernel_size`` is set, fully-connected otherwise.
    ``normalize_input`` defaults to the historical per-kind value (fc ``True``,
    conv ``False``) when left ``None``.
    """

    p: int
    activation: str = "relu"
    transform: str = "dense"
    kernel_size: int | None = None
    padding: int = 0
    stride: int = 1
    normalize_input: bool | None = None

    @property
    def is_conv(self) -> bool:
        return self.kernel_size is not None

    @property
    def normalize(self) -> bool:
        """Resolved ``normalize_input`` (fc→True, conv→False when unset)."""
        if self.normalize_input is not None:
            return self.normalize_input
        return not self.is_conv


@dataclass(frozen=True)
class PoolConfig:
    """End-of-block spatial pool (on the wide post-σ features)."""

    mode: str = "max"
    kernel_size: int = 2
    stride: int | None = None
    padding: int = 0

    @property
    def effective_stride(self) -> int:
        return self.kernel_size if self.stride is None else self.stride


@dataclass(frozen=True)
class BlockConfig:
    """One reduce → [pre] → expand+σ → [pool] unit."""

    expand: ExpandConfig
    reduce: ReduceConfig = field(default_factory=ReduceConfig)
    pre: tuple[str, ...] = ()
    pool: PoolConfig | None = None


@dataclass(frozen=True)
class BlockedModelConfig:
    """A reduce-first model: ``L`` blocks + a final reduction → readout."""

    blocks: tuple[BlockConfig, ...]
    final_reduce: ReduceConfig = field(default_factory=ReduceConfig)


# ---------------------------------------------------------------------- #
# Parsing (nested dict → dataclasses)
# ---------------------------------------------------------------------- #


def _require_dict(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BlockConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(d: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise BlockConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed {sorted(allowed)}"
        )


_REDUCE_KEYS = frozenset(
    {
        "k",
        "whiten",
        "whiten_mode",
        "linear_svd",
        "linear_k",
        "include_linear_mean",
        "orthogonalize_to_mean",
    }
)
_WHITEN_MODES = frozenset({"std", "rms", "centered"})
_EXPAND_KEYS = frozenset(
    {
        "p",
        "activation",
        "transform",
        "kernel_size",
        "padding",
        "stride",
        "normalize_input",
    }
)
_POOL_KEYS = frozenset({"mode", "kernel_size", "stride", "padding"})
_BLOCK_KEYS = frozenset({"reduce", "expand", "pre", "pool"})


def parse_reduce(d: dict[str, Any] | None, where: str) -> ReduceConfig:
    if d is None:
        return ReduceConfig()
    d = _require_dict(d, where)
    _reject_unknown(d, _REDUCE_KEYS, where)
    mode = str(d.get("whiten_mode", "std")).lower()
    if mode not in _WHITEN_MODES:
        raise BlockConfigError(
            f"{where}: whiten_mode must be one of {sorted(_WHITEN_MODES)}, got {mode!r}"
        )
    return ReduceConfig(
        k=d.get("k"),
        whiten=bool(d.get("whiten", False)),
        whiten_mode=mode,
        linear_svd=bool(d.get("linear_svd", False)),
        linear_k=int(d.get("linear_k", 0)),
        include_linear_mean=bool(d.get("include_linear_mean", False)),
        orthogonalize_to_mean=bool(d.get("orthogonalize_to_mean", True)),
    )


def parse_expand(d: dict[str, Any], where: str) -> ExpandConfig:
    d = _require_dict(d, where)
    _reject_unknown(d, _EXPAND_KEYS, where)
    if "p" not in d:
        raise BlockConfigError(f"{where}: expand requires 'p'")
    ks = d.get("kernel_size")
    return ExpandConfig(
        p=int(d["p"]),
        activation=str(d.get("activation", "relu")),
        transform=str(d.get("transform", "dense")),
        kernel_size=None if ks is None else int(ks),
        padding=int(d.get("padding", 0)),
        stride=int(d.get("stride", 1)),
        normalize_input=d.get("normalize_input"),
    )


def parse_pool(d: dict[str, Any] | None, where: str) -> PoolConfig | None:
    if d is None:
        return None
    d = _require_dict(d, where)
    _reject_unknown(d, _POOL_KEYS, where)
    stride = d.get("stride")
    return PoolConfig(
        mode=str(d.get("mode", "max")),
        kernel_size=int(d.get("kernel_size", 2)),
        stride=None if stride is None else int(stride),
        padding=int(d.get("padding", 0)),
    )


def parse_block(d: dict[str, Any], where: str) -> BlockConfig:
    d = _require_dict(d, where)
    _reject_unknown(d, _BLOCK_KEYS, where)
    if "expand" not in d:
        raise BlockConfigError(f"{where}: block requires an 'expand'")
    pre = tuple(str(op) for op in d.get("pre", ()))
    return BlockConfig(
        expand=parse_expand(d["expand"], f"{where}.expand"),
        reduce=parse_reduce(d.get("reduce"), f"{where}.reduce"),
        pre=pre,
        pool=parse_pool(d.get("pool"), f"{where}.pool"),
    )


def parse_block_config(d: dict[str, Any]) -> BlockedModelConfig:
    """Parse a nested ``{blocks: [...], final_reduce: {...}}`` dict.

    Accepts a plain ``dict`` (convert OmegaConf via ``OmegaConf.to_object`` first).
    Rejects unknown keys at every level so a typo fails loudly here rather than
    silently mis-specifying an experiment.
    """
    d = _require_dict(d, "config")
    _reject_unknown(d, frozenset({"blocks", "final_reduce"}), "config")
    raw_blocks = d.get("blocks", [])
    if not isinstance(raw_blocks, (list, tuple)):
        raise BlockConfigError("'blocks' must be a list")
    blocks = tuple(parse_block(b, f"blocks[{i}]") for i, b in enumerate(raw_blocks))
    return BlockedModelConfig(
        blocks=blocks,
        final_reduce=parse_reduce(d.get("final_reduce"), "final_reduce"),
    )


# ---------------------------------------------------------------------- #
# Validation (dry-run the shape flow before building)
# ---------------------------------------------------------------------- #


@dataclass
class _Shape:
    """The running feature shape during the validation dry-run.

    ``width`` is the vector dim when ``flat`` else the image channel count — the
    width a reduction operates over; ``h`` / ``w`` are the spatial dims (image only).
    """

    flat: bool  # True ⇒ a (N, width) vector; False ⇒ a (N, width, H, W) image
    width: int
    h: int = 0
    w: int = 0


def _check_reduce_fit_options(r: ReduceConfig, where: str) -> None:
    """Mirror the weighted-spec ``__post_init__`` rules for a reduction."""
    if r.k is None:
        # Identity reduction: the fit-shaping flags are meaningless.
        for flag, name in (
            (r.whiten, "whiten"),
            (r.linear_svd, "linear_svd"),
            (r.include_linear_mean, "include_linear_mean"),
        ):
            if flag:
                raise BlockConfigError(
                    f"{where}: {name}=True requires a reduction (set reduce.k)"
                )
        return
    if r.k <= 0:
        raise BlockConfigError(f"{where}: reduce.k must be > 0 or null, got {r.k}")
    if r.linear_svd and r.include_linear_mean:
        raise BlockConfigError(
            f"{where}: linear_svd and include_linear_mean are mutually exclusive"
        )
    if r.linear_svd and r.linear_k <= 0:
        raise BlockConfigError(
            f"{where}: linear_svd=True requires linear_k > 0 (= number of classes)"
        )


def _validate_reduce(shape: _Shape, r: ReduceConfig, where: str) -> _Shape:
    _check_reduce_fit_options(r, where)
    if r.k is None:
        return shape  # identity
    # Output width incl. the linear-SVD / linear-mean prepend.
    if r.linear_svd:
        keep = r.k + r.linear_k
    elif r.include_linear_mean:
        keep = r.k + 1
    else:
        keep = r.k
    if keep > shape.width:
        raise BlockConfigError(
            f"{where}: reduction keeps {keep} directions "
            f"({'k+linear_k' if r.linear_svd else 'k'}) but the incoming features "
            f"have only {shape.width} ({'dims' if shape.flat else 'channels'})"
        )
    if shape.flat:
        return _Shape(True, keep)
    return _Shape(False, keep, shape.h, shape.w)


def _validate_pre(shape: _Shape, ops: tuple[str, ...], where: str) -> _Shape:
    for op in ops:
        if op not in _PRE_OPS:
            raise BlockConfigError(
                f"{where}: unknown pre op {op!r}; allowed {sorted(_PRE_OPS)} "
                "(pooling is an end-of-block 'pool', not a pre op)"
            )
        if shape.flat:
            raise BlockConfigError(
                f"{where}: {op!r} expects image features but the current "
                "representation is already flat"
            )
        if op == "flatten":
            shape = _Shape(True, shape.width * shape.h * shape.w)
        # l2norm leaves the shape unchanged.
    return shape


def _validate_expand(shape: _Shape, e: ExpandConfig, where: str) -> _Shape:
    if e.p <= 0:
        raise BlockConfigError(f"{where}: expand.p must be > 0, got {e.p}")
    if e.activation not in ACTIVATIONS:
        raise BlockConfigError(
            f"{where}: unknown activation {e.activation!r}; "
            f"one of {sorted(ACTIVATIONS)}"
        )
    if e.transform not in _TRANSFORMS:
        raise BlockConfigError(
            f"{where}: transform must be one of {sorted(_TRANSFORMS)}, "
            f"got {e.transform!r}"
        )
    if e.is_conv:
        if shape.flat:
            raise BlockConfigError(
                f"{where}: conv expand (kernel_size set) needs image features, "
                "but the current representation is flat (insert no flatten before "
                "a conv)"
            )
        ks = e.kernel_size
        assert ks is not None
        h = (shape.h + 2 * e.padding - ks) // e.stride + 1
        w = (shape.w + 2 * e.padding - ks) // e.stride + 1
        if h <= 0 or w <= 0:
            raise BlockConfigError(
                f"{where}: conv (kernel_size={ks}, padding={e.padding}, "
                f"stride={e.stride}) collapses spatial dims {shape.h}x{shape.w}"
            )
        return _Shape(False, e.p, h, w)
    if not shape.flat:
        raise BlockConfigError(
            f"{where}: fully-connected expand (no kernel_size) needs flat "
            "features; add 'flatten' to the block's pre list"
        )
    return _Shape(True, e.p)


def _validate_pool(shape: _Shape, p: PoolConfig, where: str) -> _Shape:
    if shape.flat:
        raise BlockConfigError(f"{where}: pool needs image features, got flat")
    if p.mode not in _POOL_MODES:
        raise BlockConfigError(
            f"{where}: pool.mode must be one of {sorted(_POOL_MODES)}, got {p.mode!r}"
        )
    if p.kernel_size <= 0:
        raise BlockConfigError(f"{where}: pool.kernel_size must be > 0")
    s = p.effective_stride
    h = (shape.h + 2 * p.padding - p.kernel_size) // s + 1
    w = (shape.w + 2 * p.padding - p.kernel_size) // s + 1
    if h <= 0 or w <= 0:
        raise BlockConfigError(
            f"{where}: pool (kernel_size={p.kernel_size}, stride={s}) collapses "
            f"spatial dims {shape.h}x{shape.w}"
        )
    return _Shape(False, shape.width, h, w)


def validate_block_config(
    cfg: BlockedModelConfig, input_shape: tuple[int, ...]
) -> None:
    """Dry-run the shape flow of a block config; raise on the first inconsistency.

    Verifies, *before any tensor is allocated*: structural ordering (conv expands
    and pools only on image features; flatten only on image features; fc expands
    only on flat features), the dimensional chain (a reduction never keeps more
    directions than its input has; convs/pools never collapse the spatial map),
    and the per-reduction vector rules (``linear_svd`` ⇒ ``linear_k>0``;
    ``whiten`` / ``linear_svd`` / ``include_linear_mean`` require an actual
    reduction).  Raises :class:`BlockConfigError` with a located message.

    ``input_shape`` is ``(D,)`` (FFN) or ``(C, H, W)`` (CNN), matching
    ``SpectralModel``.
    """
    if len(input_shape) == 1:
        shape = _Shape(True, int(input_shape[0]))
    elif len(input_shape) == 3:
        c, h, w = (int(v) for v in input_shape)
        shape = _Shape(False, c, h, w)
    else:
        raise BlockConfigError(
            f"input_shape must be 1-D (FFN) or 3-D (CNN), got {tuple(input_shape)}"
        )

    for i, block in enumerate(cfg.blocks):
        where = f"blocks[{i}]"
        shape = _validate_reduce(shape, block.reduce, f"{where}.reduce")
        shape = _validate_pre(shape, block.pre, where)
        shape = _validate_expand(shape, block.expand, f"{where}.expand")
        if block.pool is not None:
            shape = _validate_pool(shape, block.pool, f"{where}.pool")

    _validate_reduce(shape, cfg.final_reduce, "final_reduce")


def ffn_blocks(
    layers: Sequence[tuple[int, int | None, dict[str, Any]]],
) -> BlockedModelConfig:
    """Compact ``[(p, k, opts), ...]`` FFN authoring → reduce-first block config.

    The ergonomic builder for programmatic all-fully-connected models (sweeps,
    relu/kernel studies).  Layer ℓ's ``k`` (+ its reduce-side opts) becomes block
    ℓ+1's ``reduce``; the last layer's becomes ``final_reduce``; block 0's reduce
    is identity (``V_{-1}=I``).  ``k=None`` is the identity reduction (the old
    ``eigenreduction=False`` case).  ``opts`` carries the expand ``activation`` and
    any reduce options (``whiten`` / ``linear_svd`` / ``linear_k`` /
    ``include_linear_mean`` / ``orthogonalize_to_mean``).
    """

    def _reduce(k: int | None, opts: dict[str, Any]) -> dict[str, Any]:
        if k is None:
            return {"k": None}
        return {"k": k, **{key: opts[key] for key in opts if key != "activation"}}

    blocks: list[dict[str, Any]] = []
    for i, (p, _k, opts) in enumerate(layers):
        reduce = {"k": None} if i == 0 else _reduce(layers[i - 1][1], layers[i - 1][2])
        blocks.append(
            {
                "reduce": reduce,
                "expand": {"p": p, "activation": opts.get("activation", "relu")},
            }
        )
    final = _reduce(layers[-1][1], layers[-1][2]) if layers else {"k": None}
    return parse_block_config({"blocks": blocks, "final_reduce": final})
