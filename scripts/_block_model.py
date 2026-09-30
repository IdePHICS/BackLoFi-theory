"""Bridge: build a reduce-first ``SpectralModel`` from an OmegaConf config or a
flat ``layer_specs`` dict list.

Generators call :func:`build_block_model` when a config is in the native
``blocks:`` schema (detected by :func:`is_block_config`); generators that still
author a flat ``layer_specs`` dict list go through
:func:`block_model_from_layer_dicts`, which re-brackets the dicts into the block
schema via :func:`layer_specs_to_blocks`.

:func:`layer_specs_to_blocks` re-brackets a legacy ``layer_specs`` list into the
``blocks`` / ``final_reduce`` schema — its output is a *native* block config
(validated by ``validate_block_config``), not a runtime shim.  The off-by-one
(block ℓ's reduce = layer ℓ-1's reduce-fields) is the whole content of the
expand-first ≡ reduce-first equivalence.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from neural_lofi.models.block_config import parse_block_config
from neural_lofi.models.spectral import SpectralModel

# Legacy ``layer_type`` aliases (mirrors layer_specs.LAYER_TYPE_ALIASES).
_ALIAS = {"random_features": "fully_connected"}
_WEIGHTED = {"fully_connected", "conv"}
# Reduce-side fields read off a weighted layer (the eigenreduction V_ℓ).
_REDUCE_KEYS = (
    "whiten",
    "whiten_mode",
    "linear_svd",
    "linear_k",
    "include_linear_mean",
    "orthogonalize_to_mean",
)


def is_block_config(cfg: DictConfig) -> bool:
    """True if ``cfg`` uses the native reduce-first ``blocks:`` schema."""
    return "blocks" in cfg


def blocks_to_layer_dicts(cfg: DictConfig) -> list[dict[str, Any]]:
    """Reconstruct the full legacy ``layer_specs`` list from a block config.

    The faithful inverse of :func:`layer_specs_to_blocks` — used for
    **result-filename / logging continuity** so every generator's existing
    filename builder works unchanged.  Per block: emit its ``pre`` ops
    (flatten / l2norm), then the weighted layer (``p`` / activation / conv geometry
    from its expand; ``k`` and reduce-side fields from the *next* block's reduce,
    or ``final_reduce`` for the last — the off-by-one), then its ``pool``.
    """
    blocks = OmegaConf.to_object(cfg.blocks)
    final = (
        OmegaConf.to_object(cfg.final_reduce) if "final_reduce" in cfg else {"k": None}
    )
    out: list[dict[str, Any]] = []
    for i, block in enumerate(blocks):
        for op in block.get("pre", []):
            out.append({"layer_type": op})
        exp = dict(block["expand"])
        nxt = blocks[i + 1]["reduce"] if i + 1 < len(blocks) else final
        k = nxt.get("k")
        entry: dict[str, Any] = {"p": exp.get("p"), "activation": exp.get("activation")}
        if "kernel_size" in exp:
            entry["layer_type"] = "conv"
            for key in ("kernel_size", "padding", "stride"):
                if key in exp:
                    entry[key] = exp[key]
        else:
            entry["layer_type"] = "fully_connected"
        for key in ("transform", "normalize_input"):
            if key in exp:
                entry[key] = exp[key]
        if k is None:
            entry["eigenreduction"] = False
        else:
            entry["eigenreduction"] = True
            entry["k"] = k
            for key in (
                "whiten",
                "whiten_mode",
                "linear_svd",
                "linear_k",
                "include_linear_mean",
                "orthogonalize_to_mean",
            ):
                if key in nxt:
                    entry[key] = nxt[key]
        out.append(entry)
        if "pool" in block:
            pool = block["pool"]
            pentry: dict[str, Any] = {"layer_type": "pooling"}
            if "mode" in pool:
                pentry["pool_mode"] = pool["mode"]
            if "kernel_size" in pool:
                pentry["pool_kernel_size"] = pool["kernel_size"]
            if "stride" in pool:
                pentry["pool_stride"] = pool["stride"]
            if "padding" in pool:
                pentry["pool_padding"] = pool["padding"]
            out.append(pentry)
    return out


def build_block_model(
    cfg: DictConfig, *, input_shape: tuple[int, ...], seed: int = 0
) -> SpectralModel:
    """Build a reduce-first ``SpectralModel`` from a config's ``blocks`` section.

    Converts the OmegaConf ``blocks`` / ``final_reduce`` to plain dicts, validates
    them (:func:`validate_block_config`, via ``from_block_config``), and builds.
    """
    section: dict[str, Any] = {"blocks": OmegaConf.to_object(cfg.blocks)}
    if "final_reduce" in cfg:
        section["final_reduce"] = OmegaConf.to_object(cfg.final_reduce)
    parsed = parse_block_config(section)
    return SpectralModel.from_block_config(
        parsed, input_shape=tuple(input_shape), seed=seed
    )


_GEOM_KEYS = ("kernel_size", "padding", "stride")


def _expand_fields(
    spec: dict[str, Any], *, is_conv: bool, conv_defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Expand-side (``W_ℓ``) fields of a weighted layer dict → expand config.

    For conv layers, geometry (``kernel_size`` / ``padding`` / ``stride``) is taken
    from the layer, falling back to ``conv_defaults`` (the config's top-level conv
    keys, which the legacy CNN loader applied via ``cfg.get``).
    """
    if spec.get("p") is None:
        raise ValueError(
            "cannot convert a layer with p=null (data-dependent width) to a "
            "block expand; set an explicit p first."
        )
    out: dict[str, Any] = {"p": spec["p"]}
    if "activation" in spec:
        out["activation"] = spec["activation"]
    if "transform" in spec:
        out["transform"] = spec["transform"]
    if "normalize_input" in spec:
        out["normalize_input"] = spec["normalize_input"]
    if is_conv:
        defaults = conv_defaults or {}
        for key in _GEOM_KEYS:
            if key in spec:
                out[key] = spec[key]
            elif key in defaults:
                out[key] = defaults[key]
        out.setdefault("kernel_size", 5)  # legacy CNN loader default
    return out


def _reduce_fields(spec: dict[str, Any]) -> dict[str, Any]:
    """Reduce-side (``V_ℓ``) fields of a weighted layer dict → reduce config.

    ``eigenreduction: false`` (keep full width, no reduction) maps to ``k: null``.
    """
    if not spec.get("eigenreduction", True):
        return {"k": None}
    out: dict[str, Any] = {"k": spec.get("k")}
    for key in _REDUCE_KEYS:
        if key in spec:
            out[key] = spec[key]
    return out


def _infer_layer_type(
    spec: dict[str, Any], *, conv_defaults: dict[str, Any] | None
) -> str:
    """Infer a layer's canonical type, matching the legacy loaders' defaults.

    Explicit ``layer_type`` wins (aliases resolved).  Missing ``layer_type`` →
    ``conv`` when the layer or the config carries conv geometry (the CNN loader
    defaults to conv), else ``fully_connected`` (the FFN loader default).
    """
    raw = spec.get("layer_type")
    if raw is not None:
        return _ALIAS.get(raw, raw)
    if "kernel_size" in spec or conv_defaults:
        return "conv"
    return "fully_connected"


def layer_specs_to_blocks(
    layer_specs: list[dict[str, Any]],
    *,
    first_reduce: dict[str, Any] | None = None,
    conv_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-bracket a legacy ``layer_specs`` list into a ``{blocks, final_reduce}`` dict.

    Pure dict transformation (the one-time config migration).  ``pool`` →
    preceding block's ``pool`` (end-of-block); ``flatten`` / ``l2norm`` → next
    block's ``pre``; block 0's reduce is ``first_reduce`` (default identity).
    ``conv_defaults`` supplies top-level conv geometry for conv layers that omit
    it (and disambiguates a missing ``layer_type`` as conv).
    """
    blocks: list[dict[str, Any]] = []
    pending_pre: list[str] = []
    reduce_for_next: dict[str, Any] = first_reduce or {"k": None}
    for raw in layer_specs:
        spec = dict(raw)
        lt = _infer_layer_type(spec, conv_defaults=conv_defaults)
        if lt in _WEIGHTED:
            block: dict[str, Any] = {
                "reduce": reduce_for_next,
                "expand": _expand_fields(
                    spec, is_conv=lt == "conv", conv_defaults=conv_defaults
                ),
            }
            if pending_pre:
                block["pre"] = pending_pre
                pending_pre = []
            blocks.append(block)
            reduce_for_next = _reduce_fields(spec)
        elif lt == "pooling":
            if not blocks:
                raise ValueError("pooling before any weighted layer")
            pool: dict[str, Any] = {}
            if "pool_mode" in spec:
                pool["mode"] = spec["pool_mode"]
            if "pool_kernel_size" in spec:
                pool["kernel_size"] = spec["pool_kernel_size"]
            if spec.get("pool_stride") is not None:
                pool["stride"] = spec["pool_stride"]
            if "pool_padding" in spec:
                pool["padding"] = spec["pool_padding"]
            blocks[-1]["pool"] = pool
        elif lt in ("flatten", "l2norm"):
            pending_pre.append(lt)
        else:
            raise ValueError(f"unknown layer_type {lt!r}")
    if pending_pre:
        raise ValueError(
            f"trailing stateless op(s) {pending_pre} with no following weighted layer"
        )
    return {"blocks": blocks, "final_reduce": reduce_for_next}


def block_model_from_layer_dicts(
    layer_dicts: list[dict[str, Any]], *, input_shape: tuple[int, ...], seed: int = 0
) -> SpectralModel:
    """Build a reduce-first ``SpectralModel`` from a flat layer-dict list.

    The ergonomic CNN/mixed authoring path: re-bracket the dicts into the block
    schema (:func:`layer_specs_to_blocks`) and build.  FFN-only callers can use
    :func:`neural_lofi.models.block_config.ffn_blocks` instead.
    """
    return SpectralModel.from_block_config(
        parse_block_config(layer_specs_to_blocks(layer_dicts)),
        input_shape=tuple(input_shape),
        seed=seed,
    )
