"""LoFi-vs-GD representation extraction + the standardized comparison call.

Ports the FC LoFi subspace-recovery study's representation scheme to the
reduce-first models, generalized to **both** FC and conv blocks, and exposes one
entry point — :func:`compare_lofi_gd_features` — that runs any set of registry
metrics (:data:`~.feature_metrics.METRICS`) on any set of representation pairs.

Two representation pairs per block ``ℓ`` (``ref`` = spectral, ``cmp`` = GD):

* ``"svd_principal"`` — ref: the lofi **reduced** features ``V_{ℓ-1} a_{ℓ-1}``
  (block ℓ's ``expand_input``, the k label-aware directions); cmp: the GD
  layer's weight replaced by its top-k **right singular directions** applied to
  the GD forward's incoming features (+ the layer activation).  At a warm-start
  init the folded ``U_ℓ = W_ℓ V_{ℓ-1}`` has row space equal to the reduce
  subspace, so for **FC blocks** recovery = 1 by construction (exact with
  ``apply_activation=False``) — the built-in correctness gate.  For **conv
  blocks** the reference is the reduced map at the *same* location (center
  tap) while a k×k filter mixes neighboring locations, so fold-time recovery
  is < 1 by construction: read conv recovery as a trajectory relative to its
  own step-0 value.  Blocks with an identity reduce carry no lofi subspace and
  are skipped.
* ``"raw"`` — ref: the lofi wide post-σ block output ``a_ℓ``; cmp: the GD
  block output (after the block's pool, matching ``forward_features``).

Conv features are compared **locations-as-samples**: a seeded subset of spatial
locations (identical on both sides, all images) caps the sample-matrix at
``max_samples`` rows so the metric cost is architecture-independent.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..models.backprop import BackpropModel
from ..models.spectral import SpectralModel, _Block
from .feature_metrics import compute_metrics

# ponytail: exact SVD below this min-dim, randomized top-k above (conv folds
# reach (5120, 46080); exact SVD there is seconds-to-minutes per snapshot).
_EXACT_SVD_MAX_DIM = 2048


def canonicalize_svd_direction_signs(vh: Tensor) -> Tensor:
    """Make right-singular-vector signs deterministic (largest entry positive)."""
    if vh.numel() == 0:
        return vh
    max_abs_idx = vh.abs().argmax(dim=1)
    signs = torch.sign(vh[torch.arange(vh.shape[0], device=vh.device), max_abs_idx])
    return vh * torch.where(signs == 0, torch.ones_like(signs), signs).unsqueeze(1)


@torch.no_grad()
def top_right_singular_directions(weight: Tensor, *, rank: int) -> Tensor:
    """Top-``rank`` right singular vectors of a 2-D weight, rows orthonormal."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got {tuple(weight.shape)}")
    use_rank = min(int(rank), *weight.shape)
    if use_rank <= 0:
        raise ValueError(f"Cannot compute SVD directions with rank={rank}")
    w = weight.detach().float()
    if min(w.shape) <= _EXACT_SVD_MAX_DIM:
        vh = torch.linalg.svd(w.cpu(), full_matrices=False).Vh[:use_rank]
    else:
        q = min(use_rank + 64, *w.shape)
        _, _, v = torch.svd_lowrank(w, q=q, niter=6)
        vh = v.T[:use_rank].cpu()
    return canonicalize_svd_direction_signs(vh.contiguous())


# --------------------------------------------------------------------------- #
# Model walking
# --------------------------------------------------------------------------- #
def _gd_segments(
    model: BackpropModel,
) -> Iterator[tuple[int, list[nn.Module], nn.ModuleDict, nn.Module | None]]:
    """Yield ``(block_idx, pre_layers, trainable, pool)`` per block, in order."""
    layers = list(model.layers)
    pos = 0
    for i, bcfg in enumerate(model.block_config.blocks):
        pre = [layers[pos + j] for j in range(len(bcfg.pre))]
        pos += len(bcfg.pre)
        trainable = cast(nn.ModuleDict, layers[pos])
        pos += 1
        pool = None
        if bcfg.pool is not None:
            pool = layers[pos]
            pos += 1
        yield i, pre, trainable, pool


def _principal_directions(
    gd_model: BackpropModel, ranks: dict[int, int]
) -> dict[int, Tensor]:
    """Per-block top-k right singular directions of each trainable weight (2-D)."""
    out: dict[int, Tensor] = {}
    for i, _pre, trainable, _pool in _gd_segments(gd_model):
        if i not in ranks:
            continue
        key = "conv" if "conv" in trainable else "linear"
        w = cast(nn.Conv2d | nn.Linear, trainable[key]).weight
        out[i] = top_right_singular_directions(w.reshape(w.shape[0], -1), rank=ranks[i])
    return out


def _spectral_ranks(model: SpectralModel) -> dict[int, int]:
    """Per-block reduce rank k (identity reduces are absent — no lofi subspace)."""
    ranks: dict[int, int] = {}
    for i, block in enumerate(model.blocks):
        red = cast(_Block, block).reduce
        if not red.is_identity:
            ranks[i] = int(red.V.shape[0] if red.is_conv else red.V.shape[1])
    return ranks


# --------------------------------------------------------------------------- #
# Location subsampling (conv → locations-as-samples with a bounded row count)
# --------------------------------------------------------------------------- #
# ponytail: 4096 column cap for the one fat matrix (conv->fc reduced ref,
# k*H*W columns); a patch-unfolded conv reference is the upgrade path if the
# center-tap caveat ever binds.
_MAX_FEATURES = 4096


def _sample_matrix(
    feat: Tensor, *, n_images: int, max_samples: int, seed: int
) -> Tensor:
    """CPU float32 sample×feature matrix with a seeded, side-aligned row cap.

    4-D features keep a fixed seeded subset of spatial locations (the same for
    every image and — because the choice depends only on ``seed`` and ``H*W`` —
    for both sides of a pair), sized so ``n_images * n_locations ≈ max_samples``.
    2-D features pass through rows (the probe size caps them) but wide ones get
    a seeded **column** subset at ``_MAX_FEATURES`` — a span subsample that keeps
    the per-snapshot SVD affordable (only the conv→fc reduced ref is that wide).
    """
    if feat.ndim == 2:
        out = feat.detach().to(device="cpu", dtype=torch.float32)
        if out.shape[1] > _MAX_FEATURES:
            gen = torch.Generator().manual_seed(seed + out.shape[1])
            out = out[:, torch.randperm(out.shape[1], generator=gen)[:_MAX_FEATURES]]
        return out
    n, c, h, w = feat.shape
    flat = feat.detach().permute(0, 2, 3, 1).reshape(n, h * w, c)
    keep = min(h * w, max(1, max_samples // max(n_images, 1)))
    if keep < h * w:
        gen = torch.Generator().manual_seed(seed + h * w)
        locs = torch.randperm(h * w, generator=gen)[:keep].to(flat.device)
        flat = flat[:, locs]
    return flat.reshape(n * keep, c).to(device="cpu", dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Feature extraction sweeps (one forward pass each, chunked over the probe)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def spectral_probe_features(
    model: SpectralModel,
    x: Tensor,
    *,
    representations: Sequence[str] = ("svd_principal", "raw"),
    max_samples: int = 8192,
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, dict[int, Tensor]]:
    """The lofi reference features per block, as capped CPU sample matrices.

    ``"svd_principal"`` → the reduced features (block ℓ's ``expand_input`` on
    ``a_{ℓ-1}``), only for non-identity reduces; ``"raw"`` → the wide block
    output ``a_ℓ``.  Compute once per run — the lofi model never changes.
    """
    ranks = _spectral_ranks(model)
    n_images = int(x.shape[0])
    chunks: dict[str, dict[int, list[Tensor]]] = {r: {} for r in representations}
    for start in range(0, n_images, batch_size):
        a = x[start : start + batch_size]
        for i, block_mod in enumerate(model.blocks):
            block = cast(_Block, block_mod)
            if "svd_principal" in chunks and i in ranks:
                r = block.expand_input(a)
                chunks["svd_principal"].setdefault(i, []).append(
                    _sample_matrix(
                        r, n_images=n_images, max_samples=max_samples, seed=seed
                    )
                )
            a = block(a)
            if "raw" in chunks:
                chunks["raw"].setdefault(i, []).append(
                    _sample_matrix(
                        a, n_images=n_images, max_samples=max_samples, seed=seed
                    )
                )
    return {
        rep: {i: torch.cat(parts) for i, parts in per_block.items()}
        for rep, per_block in chunks.items()
    }


@torch.no_grad()
def backprop_probe_features(
    model: BackpropModel,
    x: Tensor,
    *,
    directions: dict[int, Tensor] | None,
    representations: Sequence[str] = ("svd_principal", "raw"),
    apply_activation: bool = True,
    max_samples: int = 8192,
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, dict[int, Tensor]]:
    """The GD-side features per block, aligned with :func:`spectral_probe_features`.

    ``"svd_principal"`` needs ``directions`` (from :func:`_principal_directions`
    via :func:`compare_lofi_gd_features`, or precomputed): the incoming features
    are projected on the layer's top-k right singular directions (+ activation
    when ``apply_activation``); ``"raw"`` is the block output after its pool.
    """
    n_images = int(x.shape[0])
    chunks: dict[str, dict[int, list[Tensor]]] = {r: {} for r in representations}
    for start in range(0, n_images, batch_size):
        h = x[start : start + batch_size]
        for i, pre, trainable, pool in _gd_segments(model):
            for op in pre:
                h = op(h)
            key = "conv" if "conv" in trainable else "linear"
            if "svd_principal" in chunks and directions and i in directions:
                dirs = directions[i].to(device=h.device, dtype=h.dtype)
                if key == "conv":
                    conv = cast(nn.Conv2d, trainable["conv"])
                    filters = dirs.reshape(-1, *conv.weight.shape[1:])
                    feat = F.conv2d(
                        h, filters, stride=conv.stride, padding=conv.padding
                    )
                else:
                    feat = F.linear(h, dirs)
                if apply_activation:
                    feat = trainable["activation"](feat)
                chunks["svd_principal"].setdefault(i, []).append(
                    _sample_matrix(
                        feat, n_images=n_images, max_samples=max_samples, seed=seed
                    )
                )
            h = trainable["activation"](trainable[key](h))
            if pool is not None:
                h = pool(h)
            if "raw" in chunks:
                chunks["raw"].setdefault(i, []).append(
                    _sample_matrix(
                        h, n_images=n_images, max_samples=max_samples, seed=seed
                    )
                )
    return {
        rep: {i: torch.cat(parts) for i, parts in per_block.items()}
        for rep, per_block in chunks.items()
    }


# --------------------------------------------------------------------------- #
# The standardized comparison call
# --------------------------------------------------------------------------- #
def _slim_for_dcor(
    feat: Tensor, *, max_channels: int, max_samples: int, seed: int
) -> Tensor:
    """dcor cost control: seeded channel subset + row cap (dcor is O(C²·N²))."""
    if max_channels > 0 and feat.shape[1] > max_channels:
        gen = torch.Generator().manual_seed(seed + feat.shape[1])
        feat = feat[:, torch.randperm(feat.shape[1], generator=gen)[:max_channels]]
    if max_samples > 0 and feat.shape[0] > max_samples:
        feat = feat[:max_samples]
    return feat


# Default metric set PER REPRESENTATION.  recovery on the wide `raw` spans is
# near-vacuous (two ~C-dim spans in an S-dim sample space, chance floor ~C/S)
# AND its (S, 5120)-scale SVDs dominate the probe cost — so raw gets dcor only.
DEFAULT_METRICS: Mapping[str, Sequence[str]] = {
    "svd_principal": ("recovery", "dcor"),
    "raw": ("dcor",),
}


@torch.no_grad()
def compare_lofi_gd_features(
    spectral_model: SpectralModel,
    gd_model: BackpropModel,
    x: Tensor,
    *,
    metrics: Sequence[str] | Mapping[str, Sequence[str]] = DEFAULT_METRICS,
    representations: Sequence[str] = ("svd_principal", "raw"),
    metric_params: dict[str, dict[str, object]] | None = None,
    apply_activation: bool = True,
    max_samples: int = 8192,
    dcor_max_channels: int = 64,
    dcor_max_samples: int = 1024,
    seed: int = 0,
    batch_size: int = 256,
    spectral_cache: dict[str, dict[int, Tensor]] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run the named registry metrics on lofi-vs-GD feature pairs, per block.

    Returns ``{"block_ℓ": {representation: {metric column: value}}}`` with the
    metric convention ``ref`` = spectral, ``cmp`` = GD.  ``metrics`` is either a
    flat list (applied to every representation) or a map representation→list
    (see :data:`DEFAULT_METRICS`).  dcor runs on CUDA when available unless
    ``metric_params`` overrides its ``device``.  Pass ``spectral_cache`` (a
    prior :func:`spectral_probe_features` result on the *same* probe ``x``) to
    skip the lofi forward — the reference never changes across GD snapshots.
    """
    if spectral_cache is None:
        spectral_cache = spectral_probe_features(
            spectral_model,
            x,
            representations=representations,
            max_samples=max_samples,
            seed=seed,
            batch_size=batch_size,
        )
    ranks = _spectral_ranks(spectral_model)
    directions = (
        _principal_directions(gd_model, ranks)
        if "svd_principal" in representations
        else None
    )
    gd_feats = backprop_probe_features(
        gd_model,
        x,
        directions=directions,
        representations=representations,
        apply_activation=apply_activation,
        max_samples=max_samples,
        seed=seed,
        batch_size=batch_size,
    )

    mp: dict[str, dict[str, object]] = {
        k: dict(v) for k, v in (metric_params or {}).items()
    }
    if torch.cuda.is_available():
        mp.setdefault("dcor", {}).setdefault("device", "cuda")

    out: dict[str, dict[str, dict[str, float]]] = {}
    for rep in representations:
        rep_metrics = metrics.get(rep, ()) if isinstance(metrics, Mapping) else metrics
        for i, ref in spectral_cache.get(rep, {}).items():
            cmp = gd_feats.get(rep, {}).get(i)
            if cmp is None:
                continue
            row: dict[str, float] = {}
            for name in rep_metrics:
                if name == "dcor":
                    pair = (
                        _slim_for_dcor(
                            ref,
                            max_channels=dcor_max_channels,
                            max_samples=dcor_max_samples,
                            seed=seed,
                        ),
                        _slim_for_dcor(
                            cmp,
                            max_channels=dcor_max_channels,
                            max_samples=dcor_max_samples,
                            seed=seed + 1,
                        ),
                    )
                else:
                    pair = (ref, cmp)
                row.update(compute_metrics(*pair, names=[name], params=mp))
            out.setdefault(f"block_{i}", {})[rep] = row
    return out
