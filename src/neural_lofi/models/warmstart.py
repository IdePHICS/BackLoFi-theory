"""Warm-start a :class:`BackpropModel` from a fitted reduce-first ``SpectralModel``.

A reduce-first block is ``a_ℓ = pool(σ(W_ℓ · (V_{ℓ-1} a_{ℓ-1}) / c_ℓ) / √P_ℓ)`` —
the reduction ``V_{ℓ-1}`` and the random expansion ``W_ℓ`` are **adjacent linear
factors** (no nonlinearity between them), so they fold into a single weight

    U_ℓ = W_ℓ · (V_{ℓ-1} / s_{ℓ-1}) / (√P_{ℓ-1} · c_ℓ)        (U_0 = W_0 V_{-1} / c_0)

and the whole NLoFi network collapses **exactly** into a plain trainable net
``a_ℓ = σ(U_ℓ a_{ℓ-1})`` (the ``1/√P`` folded forward; the readout absorbs the
final reduction ``V_{L-1}`` and ``1/√P_{L-1}``).  This is the lofi→GD warm-start:
``backprop_model(x) == spectral_model(x)`` at init, then GD fine-tunes ``U``.

The reduce-first structure makes the fold **native and uniform for FFN and CNN**:
``V_{ℓ-1}`` is block ℓ's own ``reduce`` (read directly, no off-by-one), and because
the reduce sits at the *start* of a block — after the previous block's pool — no
reduction ever lies between a conv's σ and its pool, so the max-pool fold is exact.

Controls (label-aware signal isolation, not factoring):

* ``randomize_reduction`` — replace every ``V`` with a random orthonormal matrix
  of the same shape (same rank/width, label-agnostic).
* ``randomize_readout`` — re-draw the classifier instead of folding the final
  reduction (the "no-last" condition).

Restrictions: ``transform="dense"`` (the random ``W`` must be materialized to
fold; SORF keeps it implicit) and no ``l2norm`` in a block's ``pre`` (nonlinear,
cannot fold).  The fold carries the whitening **scale** but not a mean, so it is
**bit-exact for the mean-free whitening modes** (``whiten=False``, or
``whiten_mode="std"``/``"rms"`` which set μ=0) and only a scale-only approximation
for ``whiten_mode="centered"`` (whose per-direction mean has no folded bias).
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from .backprop import BackpropModel
from .spectral import SpectralModel, _Block, _Reduce


def _orthonormal_like(v: Tensor, generator: torch.Generator) -> Tensor:
    """Random orthonormal matrix with the same shape as ``v`` (columns ``QᵀQ=I``)."""
    g = torch.randn(*v.shape, generator=generator, dtype=v.dtype)
    q, _ = torch.linalg.qr(g, mode="reduced")
    return q.to(v)


def _reduce_matrix(
    red: _Reduce, *, randomize: bool, generator: torch.Generator
) -> Tensor | None:
    """The reduction as a 2-D matrix (whitening scale folded in); ``None`` = identity.

    ``M`` maps the incoming width to the kept width: conv reduce →
    ``(keep, in_channels)``; fc reduce → ``(in, keep)``.  Identity reductions
    (``reduce.k=null``) return ``None``.  The centered whitening *mean* is
    intentionally not folded (no per-layer bias to carry it).
    """
    if red.is_identity:
        return None
    v = red.V.detach()
    if red.is_conv:
        m = v[:, :, 0, 0]  # (keep, in_channels)
        if randomize:
            m = _orthonormal_like(m.T, generator).T
        elif red.whiten:
            m = m / red.scale.detach()[:, None]
        return m
    # fully-connected reduce: (in, keep)
    if randomize:
        v = _orthonormal_like(v, generator)
    elif red.whiten:
        v = v / red.scale.detach()[None, :]
    return v


def backprop_from_spectral(
    model: SpectralModel,
    *,
    output_dim: int | None = None,
    randomize_reduction: bool = False,
    randomize_readout: bool = False,
    seed: int = 0,
) -> BackpropModel:
    """Fold a fitted reduce-first ``SpectralModel`` into a trainable ``BackpropModel``.

    Per block ``U_ℓ = W_ℓ (V_{ℓ-1}/s) / (√P_{ℓ-1} c_ℓ)`` (matmul for fc,
    ``einsum`` for conv, spatial ``einsum`` across the conv→fc flatten); the
    classifier folds the final reduction and ``1/√P_{L-1}``.  Bit-equal to the
    lofi forward at init (``whiten=False``).
    """
    if getattr(model, "block_config", None) is None:
        raise ValueError(
            "backprop_from_spectral requires a reduce-first (block) SpectralModel "
            "built via SpectralModel.from_block_config."
        )
    gen = torch.Generator()
    gen.manual_seed(seed)

    if output_dim is None:
        ro = model.readout_weight.detach()
        output_dim = int(ro.shape[0]) if ro.ndim == 2 else 1
    bp = BackpropModel.from_block_config(
        model.block_config,
        input_shape=tuple(model.input_shape),
        output_dim=output_dim,
        classifier_bias=True,
    )
    # The folded twin's trainable layers are the nn.ModuleDicts (conv/linear +
    # activation), in block order; stateless pre/pool entries are plain modules.
    bp_weighted = [
        cast(nn.ModuleDict, layer)
        for layer in bp.layers
        if isinstance(layer, nn.ModuleDict)
    ]

    with torch.no_grad():
        prev_p: int | None = None
        for ell, (bcfg, bpl) in enumerate(
            zip(model.block_config.blocks, bp_weighted, strict=True)
        ):
            block = cast(_Block, model.blocks[ell])
            w = _dense_weight(block, ell)
            c = float(block.c.detach().reshape(-1)[0])
            scale = c if prev_p is None else math.sqrt(prev_p) * c
            m = _reduce_matrix(
                block.reduce, randomize=randomize_reduction, generator=gen
            )
            if bcfg.expand.is_conv:
                u = w if m is None else torch.einsum("qkab,ki->qiab", w, m)
                bpl["conv"].weight.data.copy_((u / scale).to(bpl["conv"].weight))
            else:
                if "l2norm" in bcfg.pre:
                    raise ValueError("cannot fold across an l2norm pre op")
                if m is None:
                    u = w
                elif "flatten" in bcfg.pre and block.reduce.is_conv:
                    # conv→fc: fold the channel reduce spatially into the fc weight.
                    keep, in_ch = m.shape
                    hw = w.shape[0] // keep
                    w_r = w.reshape(keep, hw, w.shape[1])
                    u = torch.einsum("ki,ksj->isj", m, w_r).reshape(
                        in_ch * hw, w.shape[1]
                    )
                else:
                    u = m @ w
                bpl["linear"].weight.data.copy_((u / scale).T.to(bpl["linear"].weight))
            prev_p = int(bcfg.expand.p)

        _fold_readout(
            model, bp, prev_p, randomize_readout=randomize_readout, generator=gen
        )
    return bp


def _dense_weight(block: torch.nn.Module, ell: int) -> Tensor:
    w = getattr(block, "weight", None)
    if w is None or w.numel() == 0:
        raise ValueError(
            f"block {ell} has no dense random weight (transform != 'dense'); "
            "the fold needs a materialized W. Re-fit with transform='dense'."
        )
    return w.detach()


def _fold_readout(
    model: SpectralModel,
    bp: BackpropModel,
    prev_p: int | None,
    *,
    randomize_readout: bool,
    generator: torch.Generator,
) -> None:
    """Fold the final reduction + ``1/√P_{L-1}`` into the classifier (or re-draw it)."""
    clf = bp.classifier
    if randomize_readout:
        bound = 1.0 / math.sqrt(clf.weight.shape[1])
        clf.weight.data.copy_(
            torch.empty_like(clf.weight.cpu())
            .uniform_(-bound, bound, generator=generator)
            .to(clf.weight)
        )
        if clf.bias is not None:
            clf.bias.data.copy_(
                torch.empty_like(clf.bias.cpu())
                .uniform_(-bound, bound, generator=generator)
                .to(clf.bias)
            )
        return

    ro = model.readout_weight.detach()
    inv_sqrt_p = 1.0 / math.sqrt(prev_p) if prev_p else 1.0
    m = _reduce_matrix(model.final_reduction, randomize=False, generator=generator)
    if m is None:
        wc = ro.reshape(1, -1) if ro.ndim == 1 else ro
        wc = wc * inv_sqrt_p
    elif model.final_reduction.is_conv:
        raise ValueError(
            "folding a conv final reduction (no flatten before the readout) is not "
            "supported; flatten before the final reduction."
        )
    else:  # fc final reduction: m is (P_last, k_L)
        wc = (ro @ m.T if ro.ndim == 2 else (m @ ro).unsqueeze(0)) * inv_sqrt_p
    clf.weight.data.copy_(wc.to(clf.weight))
    if clf.bias is not None:
        b = model.readout_bias.detach().reshape(-1)
        if b.numel() == 1 and clf.bias.numel() > 1:
            b = b.expand(clf.bias.numel())
        clf.bias.data.copy_(b.to(clf.bias))


def random_backprop_like(
    model: SpectralModel, *, output_dim: int, classifier_bias: bool = True
) -> BackpropModel:
    """A random-init ``BackpropModel`` with the same folded architecture.

    The ``plain-gd`` baseline / GD twin: same layer widths and structure as
    :func:`backprop_from_spectral` would produce, but freshly random-initialised
    (no warm-start).  Seed via ``set_seed`` before calling for reproducibility.
    """
    return BackpropModel.from_block_config(
        model.block_config,
        input_shape=tuple(model.input_shape),
        output_dim=output_dim,
        classifier_bias=classifier_bias,
    )
