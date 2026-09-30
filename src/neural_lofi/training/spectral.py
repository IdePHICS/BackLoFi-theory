"""Unified spectral trainer (reduce-first blocks, streaming).

Owns the NLoFi block fit for both feedforward and convolutional
``SpectralModel`` instances (built via ``from_block_config``).
:meth:`SpectralTrainer.fit` delegates to :meth:`SpectralTrainer._fit_blocked`,
which walks the blocks front-to-back; per block ℓ it (1) fits the input
normalization ``c_ℓ`` on the block's reduced+pre expand-input, then (2) fits the
*next* reduction ``V_ℓ`` on block ℓ's wide post-σ output ``a_ℓ`` (block ℓ+1's
reduce, or the final reduction).  Nothing of size ``N`` is stored: each pass
re-derives its features by forwarding the raw data through the already-fitted
prefix (``model.forward_to_block``) — the model is the only store.

The reduction fit accumulates the signed covariance of ``a_ℓ`` in a float64
``(P, P)`` / ``(c, P, P)`` accumulator (GPU/CPU per free memory) and solves the
top-``k`` projection; whitening (centered z-score) is fused in via single-pass
Welford moments.  Scalar targets use the closed-form signed-covariance
eigensolver; vector targets use the alternating ``(a, v)`` transverse solver
(``linear_svd`` required).  The readout streams the covariance-form sufficient
statistics of the final-reduced features and is solved by GCV
(:func:`fit_ridge_readout_from_covariance`).
"""

import logging
import math
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..models.spectral import SpectralModel, _Block, _Reduce
from .base import (
    BaseTrainer,
    BaseTrainerConfig,
    fit_ridge_readout_from_covariance,
)
from .eigen import (
    _is_hard_one_hot,
    alternating_vector_eigen_transverse_from_covariance,
    covariance_chunk_signed_accumulate_,
    covariance_chunk_signed_vector_accumulate_,
    flatten_spatial,
    flatten_spatial_vector,
    linear_mean_prepend_block_from_covariance,
    welford_cov_accumulate_,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# Module-level constants
# ---------------------------------------------------------------------- #

# Pure ``0/0`` guard for the centered z-score whitening std: a genuinely-null kept
# direction (variance exactly 0) would otherwise divide by 0.  It never binds in
# practice — centering makes ``var_j = v_jᵀ Cov v_j`` a PSD quadratic form, so the
# bound on ``ĥ = (f − μ)/σ`` is structural (mean-0 / unit-variance), not a clamp;
# there is no relative floor.
WHITEN_EPS = 1e-12

# Shares the model-side ``NLOFI_FINITE_CHECK`` env flag: when set, each whitened
# layer logs its per-direction std distribution (min/median/max) and the largest
# ``|μ_j/σ_j|`` — the mean-over-std factor the legacy std path used to amplify and
# that centering now removes (should stay O(1)).
_WHITEN_DEBUG = os.environ.get("NLOFI_FINITE_CHECK", "") not in ("", "0", "false")

__all__ = [
    "SpectralTrainer",
    "SpectralTrainerConfig",
]


def _cols_to_conv_kernel(V_cols: Tensor, device: torch.device | None = None) -> Tensor:
    """``(P, K)`` eigenvector columns → ``(K, P, 1, 1)`` 1×1 conv weight."""
    v = V_cols.T.unsqueeze(-1).unsqueeze(-1)
    return v.to(device) if device is not None else v


@dataclass
class SpectralTrainerConfig(BaseTrainerConfig):
    """Configuration for :class:`SpectralTrainer`.

    All fields are valid in both modes; CNN-only or FFN-only fields
    are simply ignored when fitting the other mode.

    Parameters
    ----------
    alpha_min, alpha_max, alpha_num
        Log-spaced grid of ridge regularisation candidates for the GCV
        covariance-form readout.
    covariance_chunk_size
        Legacy CNN knob (kept for config back-compat); ignored by the
        streaming worker, whose accumulation granularity is the loader batch.
    inner_max_iter, inner_tol, inner_patience
        Vector-label only (used when the targets are ``(N, c)``): controls
        for the alternating ``(a, v)`` Rayleigh maximization inner loop —
        max iterations, convergence tolerance on ``|λ^(t) − λ^(t-1)|``, and
        the number of consecutive below-tolerance steps required to stop.
        Ignored for scalar targets, which use the closed-form eigensolver.
    """

    alpha_min: float = -6.0
    alpha_max: float = 6.0
    alpha_num: int = 50
    covariance_chunk_size: int = 0
    inner_max_iter: int = 20
    inner_tol: float = 1e-4
    inner_patience: int = 3
    # dtype of the layer covariance accumulators (cov / c_stack / u / m / M2).
    # "float64" (default) is the precise, golden-matching path; "float32" halves
    # the accumulator memory and speeds the accumulation + eigensolve (notably on
    # Ada, FP64 1:64) at the cost of precision in the small eigenvalues — the top
    # eigenvectors are typically unaffected.  The GEMMs feeding the accumulator
    # are float32 either way; this only changes the buffer + the eigensolve dtype.
    accum_dtype: str = "float64"
    # Fraction of free CUDA memory to keep in reserve when deciding whether the
    # float64 (P,P)/(c,P,P) covariance accumulator lives on GPU (else CPU).
    # Only consulted in CUDA runs.
    accum_mem_margin: float = 0.2
    # Allow TF32 tensor cores for float32 matmuls for the duration of the fit
    # (set + restored around :meth:`_fit_blocked`).  The per-class Gram GEMMs
    # ``zᵀz`` are compute-bound and dominate the covariance build at large P;
    # TF32 runs them ~2.2x faster on Ada (measured 32→73 TFLOPS at P=8192) at
    # ~2e-4 relative error on the covariance entries (vs ~5e-7 plain fp32) —
    # the top-k eigenvectors are typically unaffected, but this is opt-in so
    # the default path stays bit-identical.  Convs already follow the separate
    # ``torch.backends.cudnn.allow_tf32`` flag (True by default in torch).
    tf32_matmul: bool = False


class SpectralTrainer(BaseTrainer):
    """Train any :class:`SpectralModel` via layer-wise eigenreduction + ridge.

    Both architecture- and label-agnostic at the API level.  All the
    surrounding scaffolding (dataset collection, label-free layers, the
    forward recipe ``z = σ(h @ W / c) / √P``, the conv-chunk loop, the
    ridge readout) is shared; a single internal axis varies:

    * **Labels** — from the rank of the targets ``y`` in :meth:`fit`.
      Scalar targets ``(N,)`` use the closed-form signed-covariance
      eigensolver; vector targets ``(N, c)`` use the alternating
      ``(a, v)`` Rayleigh maximization of
      :func:`neural_lofi.training.eigen.alternating_vector_eigen`.  The
      *only* thing that differs between the two is how each block's
      reduction ``V_ℓ`` is computed.

    Architecture (feedforward vs convolutional) is **not** a dispatch axis:
    the block config determines which kinds appear, and :meth:`_fit_blocked`
    drives them all through the same block loop.

    Returns the same results-dict shape in every case
    (``{"layers": [...], "final": {...}}``).
    """

    def __init__(
        self,
        model: SpectralModel,
        config: SpectralTrainerConfig,
    ) -> None:
        super().__init__(model, config)
        if not isinstance(model, SpectralModel):  # type: ignore[unreachable]
            raise TypeError(
                f"SpectralTrainer requires a SpectralModel, got {type(model).__name__}"
            )
        # Per-fit label metadata, peeked once from the loader at the start of a
        # fit (see :meth:`_peek_label_shape`).  ``_is_vector`` selects the
        # multi-class solver/readout path; ``_n_outputs`` is the readout width
        # (``c`` for vector, ``1`` for scalar).
        self._is_vector: bool = False
        self._n_outputs: int = 1
        # Whether the (vector) targets are HARD one-hot — peeked once with the
        # label shape.  Lets the accumulator skip its per-chunk one-hot sync and
        # the whiten moments be *derived* from the class Grams (no extra M2 GEMM).
        self._is_one_hot: bool = False

    # ------------------------------------------------------------------ #
    # Public entry point — dispatches on the label rank only
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Fit all blocks + the ridge readout; return ``(model, results)``.

        Streaming reduce-first fit: delegates to :meth:`_fit_blocked`, which
        walks the blocks front-to-back, re-deriving each block's features by
        forwarding the raw data through the already-fitted prefix
        (``model.forward_to_block``) — nothing of size ``N`` is stored.
        Dispatches on a single axis — the rank of the targets (scalar ``(N,)``
        vs vector ``(N, c)``) — for the per-reduction solver and readout shape.
        Architecture (FFN vs CNN) is *not* a dispatch axis.

        Parameters
        ----------
        loader : DataLoader
            Training data.  ``X`` is ``(N, D)`` for FFN-mode or
            ``(N, C, H, W)`` for CNN-mode (matches ``model.input_shape``);
            ``y`` is ``(N,)`` for scalar labels or ``(N, c)`` for vector
            (multi-class) labels.
        test_loader : DataLoader | None
            Optional test loader passed through ``**kwargs`` for
            evaluation alongside training.

        Returns
        -------
        tuple[nn.Module, dict[str, Any]]
            The trained model plus a results dict with two keys —
            ``"layers"`` (per-reduction info) and ``"final"`` (ridge
            output: ``train_mse``, ``ridge_alpha``, ``accuracy``, and
            optional ``test_*``).
        """
        test_loader = cast("DataLoader | None", kwargs.get("test_loader"))
        return self._fit_blocked(loader, test_loader=test_loader)

    @staticmethod
    @torch.no_grad()
    def _peek_label_shape(loader: DataLoader) -> tuple[bool, int, bool]:
        """Peek the first batch labels → ``(is_vector, n_outputs, is_one_hot)``.

        Re-iterable (no consume).  ``is_vector`` is True only for genuine
        multi-class targets (``y.ndim == 2`` *and* ``c > 1``) — a ``(N, 1)``
        target is treated as scalar.  ``n_outputs`` is the readout width
        (``c`` for vector, ``1`` for scalar).  ``is_one_hot`` is True only for a
        vector target whose first batch is hard one-hot (the encoding is constant
        across a fit, so one batch decides it).  Empty loader → ``(False, 1, False)``.
        """
        for _x, y in loader:
            if y.ndim == 2 and y.shape[1] > 1:
                return True, int(y.shape[1]), _is_hard_one_hot(y.to(torch.float32))
            return False, 1, False
        return False, 1, False

    # ================================================================== #
    # Shared accumulator / whitening / readout / metric helpers
    # ================================================================== #

    def _accumulator_device(self, p: int, n_class: int = 1) -> torch.device:
        """Device for the float64 ``(n_class, P, P)`` covariance accumulator.

        GPU when it fits free CUDA memory (``mem_get_info`` minus
        ``config.accum_mem_margin``), else CPU.  CPU on non-CUDA runs.
        """
        run_device = torch.device(self.config.device)
        if run_device.type != "cuda":
            return torch.device("cpu")
        itemsize = 4 if self._accum_dtype() == torch.float32 else 8
        nbytes = n_class * p * p * itemsize
        free, _total = torch.cuda.mem_get_info(run_device)
        margin = float(getattr(self.config, "accum_mem_margin", 0.2))
        return run_device if nbytes < free * (1.0 - margin) else torch.device("cpu")

    def _accum_dtype(self) -> torch.dtype:
        """Resolve the configured covariance-accumulator dtype (default float64)."""
        return (
            torch.float32
            if getattr(self.config, "accum_dtype", "float64") == "float32"
            else torch.float64
        )

    @staticmethod
    def _whiten_stats_from_covariance(
        v_cols: Tensor, s_cov: Tensor, mu_z: Tensor, count: int
    ) -> tuple[Tensor, Tensor]:
        """Per-direction ``(mean, std)`` from the single-pass Welford moments.

        ``v_cols`` is the ``(P, K)`` projection, ``s_cov`` the **unnormalised**
        centered co-moment ``S = Σ (z−μ)(z−μ)ᵀ`` and ``mu_z`` the running mean
        ``μ_z``, both over ``count`` patches (accumulated by
        :func:`welford_cov_accumulate_`).  For projected feature ``f_j = z·v_j``:

            mean_j = μ_z·v_j
            var_j  = v_jᵀ (S / (count−1)) v_j      # PSD quadratic form, no subtraction
            std_j  = sqrt(var_j).clamp_min(WHITEN_EPS)

        ``var_j`` is a sum of PSD quadratic forms with **no subtraction**, so the
        float32 ``s_cov`` is sufficient (the arithmetic is promoted to float64 only
        to reduce summation rounding) — unlike forming the std from raw moments
        ``Σf² − (Σf)²/m``, which is catastrophic-cancellation-prone at the ~5e7
        patch counts of wide conv layers (the removed std path).
        ``WHITEN_EPS`` is a pure ``0/0`` guard
        for a genuinely-null direction (never binds in practice); there is **no**
        relative floor — the bound on ``ĥ = (f − mean)/std`` is structural
        (centering makes every direction zero-mean / unit-variance), not a clamp.
        Returns ``(mean_proj, std)`` as ``(K,)`` tensors.
        """
        vc = v_cols.to(dtype=torch.float64, device=s_cov.device)  # (P, K)
        cov = s_cov.to(torch.float64) / max(count - 1, 1)
        var = ((cov @ vc) * vc).sum(dim=0).clamp_min(0.0)  # diag(Vᵀ Cov V) -> (K,)
        std = var.sqrt().clamp_min(WHITEN_EPS)
        mean_proj = mu_z.to(torch.float64) @ vc  # (K,)
        return mean_proj, std

    @torch.no_grad()
    def _readout_from_feature_batches(
        self,
        feature_batches: Iterator[tuple[Tensor, Tensor]],
        loader: DataLoader,
        test_loader: DataLoader | None,
        *,
        on_batch: Callable[[Tensor, Tensor], None] | None = None,
    ) -> dict[str, Any]:
        """Accumulate readout moments from ``(features_2d, y)`` batches → GCV solve.

        Used by the reduce-first readout pass (:meth:`_fit_blocked_readout`).
        float32 GEMM → float64 accumulate (the FP64 GEMM would be ~8x slower on
        Ada for no benefit; the ridge solve is unchanged).  ``on_batch`` is an
        optional per-batch hook.
        """
        model: SpectralModel = self.model  # type: ignore[assignment]
        config: SpectralTrainerConfig = self.config  # type: ignore[assignment]
        dev = config.device
        readout_dim = int(model.readout_input_dim)
        c_out = self._n_outputs
        gram = torch.zeros(readout_dim, readout_dim, dtype=torch.float64, device=dev)
        xty = torch.zeros(readout_dim, c_out, dtype=torch.float64, device=dev)
        sum_x = torch.zeros(readout_dim, dtype=torch.float64, device=dev)
        sum_y = torch.zeros(c_out, dtype=torch.float64, device=dev)
        yty = torch.zeros(c_out, c_out, dtype=torch.float64, device=dev)
        n = 0
        for feats, y in feature_batches:
            if on_batch is not None:
                on_batch(feats, y)
            yb = y if y.ndim == 2 else y.unsqueeze(1)
            gram += (feats.transpose(0, 1) @ feats).double()
            xty += (feats.transpose(0, 1) @ yb).double()
            sum_x += feats.sum(dim=0).double()
            sum_y += yb.sum(dim=0).double()
            yty += (yb.transpose(0, 1) @ yb).double()
            n += feats.shape[0]
        final = fit_ridge_readout_from_covariance(
            model,
            gram=gram,
            xty=xty,
            sum_x=sum_x,
            sum_y=sum_y,
            yty=yty,
            n_samples=n,
            alpha_min=config.alpha_min,
            alpha_max=config.alpha_max,
            alpha_num=config.alpha_num,
            verbose=config.verbose,
        )
        final["accuracy"] = self._stream_accuracy(loader)
        if test_loader is not None:
            final.update(self._stream_test_metrics(test_loader))
        return final

    @torch.no_grad()
    def _stream_accuracy(self, loader: DataLoader) -> float:
        """Train accuracy via a streaming pass through the full model + readout."""
        model: SpectralModel = self.model  # type: ignore[assignment]
        dev = self.config.device
        correct = 0
        n = 0
        for x, y in loader:
            pred = model(x.to(dev, non_blocking=True))
            yt = y.to(dev, non_blocking=True).to(torch.float32)
            yv = yt if yt.ndim == 2 else yt.unsqueeze(1)
            pv = pred if pred.ndim == 2 else pred.unsqueeze(1)
            if self._is_vector:
                correct += int((pv.argmax(dim=1) == yv.argmax(dim=1)).sum().item())
            else:
                labels = (pv[:, 0] >= 0).to(pv.dtype) * 2 - 1
                correct += int((yv[:, 0] == labels).sum().item())
            n += yv.shape[0]
        return correct / n

    @torch.no_grad()
    def _stream_test_metrics(self, loader: DataLoader) -> dict[str, Any]:
        """Test metrics (mse / accuracy / SEMs) via streaming the full model.

        Mirrors the metric formulas of :func:`fit_linear_readout` so the
        ``final`` dict carries the same ``test_*`` keys.
        """
        model: SpectralModel = self.model  # type: ignore[assignment]
        dev = self.config.device
        sq_total = 0.0
        persq_sum = 0.0
        persq_sq = 0.0
        correct = 0
        n = 0
        m_cols = 1
        for x, y in loader:
            pred = model(x.to(dev, non_blocking=True))
            yt = y.to(dev, non_blocking=True).to(torch.float32)
            yv = yt if yt.ndim == 2 else yt.unsqueeze(1)
            pv = pred if pred.ndim == 2 else pred.unsqueeze(1)
            m_cols = yv.shape[1]
            sq = (yv - pv) ** 2
            sq_total += float(sq.sum().item())
            n += yv.shape[0]
            if self._is_vector:
                correct += int((pv.argmax(dim=1) == yv.argmax(dim=1)).sum().item())
            else:
                labels = (pv[:, 0] >= 0).to(pv.dtype) * 2 - 1
                correct += int((yv[:, 0] == labels).sum().item())
                persq = sq[:, 0]
                persq_sum += float(persq.sum().item())
                persq_sq += float((persq * persq).sum().item())
        test_acc = correct / n
        out: dict[str, Any] = {
            "test_mse": sq_total / (n * m_cols),
            "test_accuracy": test_acc,
            "test_accuracy_sem": float(np.sqrt(test_acc * (1.0 - test_acc) / n)),
        }
        if not self._is_vector:
            mean_sq = persq_sum / n
            var_sq = max(persq_sq / n - mean_sq * mean_sq, 0.0)
            out["test_sem"] = float((var_sq**0.5) / (n**0.5))
        return out

    # ================================================================== #
    # Reduce-first block fit.  Each reduction V_{ℓ} is fit on block ℓ's WIDE
    # output a_ℓ; the same eigen / whiten / c-norm / readout machinery, with the
    # block loop as the orchestration.
    # ================================================================== #

    @torch.no_grad()
    def _fit_blocked(
        self,
        loader: DataLoader,
        *,
        test_loader: DataLoader | None = None,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Streaming reduce-first fit over a ``from_block_config`` model.

        Optionally fits ``V_{-1}`` (block 0's reduce) on the **raw input** first,
        then walks blocks front-to-back; per block ℓ: (1) fit ``c_ℓ`` on block ℓ's
        expand input, then (2) fit the *next* reduction (block ℓ+1's reduce, or
        the final reduction) on block ℓ's wide output ``a_ℓ``.  ``V_{-1}`` is a
        degree-1, label-aware input bottleneck; ``reduce.k=null`` (identity, the
        historical default) skips it.
        """
        model: SpectralModel = self.model  # type: ignore[assignment]
        config: SpectralTrainerConfig = self.config  # type: ignore[assignment]
        cfg = model.block_config
        dev = torch.device(config.device)
        self._is_vector, self._n_outputs, self._is_one_hot = self._peek_label_shape(
            loader
        )
        n_blocks = len(model.blocks)
        layer_history: list[dict[str, Any]] = []

        # TF32 (opt-in): enable tensor-core float32 matmuls for the duration of
        # the fit, restoring the global flag afterwards.  Affects only the cuBLAS
        # GEMMs (gram / cross moments, dense FC lifts, readout moments); convs
        # follow the separate ``torch.backends.cudnn.allow_tf32``.
        tf32_prev: bool | None = None
        if config.tf32_matmul:
            tf32_prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            # V_{-1}: a real label-aware input reduction on the RAW data (block 0's
            # reduce), fit before the loop so block 0's c-norm + expand see the
            # reduced input.  ``forward_to_block(x, stop=0)`` is the raw input, so
            # the ``ell=-1`` reduction-fit reuses the same covariance/solve path.
            if n_blocks and cfg.blocks[0].reduce.k is not None:
                layer_history.append(
                    self._fit_reduction_on_block(
                        -1,
                        cfg.blocks[0].reduce,
                        cast(_Reduce, model.blocks[0].reduce),
                        loader,
                    )
                )
            for ell in range(n_blocks):
                block = cast(_Block, model.blocks[ell])
                bcfg = cfg.blocks[ell]
                # 1) c-norm on block ℓ's expand input (reduced + pre features).
                if bcfg.expand.normalize:
                    sumsq = 0.0
                    count = 0
                    for x, _y in loader:
                        a = model.forward_to_block(
                            x.to(dev, non_blocking=True), stop=ell
                        )
                        r = block.expand_input(a)
                        sumsq += float(r.double().pow(2).sum().item())
                        count += r.shape[0]
                    block.c.data.fill_(math.sqrt(sumsq / count))
                # 2) fit the next reduction on a_ℓ (block ℓ's wide output).
                if ell + 1 < n_blocks:
                    rcfg = cfg.blocks[ell + 1].reduce
                    rmod = cast(_Reduce, model.blocks[ell + 1].reduce)
                else:
                    rcfg, rmod = cfg.final_reduce, model.final_reduction
                layer_history.append(
                    self._fit_reduction_on_block(ell, rcfg, rmod, loader)
                )
            final = self._fit_blocked_readout(loader, test_loader)
        finally:
            if tf32_prev is not None:
                torch.backends.cuda.matmul.allow_tf32 = tf32_prev
        return model, {"layers": layer_history, "final": final}

    @torch.no_grad()
    def _fit_reduction_on_block(
        self, ell: int, rcfg: Any, rmod: _Reduce, loader: DataLoader
    ) -> dict[str, Any]:
        """Fit one reduction ``V`` on block ``ell``'s wide output ``a_ℓ``.

        Streams ``a_ℓ = forward_to_block(x, stop=ell+1)``, accumulates the
        signed covariance (scalar or per-class vector) + fused whitening Welford
        moments, solves the top-``k`` projection, and stores it into ``rmod``
        (with whitening).  Identity reductions (``rcfg.k is None``) are a no-op.
        """
        model: SpectralModel = self.model  # type: ignore[assignment]
        config: SpectralTrainerConfig = self.config  # type: ignore[assignment]
        dev = torch.device(config.device)

        probe: Tensor | None = None
        for x, _y in loader:
            probe = model.forward_to_block(x.to(dev, non_blocking=True), stop=ell + 1)
            break
        if probe is None:
            raise RuntimeError(f"Block {ell}: empty loader, cannot fit.")
        is_conv = probe.ndim == 4
        p = probe.shape[1]
        out_shape = tuple(probe.shape[1:])
        if rcfg.k is None:
            return {
                "block": ell,
                "reduce": "identity",
                "p": p,
                "train_output_shape": out_shape,
            }
        if self._is_vector and not rcfg.linear_svd:
            raise ValueError(f"Block {ell}: vector reduction requires linear_svd=True.")

        accum_dev = self._accumulator_device(p, self._n_outputs)
        accum_dt = self._accum_dtype()
        s_cov = mu_z = None
        if rcfg.whiten:
            s_cov = torch.zeros(p, p, dtype=torch.float32, device=accum_dev)
            mu_z = torch.zeros(p, dtype=torch.float32, device=accum_dev)
        flatten = flatten_spatial_vector if self._is_vector else flatten_spatial

        def _feats(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
            a = model.forward_to_block(x.to(dev, non_blocking=True), stop=ell + 1)
            yf_ = y.to(dev, non_blocking=True).to(torch.float32)
            if is_conv:
                zf, yf, _ = flatten(a, yf_)
                return zf, yf
            return a, yf_

        lin_svals: Tensor | None = None
        linear_k = 0
        if self._is_vector:
            c_out = self._n_outputs
            c_stack = torch.zeros(c_out, p, p, dtype=accum_dt, device=accum_dev)
            m_acc = torch.zeros(p, c_out, dtype=accum_dt, device=accum_dev)
            total = 0
            for x, y in loader:
                zf, yf = _feats(x, y)
                if s_cov is not None and mu_z is not None:
                    welford_cov_accumulate_(s_cov, mu_z, total, zf)
                total = covariance_chunk_signed_vector_accumulate_(
                    c_stack, m_acc, zf, yf, total, assume_one_hot=self._is_one_hot
                )
            if total == 0:
                raise RuntimeError(f"Block {ell}: no data accumulated.")
            linear_k = rcfg.linear_k or c_out
            _a_star, _u, lin_svals, eigvals, v_cols, _traj = (
                alternating_vector_eigen_transverse_from_covariance(
                    c_stack,
                    m_acc,
                    rcfg.k,
                    linear_k=linear_k,
                    max_iter=config.inner_max_iter,
                    tol=config.inner_tol,
                    patience=config.inner_patience,
                )
            )
            v_cols = v_cols.to(torch.float32)
        else:
            cov = torch.zeros(p, p, dtype=accum_dt, device=accum_dev)
            u = torch.zeros(p, dtype=accum_dt, device=accum_dev)
            total = 0
            for x, y in loader:
                zf, yf = _feats(x, y)
                if s_cov is not None and mu_z is not None:
                    welford_cov_accumulate_(s_cov, mu_z, total, zf)
                total = covariance_chunk_signed_accumulate_(cov, u, zf, yf, total)
            if total == 0:
                raise RuntimeError(f"Block {ell}: no data accumulated.")
            cov = 0.5 * (cov + cov.transpose(0, 1))
            eigvals, v_cols = linear_mean_prepend_block_from_covariance(
                cov,
                u,
                k=rcfg.k,
                include_linear_mean=rcfg.include_linear_mean,
                orthogonalize=rcfg.orthogonalize_to_mean,
                target_dtype=torch.float32,
                target_device=dev,
                log_prefix=f"Block {ell} reduce: ",
            )

        self._store_reduction(rmod, rcfg, v_cols, s_cov, mu_z, total)
        entry = {
            "block": ell,
            "p": p,
            "k": rcfg.k,
            "eigenvalues": eigvals.cpu().numpy().tolist(),
            "train_output_shape": out_shape,
        }
        if lin_svals is not None:
            entry["linear_svd"] = True
            entry["linear_k"] = linear_k
            entry["linear_singular_values"] = lin_svals.cpu().numpy().tolist()
        return entry

    def _store_reduction(
        self,
        rmod: _Reduce,
        rcfg: Any,
        v_cols: Tensor,
        s_cov: Tensor | None,
        mu_z: Tensor | None,
        total: int,
    ) -> None:
        """Store fitted projection columns (+ whitening stats) into a reduction.

        Shared by the plain reduction fit and the backward-corrected fit
        (:class:`~neural_lofi.training.backward_lofi.BackwardSpectralTrainer`):
        moves ``v_cols`` into ``rmod.V`` (1×1 conv kernel for conv reductions)
        and, when ``rcfg.whiten``, derives the per-direction whitening stats
        from the fused Welford moments according to ``rcfg.whiten_mode``.
        """
        dev = torch.device(self.config.device)
        if rmod.is_conv:
            rmod.V = nn.Parameter(
                _cols_to_conv_kernel(v_cols, dev), requires_grad=False
            )
        else:
            rmod.V = nn.Parameter(v_cols.to(dev), requires_grad=False)
        if rcfg.whiten:
            assert s_cov is not None and mu_z is not None
            mean_proj, std = self._whiten_stats_from_covariance(
                v_cols, s_cov, mu_z, total
            )
            # whiten_mode picks how each kept direction j is normalised:
            #   std      : ĥ = f/σ            keeps the (label-linear DC) mean; folds
            #              faithfully (μ=0). DC dirs (σ≈0) CAN blow up at large width
            #              — the pre-refactor behaviour, and the default.
            #   rms      : ĥ = f/√E[f²]        keeps the mean; DC dirs scale by |μ| so
            #              are bounded to O(1) → conditioned AND faithful (no blow-up).
            #   centered : ĥ = (f−μ)/σ         subtracts the label-linear DC mean
            #              (removes signal); folds only approximately (no μ).
            mode = rcfg.whiten_mode
            if mode == "centered":
                mean_w, scale_w = mean_proj, std
            elif mode == "rms":
                mean_w = torch.zeros_like(mean_proj)
                scale_w = (
                    (std * std + mean_proj * mean_proj).sqrt().clamp_min(WHITEN_EPS)
                )
            else:  # "std" (default)
                mean_w = torch.zeros_like(mean_proj)
                scale_w = std
            rmod.mean = nn.Parameter(
                mean_w.to(dtype=torch.float32, device=dev), requires_grad=False
            )
            rmod.scale = nn.Parameter(
                scale_w.to(dtype=torch.float32, device=dev), requires_grad=False
            )

    @torch.no_grad()
    def _fit_blocked_readout(
        self, loader: DataLoader, test_loader: DataLoader | None
    ) -> dict[str, Any]:
        """Reduce-first readout: stream the final-reduced features, GCV-solve."""
        model: SpectralModel = self.model  # type: ignore[assignment]
        dev = self.config.device

        def feature_batches() -> Iterator[tuple[Tensor, Tensor]]:
            for x, y in loader:
                feats = model._final_features(
                    model.forward_to_block(x.to(dev, non_blocking=True))
                )
                yield feats, y.to(dev, non_blocking=True).to(torch.float32)

        return self._readout_from_feature_batches(
            feature_batches(), loader, test_loader
        )
