"""Base trainer interface and shared configuration."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..utils.helpers import resolve_device
from .eigen.robust import safe_eigh, safe_svd

log = logging.getLogger(__name__)


@dataclass
class BaseTrainerConfig:
    """Common trainer settings inherited by every concrete trainer config."""

    device: str = "cpu"
    verbose: bool = True


def _ridge_loocv(
    X: Tensor,
    y: Tensor,
    alphas: Tensor,
    *,
    promote_fp64: bool = True,
    min_resid_frac: float = 0.05,
) -> tuple[Tensor, Tensor, float]:
    """Closed-form leave-one-out ridge CV, on *X*'s device (no host copy).

    Reproduces ``sklearn`` ``RidgeCV`` (its default efficient-LOO path with
    ``fit_intercept=True``) in pure torch so it runs on whatever device *X*
    lives on — CPU or GPU — without ever moving to host memory or numpy.

    Centers ``X``/``y``, takes one economy SVD ``X_c = U S Vᵀ`` (with
    ``d = S²``), and evaluates the closed-form leave-one-out residual
    ``(yᵢ − ŷᵢ) / (1 − Hᵢᵢ)`` for every ``α`` *without* refitting, where the
    smoother diagonal is ``Hᵢᵢ(α) = Σⱼ Uᵢⱼ² dⱼ/(dⱼ+α)``.  A single ``α`` —
    shared across all output columns, matching ``RidgeCV``'s default — is
    chosen to minimise the mean LOO MSE; the coefficients are then refit in
    closed form ``ŵ = V (S g)/(d+α)`` with ``g = Uᵀ y_c``.

    Working precision is ``float32``/``float64`` (half is upcast to
    ``float32``).  When ``promote_fp64`` is set (default) and the design is not
    comfortably tall — ``N ≤ P`` — the solve is promoted to ``float64``: in that
    near-rank-deficient regime the leverage ``Hᵢᵢ`` approaches ``1`` and the
    ``1 − Hᵢᵢ`` denominator loses all significance in ``float32`` (catastrophic
    cancellation), which selects a spurious ``α``.  ``float64`` reproduces
    ``sklearn``'s ``α`` exactly and still runs on the original device (CUDA
    ``float64`` is ~6× slower but only pays in this uncommon regime).  The tall
    ``P ≪ N`` case that the eigenreduced readout always hits stays ``float32``.

    Returns ``(coef, intercept, alpha)`` with ``coef`` of shape ``(P, m)`` and
    ``intercept`` of shape ``(m,)`` (``m`` = number of output columns; ``1``
    for a scalar target), on *X*'s original device/dtype.
    """
    out_dtype = X.dtype
    work_dtype = (
        out_dtype if out_dtype in (torch.float32, torch.float64) else torch.float32
    )
    # Ill-conditioned regime guard: promote to float64 when the centered design
    # is (near-)rank-deficient (N ≤ P), where float32 LOO is numerically unsafe.
    if promote_fp64 and X.shape[0] <= X.shape[1]:
        work_dtype = torch.float64
    x = X.to(work_dtype)
    yc = y.to(device=x.device, dtype=work_dtype)
    if yc.ndim == 1:
        yc = yc.unsqueeze(1)  # (N, 1)

    m = yc.shape[1]

    x_mean = x.mean(dim=0, keepdim=True)  # (1, P)
    y_mean = yc.mean(dim=0, keepdim=True)  # (1, m)
    x_centered = x - x_mean
    y_centered = yc - y_mean

    # Economy SVD: U (N, r), S (r,), Vh (r, P), r = min(N, P).
    u, s, vh = safe_svd(x_centered)
    d = s * s  # (r,) eigenvalues of XᵀX
    g = u.transpose(0, 1) @ y_centered  # (r, m)

    a = alphas.to(device=x.device, dtype=work_dtype).reshape(-1)  # (A,)
    # filt[j, l] = d_j / (d_j + a_l)  → the ridge "shrinkage" per (component, α).
    filt = d.unsqueeze(1) / (d.unsqueeze(1) + a.unsqueeze(0))  # (r, A)
    # Leave-one-out denominator 1 − Hᵢᵢ(α) = 1 − Σⱼ Uᵢⱼ² dⱼ/(dⱼ+α).
    one_minus_h = (1.0 - (u * u) @ filt).clamp_min(1e-12)  # (N, A)

    # Accumulate Σᵢ Σ_col LOO_residual² per α (loop over output columns to keep
    # the working set at a few (N, A) tensors; Hᵢᵢ is shared across columns).
    loo_sq = torch.zeros(a.shape[0], dtype=work_dtype, device=x.device)
    for j in range(m):
        y_hat = u @ (g[:, j].unsqueeze(1) * filt)  # (N, A)
        resid = y_centered[:, j].unsqueeze(1) - y_hat  # (N, A)
        loo = resid / one_minus_h
        loo_sq += (loo * loo).sum(dim=0)  # (A,)

    # Near-interpolation guard: in the N <= P regime the LOO can have a SPURIOUS
    # minimum at the smallest alpha, where the ridge interpolates (Hᵢᵢ -> 1,
    # 1 - Hᵢᵢ -> 0), selecting an overfit alpha (train_mse ~ 0, test blows up).
    # Restrict the argmin to alphas with at least ``min_resid_frac * N`` residual
    # effective-DOF (N - tr(H), tr(H) = Σⱼ dⱼ/(dⱼ+alpha)).  The tall P << N readout
    # never interpolates, so this is a no-op there.
    n_samp = x.shape[0]
    resid_dof = n_samp - filt.sum(0)  # (A,)
    ok = resid_dof >= max(1.0, min_resid_frac * n_samp)
    loo_sel = torch.where(ok, loo_sq, torch.full_like(loo_sq, float("inf")))
    if not bool(torch.isfinite(loo_sel).any()):
        loo_sel = loo_sq  # all alphas near-interpolate -> fall back to raw LOO
    best = int(torch.argmin(loo_sel).item())
    alpha_star = float(a[best].item())

    # Refit coefficients at the selected α: ŵ = V (S g)/(d+α).
    coef_factor = (s / (d + a[best])).unsqueeze(1) * g  # (r, m)
    coef = vh.transpose(0, 1) @ coef_factor  # (P, m)
    intercept = (y_mean - x_mean @ coef).reshape(-1)  # (m,)

    return coef.to(out_dtype), intercept.to(out_dtype), alpha_star


def fit_linear_readout(
    model: nn.Module,
    h_train: Tensor,
    y_train: Tensor,
    *,
    h_test: Tensor | None = None,
    y_test_np: np.ndarray | None = None,
    alpha_min: float = -5.0,
    alpha_max: float = 5.0,
    alpha_num: int = 50,
    promote_fp64: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit a leave-one-out ridge readout and store weights in *model*.

    Handles both **scalar** binary targets (``y_train`` shape ``(N,)``;
    accuracy by the ``ŷ ≥ 0 → ±1`` threshold, weight ``(P,)`` / bias ``(1,)``)
    and **vector** multi-class targets (``y_train`` shape ``(N, c)`` with
    ``c > 1``; accuracy by ``argmax``, weight ``(c, P)`` / bias ``(c,)``),
    dispatched on ``y_train.ndim``.

    The regularization strength is chosen by closed-form leave-one-out CV in
    pure torch on the features' device — no host copy, no sklearn — via
    :func:`_ridge_loocv` (a single shared ``α`` across outputs).  The selected
    strength is reported as ``ridge_alpha``.

    Parameters
    ----------
    model : nn.Module
        Model to attach ``readout_weight`` / ``readout_bias`` to.
    h_train : Tensor
        Training features ``(N_train, P)``.
    y_train : Tensor
        Training targets ``(N_train,)`` (scalar) or ``(N_train, c)`` (vector).
    h_test : Tensor | None
        Test features ``(N_test, P)``.
    y_test_np : ndarray | None
        Test targets as a NumPy array (``(N_test,)`` or ``(N_test, c)``).
    alpha_min, alpha_max, alpha_num : float
        Log10 bounds and count for regularization candidates.
    promote_fp64 : bool
        When set (default), promote the LOO solve to ``float64`` in the
        ill-conditioned ``N ≤ P`` regime (see :func:`_ridge_loocv`).  The
        eigenreduced readout is always tall (``P = k ≪ N``), so this is a
        guardrail for off-label feature shapes and a no-op in the common path.
    verbose : bool
        Whether to log ridge metrics.

    Returns
    -------
    dict[str, Any]
        Metrics dict with keys ``train_mse``, ``ridge_alpha``, ``accuracy``,
        and — when test data is given — ``test_mse``, ``test_accuracy``,
        ``test_accuracy_sem`` (plus ``test_sem`` for scalar targets).
    """
    is_vector = y_train.ndim == 2 and y_train.shape[1] > 1

    # Align targets with the features' device so the closed-form solve and the
    # downstream metric tensors never straddle host/accelerator memory (callers
    # occasionally hand us a host ``y`` alongside on-device features).
    y_train = y_train.to(h_train.device)

    alphas = torch.logspace(alpha_min, alpha_max, alpha_num)
    coef, intercept, ridge_alpha = _ridge_loocv(
        h_train, y_train, alphas, promote_fp64=promote_fp64
    )
    # coef: (P, m), intercept: (m,)

    if is_vector:
        model.readout_weight = nn.Parameter(
            coef.transpose(0, 1).contiguous(), requires_grad=False
        )  # (c, P)
        model.readout_bias = nn.Parameter(
            intercept.contiguous(), requires_grad=False
        )  # (c,)
    else:
        model.readout_weight = nn.Parameter(
            coef[:, 0].contiguous(), requires_grad=False
        )  # (P,)
        model.readout_bias = nn.Parameter(
            intercept.contiguous(), requires_grad=False
        )  # (1,)

    # Predictions / metrics computed on-device from the same coef/intercept.
    y_tr = y_train.to(coef.dtype)
    y_tr2 = y_tr.unsqueeze(1) if y_tr.ndim == 1 else y_tr  # (N, m)
    pred_tr = h_train.to(coef.dtype) @ coef + intercept  # (N, m)
    train_mse = float(((y_tr2 - pred_tr) ** 2).mean().item())

    if is_vector:
        train_acc = float(
            (pred_tr.argmax(dim=1) == y_tr2.argmax(dim=1)).float().mean().item()
        )
    else:
        pred_labels = (pred_tr[:, 0] >= 0).to(coef.dtype) * 2 - 1
        train_acc = float((y_tr2[:, 0] == pred_labels).float().mean().item())

    if verbose:
        log.info("Ridge: alpha=%.4e, train_mse=%.6f", ridge_alpha, train_mse)

    final: dict[str, Any] = {
        "train_mse": train_mse,
        "ridge_alpha": ridge_alpha,
        "accuracy": train_acc,
    }

    if h_test is not None and y_test_np is not None:
        y_te2 = torch.as_tensor(y_test_np, dtype=coef.dtype, device=coef.device)
        if y_te2.ndim == 1:
            y_te2 = y_te2.unsqueeze(1)  # (N_test, m)
        pred_te = h_test.to(coef.dtype) @ coef + intercept  # (N_test, m)
        sq = (y_te2 - pred_te) ** 2
        test_mse = float(sq.mean().item())
        n_test = y_te2.shape[0]

        if is_vector:
            test_acc = float(
                (pred_te.argmax(dim=1) == y_te2.argmax(dim=1)).float().mean().item()
            )
        else:
            pred_te_labels = (pred_te[:, 0] >= 0).to(coef.dtype) * 2 - 1
            test_acc = float((y_te2[:, 0] == pred_te_labels).float().mean().item())
            # SEM of the per-sample squared residual (population std, ddof=0).
            test_sem = float((sq[:, 0].std(unbiased=False) / (n_test**0.5)).item())
            final["test_sem"] = test_sem

        test_acc_sem = float(np.sqrt(test_acc * (1.0 - test_acc) / n_test))

        final["test_mse"] = test_mse
        final["test_accuracy"] = test_acc
        final["test_accuracy_sem"] = test_acc_sem

        if verbose:
            log.info(
                "  Test: mse=%.6f, accuracy=%.4f±%.4f",
                test_mse,
                test_acc,
                test_acc_sem,
            )

    return final


def fit_ridge_readout_from_covariance(
    model: nn.Module,
    *,
    gram: Tensor,
    xty: Tensor,
    sum_x: Tensor,
    sum_y: Tensor,
    yty: Tensor,
    n_samples: int,
    alpha_min: float = -5.0,
    alpha_max: float = 5.0,
    alpha_num: int = 50,
    verbose: bool = True,
    min_resid_frac: float = 0.05,
) -> dict[str, Any]:
    """Fit a ridge readout from covariance-form sufficient statistics (GCV).

    Streamable counterpart of :func:`fit_linear_readout`: instead of the
    per-sample feature matrix ``X`` it consumes only the moments a streaming
    pass can accumulate — the Gram ``G = XᵀX`` and cross-moment ``B = XᵀY``
    plus the bias moments — and selects the ridge strength by **generalized
    cross-validation** (GCV) rather than leave-one-out (LOO is *not*
    streamable; it needs per-sample ``X``).  The only ``(D, D)`` object is the
    Gram, never an ``N``-sized array.

    Sets ``readout_weight`` / ``readout_bias`` on *model* (scalar ``(D,)/(1,)``
    when ``c == 1``; vector ``(c, D)/(c,)`` when ``c > 1``) and returns the two
    metrics derivable from moments alone.  **Accuracy and test metrics are not
    computed here** — they need per-sample predictions (sign / argmax), so the
    caller streams them through the full model after this attaches the readout.

    Parameters
    ----------
    model : nn.Module
        Model to attach ``readout_weight`` / ``readout_bias`` to.
    gram : Tensor
        ``G = XᵀX``, shape ``(D, D)`` (``D = readout_input_dim``).
    xty : Tensor
        ``B = XᵀY``, shape ``(D, c)`` (``c = 1`` for scalar targets).
    sum_x, sum_y : Tensor
        ``Σx`` ``(D,)`` and ``Σy`` ``(c,)`` (for centering / the intercept).
    yty : Tensor
        ``YᵀY``, shape ``(c, c)`` (for the GCV residual sum of squares).
    n_samples : int
        Number of training samples ``N``.
    alpha_min, alpha_max, alpha_num : float
        Log10 bounds and count for the regularization grid.
    verbose : bool
        Whether to log the selected ``α`` and train MSE.

    Returns
    -------
    dict[str, Any]
        ``{"train_mse", "ridge_alpha"}`` (both computed from the moments).
    """
    # Work entirely in float64 on the moments' device; D ≤ P is moderate, so a
    # single dense eigendecomposition of the (D, D) Gram is cheap.
    dev = gram.device
    g = gram.to(torch.float64)
    b = xty.to(torch.float64)
    sx = sum_x.to(torch.float64).reshape(-1)
    sy = sum_y.to(torch.float64).reshape(-1)
    yy = yty.to(torch.float64)
    n = float(n_samples)
    c = b.shape[1]

    x_mean = sx / n  # (D,)
    y_mean = sy / n  # (c,)
    # Centered moments: subtract the rank-1 mean outer products.
    g_c = g - torch.outer(sx, sx) / n  # (D, D)
    b_c = b - torch.outer(sx, sy) / n  # (D, c)
    yty_c_trace = float(torch.diag(yy).sum().item() - (sy @ sy).item() / n)

    # Eigendecompose the centered Gram once: G_c = V diag(λ) Vᵀ.  cuSOLVER's syevd
    # raises LinAlgError 1254 on Grams with many (near-)repeated eigenvalues
    # (common with structured/SORF features); safe_eigh retries on CPU LAPACK.
    lam, v = safe_eigh(g_c)  # ascending λ (D,), V (D, D)
    lam = lam.clamp_min(0.0)
    b_tilde = v.transpose(0, 1) @ b_c  # (D, c)
    energy = (b_tilde * b_tilde).sum(dim=1)  # eᵢ = Σ_c B̃ᵢc²  (D,)

    alphas = torch.logspace(
        alpha_min, alpha_max, alpha_num, dtype=torch.float64, device=dev
    )  # (A,)
    lam_col = lam.unsqueeze(1)  # (D, 1)
    a_row = alphas.unsqueeze(0)  # (1, A)
    denom = lam_col + a_row  # (D, A)
    # RSS(α) = tr(YᵀY_c) − Σᵢ eᵢ (λᵢ + 2α)/(λᵢ+α)²  (→ tr − Σ eᵢ/λᵢ as α→0).
    rss = (
        yty_c_trace
        - (energy.unsqueeze(1) * (lam_col + 2.0 * a_row) / denom.pow(2)).sum(dim=0)
    ).clamp_min(0.0)  # (A,)
    df = (lam_col / denom).sum(dim=0)  # Σ λᵢ/(λᵢ+α) = tr(H) effective DOF  (A,)
    gcv = torch.full_like(rss, float("inf"))
    # Near-interpolation guard: in the N <= P (D) regime, at tiny alpha the ridge
    # interpolates so rss -> 0 while (n - df) -> ~0 but stays > 1e-9, making
    # gcv = n*rss/(n-df)^2 -> a SPURIOUS 0 that selects the alpha floor and
    # overfits (train_mse ~ 0, test blows up).  Require at least
    # ``min_resid_frac * n`` residual effective-DOF (n - tr(H)) for an alpha to be
    # eligible.  The tall D << N readout never interpolates -> no-op there.
    valid = (n - df) >= max(1.0, min_resid_frac * n)
    if not bool(valid.any()):
        valid = (n - df) > 1e-9  # all near-interpolate -> fall back to old rule
    gcv[valid] = n * rss[valid] / (n - df[valid]).pow(2)
    best = int(torch.argmin(gcv).item())
    alpha_star = float(alphas[best].item())

    # Refit at α*: w = V diag(1/(λ+α*)) B̃; bias absorbs the means.
    filt = (1.0 / (lam + alpha_star)).unsqueeze(1)  # (D, 1)
    w = v @ (filt * b_tilde)  # (D, c)
    bias = y_mean - x_mean @ w  # (c,)
    train_mse = float(rss[best].item() / (n * c))

    w32 = w.to(device=dev, dtype=torch.float32)
    bias32 = bias.to(device=dev, dtype=torch.float32)
    if c > 1:
        model.readout_weight = nn.Parameter(
            w32.transpose(0, 1).contiguous(), requires_grad=False
        )  # (c, D)
        model.readout_bias = nn.Parameter(bias32.contiguous(), requires_grad=False)
    else:
        model.readout_weight = nn.Parameter(
            w32[:, 0].contiguous(), requires_grad=False
        )  # (D,)
        model.readout_bias = nn.Parameter(
            bias32.contiguous(), requires_grad=False
        )  # (1,)

    if verbose:
        log.info(
            "Ridge (GCV/covariance): alpha=%.4e, train_mse=%.6f",
            alpha_star,
            train_mse,
        )

    return {"train_mse": train_mse, "ridge_alpha": alpha_star}


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    config : BaseTrainerConfig
        Training configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: BaseTrainerConfig,
    ) -> None:
        self.config = config
        self.config.device = resolve_device(config.device)
        self.model = model.to(self.config.device)

    @abstractmethod
    def fit(
        self, loader: DataLoader, **kwargs: object
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Train the model on *loader* and return it with results.

        Parameters
        ----------
        loader : DataLoader
            Training data loader.
        **kwargs
            Subclass-specific keyword arguments.

        Returns
        -------
        tuple[nn.Module, dict[str, Any]]
            The trained model and a results dict.
        """

    def _to_device(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        """Move an arbitrary number of tensors to ``self.config.device``."""
        return tuple(t.to(self.config.device) for t in tensors)

    @torch.no_grad()
    def _collect_dataset(
        self, loader: DataLoader, *, to_device: bool = True
    ) -> tuple[Tensor, Tensor]:
        """Drain an entire loader into two concatenated ``(X, Y)`` tensors.

        ``to_device=True`` (default) routes each batch through
        :meth:`_to_device`; ``to_device=False`` keeps the data where it is
        (used by the CNN-mode spectral fit, which moves chunks to the
        accelerator on the fly).
        """
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        for x, y in loader:
            if to_device:
                x, y = self._to_device(x, y)
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)
