"""Scalar signed-covariance eigendecomposition.

The signed covariance ``C = (1/N) Z^T diag(y) Z`` is eigendecomposed by one of
three paths, dispatched on ``P`` and ``k``:

- **Dense + eigh**: materialise the full ``(P, P)`` covariance and run
  ``torch.linalg.eigh`` (on CUDA) or ``numpy.linalg.eigh`` (CPU).  Preferred
  when *k* is large relative to *P* — ``eigh`` is O(P³) regardless of *k*.
- **Dense + eigsh**: materialise ``(P, P)`` then ``scipy.sparse.linalg.eigsh``,
  when *P* fits in memory and *k* is small enough that Lanczos beats ``eigh``.
- **LinearOperator + eigsh**: implicit matrix-vector products, so the ``(P, P)``
  matrix is never formed — O(N*P) per Lanczos iteration.  Used only when *P* is
  very large.

:func:`top_k_eig_from_covariance` is the single eigensolver core; the
covariance-accumulation paths in :mod:`..prepend` and :mod:`..vector` delegate
to it directly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import torch
from scipy.sparse.linalg import ArpackError, LinearOperator, eigsh
from torch import Tensor

from ...utils.covariance import signed_covariance_matrix, signed_covariance_matvec
from .deflation import _deflate_symmetric_any, deflated_matvec

log = logging.getLogger(__name__)

# Public tunable constants — renamed in v0.2's naming cleanup (Phase B).

# Default threshold: use LinearOperator when P exceeds this value.
# At P=25000 the (P, P) covariance is ~2.5 GB float32 — affordable on most
# machines.  The linop path only wins when P is so large that materializing
# the covariance is infeasible (e.g. P > 50000 → >10 GB).
LINOP_THRESHOLD_P = 50000

# When k exceeds P // EIGH_K_RATIO we use full eigh instead of eigsh.
# eigsh (Lanczos) time grows roughly as O(P² · k · restarts); eigh is a
# fixed O(P³).  In practice the crossover happens around k ≈ P/4.
EIGH_K_RATIO = 4


def signed_covariance_eigen(
    z: Tensor,
    y: Tensor,
    k: int,
    *,
    linop_threshold: int = LINOP_THRESHOLD_P,
    deflate_against: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Eigendecompose the signed covariance ``C = (1/N) Z^T diag(y) Z``.

    Dispatches to the dense or LinearOperator path based on *P* and *k*.

    Parameters
    ----------
    z : Tensor
        Feature matrix ``(N, P)``.
    y : Tensor
        Targets ``(N,)`` or ``(N, 1)``.
    k : int
        Number of top eigenvectors to keep.
    linop_threshold : int
        Use the LinearOperator path when ``P > linop_threshold`` and
        ``k <= P // 2``.
    deflate_against : Tensor | None
        Optional unit-norm direction ``v0 ∈ R^P``.  When provided, the
        returned eigenpairs are those of the deflated covariance
        ``C̃ = P_perp C P_perp`` with ``P_perp = I − v0 v0^T``.  The zero
        eigenvalue along ``v0`` is suppressed by construction and not
        selected as long as ``k ≤ P − 1``.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(eigenvalues, eigenvectors)`` with shapes ``(k,)`` and ``(P, k)``,
        sorted by descending absolute eigenvalue, on the same device/dtype
        as *z*.
    """
    P = z.shape[1]
    if deflate_against is not None:
        if deflate_against.ndim not in (1, 2) or deflate_against.shape[0] != P:
            raise ValueError(
                f"deflate_against must have shape ({P},) or ({P}, r), "
                f"got {tuple(deflate_against.shape)}"
            )
        deflate_against = deflate_against.to(device=z.device, dtype=z.dtype)
    if P <= linop_threshold or k > P // 2:
        return _signed_covariance_eigen_dense(z, y, k, deflate_against=deflate_against)
    return _signed_covariance_eigen_linop(z, y, k, deflate_against=deflate_against)


def top_k_eig_from_covariance(c: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Top-*k* eigenpairs (by ``|λ|``) of a dense symmetric ``(P, P)`` covariance.

    The single eigensolver core.  Both the feature-matrix path
    (:func:`_signed_covariance_eigen_dense`, which builds ``c`` from ``(z, y)``)
    and the covariance-accumulation paths (the CNN chunked-conv fit and the
    vector-label conv fit, which accumulate ``c`` directly and never
    materialise the ``(N·H·W, P)`` patch-feature matrix) delegate here, so the
    dispatch lives in exactly one place.

    Dispatch (keyed on where ``c`` already lives, so callers control the
    backend by the device they hand in):

    * ``c`` on CUDA with ``P ≤ 25000`` → full ``torch.linalg.eigh`` on the
      device.  Full decomposition is O(P³) but the BLAS-3 GPU path far
      outperforms sequential Lanczos (~1-2 s at P=20K vs 30-60 s for CPU
      eigsh).  On failure it falls back to the CPU path below.
    * CPU → full ``numpy.linalg.eigh`` when ``k ≥ P // EIGH_K_RATIO`` (eigh is
      a fixed O(P³) and beats Lanczos once ``k`` is a large fraction of ``P``),
      otherwise ``scipy`` ``eigsh`` (``which="LM"``) with the deterministic
      start vector ``v0 = 1/√P`` so degenerate eigenspaces resolve identically
      across callers.  ``eigsh`` non-convergence falls back to full ``eigh``.

    Returns ``(eigvals, eigvecs)`` on ``c``'s device/dtype, sorted by
    descending ``|λ|``.
    """
    P = c.shape[0]

    # GPU fast path: run eigh directly on the device.
    if c.is_cuda and P <= 25000:  # pragma: no cover - CUDA only
        try:
            eigvals_t, eigvecs_t = torch.linalg.eigh(c)
        except RuntimeError as exc:
            log.warning("GPU eigh at P=%d failed (%s); falling back to CPU.", P, exc)
        else:
            order = torch.argsort(eigvals_t.abs(), descending=True)[:k]
            return eigvals_t[order].contiguous(), eigvecs_t[:, order].contiguous()

    # CPU path: full eigh when k is large relative to P, else Lanczos eigsh.
    c_np = c.detach().cpu().numpy()
    if k >= P // EIGH_K_RATIO:
        log.info("Using eigh (k=%d, P=%d)", k, P)
        eigvals_np, eigvecs_np = np.linalg.eigh(c_np)
        abs_order = np.argsort(np.abs(eigvals_np))[::-1][:k]
        eigvals_np = eigvals_np[abs_order]
        eigvecs_np = eigvecs_np[:, abs_order]
    else:
        v0 = np.ones(P, dtype=c_np.dtype) / np.sqrt(P)
        try:
            eigvals_np, eigvecs_np = eigsh(c_np, k=k, which="LM", v0=v0)
        except ArpackError as exc:
            # ArpackError (base class) covers both non-convergence AND the -9999
            # "could not build an Arnoldi factorization" failure that a
            # near-degenerate covariance triggers; both fall back to dense eigh,
            # which is a direct solver and never fails on a finite symmetric matrix.
            log.warning(
                "eigsh failed (k=%d, P=%d): %s; using full eigh.",
                k,
                P,
                exc,
            )
            eigvals_np, eigvecs_np = np.linalg.eigh(c_np)
            abs_order = np.argsort(np.abs(eigvals_np))[::-1][:k]
            eigvals_np = eigvals_np[abs_order]
            eigvecs_np = eigvecs_np[:, abs_order]

    return _to_sorted_tensors(eigvals_np, eigvecs_np, c.device, c.dtype)


def _signed_covariance_eigen_dense(
    z: Tensor, y: Tensor, k: int, *, deflate_against: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Dense path: form the full ``(P, P)`` covariance, then delegate.

    Builds ``C`` on the GPU when ``z`` is on CUDA and ``P ≤ 25000`` (so
    :func:`top_k_eig_from_covariance` takes its on-device ``eigh`` branch),
    otherwise on CPU — with a float32 cast for very large ``P`` to bound the
    covariance footprint.  Eigenpairs are returned on ``z``'s device/dtype.
    """
    n = z.shape[0]
    P = z.shape[1]

    if z.is_cuda and P <= 25000:  # pragma: no cover - CUDA only
        c = signed_covariance_matrix(z, y)  # symmetrised by default, on device
    elif P > 20000:  # pragma: no cover - triggers only when P > 20000
        c = signed_covariance_matrix(z.cpu().float(), y.view(n).cpu().float())
    else:
        c = signed_covariance_matrix(z, y).cpu()

    if deflate_against is not None:
        c = _deflate_symmetric_any(
            c, deflate_against.detach().to(device=c.device, dtype=c.dtype)
        )

    eigvals, eigvecs = top_k_eig_from_covariance(c, k)
    return (
        eigvals.to(device=z.device, dtype=z.dtype),
        eigvecs.to(device=z.device, dtype=z.dtype),
    )


def signed_covariance_full_eigenvalues(z: Tensor, y: Tensor) -> Tensor:
    """Compute all eigenvalues of the signed covariance for dense features.

    Parameters
    ----------
    z : Tensor
        Feature matrix ``(N, P)``.
    y : Tensor
        Targets ``(N,)`` or ``(N, 1)``.

    Returns
    -------
    Tensor
        Eigenvalues of the signed covariance, sorted in ascending order
        (as returned by ``numpy.linalg.eigvalsh``).
    """
    n = z.shape[0]
    P = z.shape[1]
    if P > 20000:  # pragma: no cover - triggers only when P > 20000
        z_cpu = z.cpu().float()
        y_cpu = y.view(n).cpu().float()
        c_np = signed_covariance_matrix(z_cpu, y_cpu).numpy()
        eigvals_np = np.linalg.eigvalsh(c_np)
        return torch.from_numpy(eigvals_np.copy()).to(dtype=torch.float32)

    c = signed_covariance_matrix(z, y)
    eigvals_np = np.linalg.eigvalsh(c.cpu().numpy())
    return torch.from_numpy(eigvals_np.copy()).to(dtype=z.dtype)


def signed_covariance_full_eigenvalues_from_covariance(c: Tensor) -> Tensor:
    """Compute all eigenvalues from a dense signed-covariance matrix."""
    eigvals_np = np.linalg.eigvalsh(c.cpu().numpy())
    return torch.from_numpy(eigvals_np.copy()).to(dtype=c.dtype)


def _signed_covariance_eigen_linop(
    z: Tensor, y: Tensor, k: int, *, deflate_against: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """LinearOperator path: implicit matvec, never forms the (P, P) matrix."""
    P = z.shape[1]

    # Move to CPU — Lanczos matvecs are sequential, GPU kernel launch
    # overhead dominates for single-vector operations.
    z_cpu = z.cpu().float()
    y_cpu = y.view(-1).cpu().float()

    def matvec(v: np.ndarray) -> np.ndarray:
        """numpy-in/out wrapper around the torch signed-covariance matvec."""
        v_t = torch.from_numpy(v).float()
        return signed_covariance_matvec(z_cpu, y_cpu, v_t).numpy()

    op_matvec: Callable[[np.ndarray], np.ndarray] = matvec
    if deflate_against is not None:
        v0_np = deflate_against.detach().cpu().float().numpy()
        op_matvec = deflated_matvec(matvec, v0_np)

    op = LinearOperator((P, P), matvec=op_matvec, dtype=np.float32)
    v0_lanczos = np.ones(P, dtype=np.float32) / np.sqrt(P)
    eigvals_np, eigvecs_np = eigsh(op, k=k, which="LM", v0=v0_lanczos)

    return _to_sorted_tensors(eigvals_np, eigvecs_np, z.device, z.dtype)


def _to_sorted_tensors(
    eigvals_np: np.ndarray,
    eigvecs_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Convert numpy eigenpairs to sorted tensors on *device*/*dtype*."""
    eigvals = torch.from_numpy(eigvals_np.copy()).to(dtype=dtype, device=device)
    eigvecs = torch.from_numpy(eigvecs_np.copy()).to(dtype=dtype, device=device)

    order = torch.argsort(eigvals.abs(), descending=True)
    return eigvals[order], eigvecs[:, order]
