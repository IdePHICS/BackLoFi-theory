"""Diagnostic covariance spectra.

Full eigenvalue spectra (and the top eigenvectors) of the unsigned covariance
``(1/N) Z^T Z`` and the label-signed covariance ``(1/N) Z^T diag(y) Z`` of a
feature matrix.  These are *diagnostic* helpers — they serialise to plain
Python lists for JSON storage and plotting, and are off the training hot path.
They are not on the training hot path (the streaming trainer no longer records
per-layer spectra); scripts call them directly on raw features (e.g. for an
input-layer spectrum).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch import Tensor

from ...utils.covariance import covariance_matrix, signed_covariance_matrix

# Above this P, the dense ``(P, P)`` eigendecomposition is replaced by a top-k
# scipy ``eigsh`` to avoid GPU OOM / the O(P³) cost of a full ``eigh``; the
# full spectrum is then omitted (left empty) in the returned dict.
SPARSE_SPECTRA_P = 20000


@torch.no_grad()
def covariance_spectra(z: Tensor, y: Tensor, n_top: int) -> dict[str, Any]:
    """Full spectra + top-``n_top`` eigenvectors of the unsigned and signed
    covariances of features ``z``.

    Returns a dict with two prefixes — ``cov`` (unsigned) and ``signed_cov`` —
    and three keys per prefix: ``{prefix}_spectrum`` (full eigenvalues, empty
    for the large-``P`` sparse path), ``{prefix}_top_eigenvalues`` and
    ``{prefix}_top_eigenvectors`` (top-``n_top`` by ``|λ|``).  All values are
    plain Python lists for JSON storage.
    """
    P = z.shape[1]
    use_sparse = P > SPARSE_SPECTRA_P

    if use_sparse:  # pragma: no cover - triggers only when P > 20000
        # Move to CPU for large P to avoid GPU OOM.
        z_cpu = z.cpu().float()
        y_cpu = y.view(z.shape[0]).cpu().float()
        cov = covariance_matrix(z_cpu)
        signed_cov = signed_covariance_matrix(z_cpu, y_cpu)
    else:
        cov = covariance_matrix(z)
        signed_cov = signed_covariance_matrix(z, y)

    result: dict[str, Any] = {}
    for prefix, mat in [("cov", cov), ("signed_cov", signed_cov)]:
        if use_sparse:  # pragma: no cover - triggers only when P > 20000
            # For large P, use scipy sparse eigsh (top-K only); skip the full
            # spectrum.
            mat_np = mat.cpu().float().numpy()
            k_use = min(n_top, P - 1)
            eigvals_np, eigvecs_np = eigsh(mat_np, k=k_use, which="LM")
            order_np = np.argsort(np.abs(eigvals_np))[::-1]
            result[f"{prefix}_spectrum"] = []
            result[f"{prefix}_top_eigenvalues"] = eigvals_np[order_np].tolist()
            result[f"{prefix}_top_eigenvectors"] = eigvecs_np[:, order_np].tolist()
        else:
            try:
                eigvals, eigvecs = torch.linalg.eigh(mat)
            except torch.cuda.OutOfMemoryError:  # pragma: no cover - CUDA only
                eigvals, eigvecs = torch.linalg.eigh(mat.cpu())
                eigvals = eigvals.to(mat.device)
                eigvecs = eigvecs.to(mat.device)
            result[f"{prefix}_spectrum"] = eigvals.cpu().numpy().tolist()
            order_t = torch.argsort(eigvals.abs(), descending=True)[:n_top]
            result[f"{prefix}_top_eigenvalues"] = (
                eigvals[order_t].cpu().numpy().tolist()
            )
            result[f"{prefix}_top_eigenvectors"] = (
                eigvecs[:, order_t].cpu().numpy().tolist()
            )
    return result


@torch.no_grad()
def covariance_spectra_from_covariances(
    cov: Tensor, signed_cov: Tensor, n_top: int
) -> dict[str, Any]:
    """Top-``n_top`` eigenpairs from pre-accumulated ``(P, P)`` covariances.

    The covariance-form counterpart of :func:`covariance_spectra`, used when
    the unsigned and signed covariances were accumulated batch-by-batch (the FC
    batched path) rather than from a materialised feature matrix.  The full
    ``{prefix}_spectrum`` is left empty here; only the top-``n_top`` eigenpairs
    (by ``|λ|``) are returned, under the same key layout.
    """
    result: dict[str, Any] = {}
    P = cov.shape[0]
    for prefix, mat in [("cov", cov), ("signed_cov", signed_cov)]:
        mat_np = mat.cpu().float().numpy()
        k_use = min(n_top, P - 1)
        eigvals_np, eigvecs_np = eigsh(mat_np, k=k_use, which="LM")
        order = np.argsort(np.abs(eigvals_np))[::-1]
        result[f"{prefix}_spectrum"] = []
        result[f"{prefix}_top_eigenvalues"] = eigvals_np[order].tolist()
        result[f"{prefix}_top_eigenvectors"] = eigvecs_np[:, order].tolist()
    return result
