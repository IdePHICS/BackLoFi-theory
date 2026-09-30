"""GPU linalg with a CPU fallback on cuSOLVER non-convergence.

cuSOLVER's iterative eigensolver (``syevd``) and SVD (``gesvdj``) raise
``LinAlgError`` on ill-conditioned matrices with (near-)repeated spectra — common
with structured/SORF features, whose symmetries produce exactly degenerate
singular/eigen values.  LAPACK on CPU is robust to those, so retry the same
matrix there.  The covariance eigensolver in :mod:`.dense` and the diagnostic
spectra in :mod:`.spectra` already do this inline; these helpers centralise the
pattern for the vector-label SVDs and the readout solve.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

log = logging.getLogger(__name__)


def safe_svd(
    m: Tensor, *, full_matrices: bool = False
) -> tuple[Tensor, Tensor, Tensor]:
    """``torch.linalg.svd`` with a CPU fallback on cuSOLVER non-convergence."""
    try:
        return torch.linalg.svd(m, full_matrices=full_matrices)
    except torch.linalg.LinAlgError as exc:
        if not m.is_cuda:
            raise
        log.warning("GPU svd %s failed (%s); retrying on CPU.", tuple(m.shape), exc)
        u, s, vh = torch.linalg.svd(m.cpu(), full_matrices=full_matrices)
        return u.to(m.device), s.to(m.device), vh.to(m.device)


def safe_eigh(m: Tensor) -> tuple[Tensor, Tensor]:
    """``torch.linalg.eigh`` with a CPU fallback on cuSOLVER non-convergence."""
    try:
        return torch.linalg.eigh(m)
    except torch.linalg.LinAlgError as exc:
        if not m.is_cuda:
            raise
        log.warning("GPU eigh %s failed (%s); retrying on CPU.", tuple(m.shape), exc)
        w, v = torch.linalg.eigh(m.cpu())
        return w.to(m.device), v.to(m.device)
