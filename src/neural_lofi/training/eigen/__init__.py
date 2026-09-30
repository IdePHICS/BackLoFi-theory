"""Signed-covariance eigendecomposition for the ridge spectral trainers.

Split by concern across the submodules:

- :mod:`.deflation`  — subspace-projection algebra (rank-update deflation).
- :mod:`.dense`      — the scalar signed-covariance eigensolver (dense eigh /
  eigsh / matrix-free Lanczos) and ``top_k_eig_from_covariance``, the single
  eigensolver core; plus the ``LINOP_THRESHOLD_P`` / ``EIGH_K_RATIO`` knobs.
- :mod:`.prepend`    — linear-mean (degree-1) direction + prepend blocks.
- :mod:`.vector`     — vector-label alternating Rayleigh maximization.
- :mod:`.chunked`    — spatial reshaping + chunked covariance accumulation.
- :mod:`.spectra`    — diagnostic full covariance spectra (called directly).

This package replaces the former single ``training/eigen_utils.py`` module;
import the same names from ``neural_lofi.training.eigen``.
"""

from __future__ import annotations

from .chunked import (
    _is_hard_one_hot,
    auto_batch_size,
    covariance_chunk_signed,
    covariance_chunk_signed_accumulate_,
    covariance_chunk_signed_vector,
    covariance_chunk_signed_vector_accumulate_,
    flatten_spatial,
    flatten_spatial_vector,
    welford_cov_accumulate_,
)
from .deflation import (
    _deflate_symmetric_any,
    deflate_symmetric,
    deflate_symmetric_subspace,
    deflated_matvec,
)
from .dense import (
    EIGH_K_RATIO,
    LINOP_THRESHOLD_P,
    _signed_covariance_eigen_dense,
    _signed_covariance_eigen_linop,
    _to_sorted_tensors,
    signed_covariance_eigen,
    signed_covariance_full_eigenvalues,
    signed_covariance_full_eigenvalues_from_covariance,
    top_k_eig_from_covariance,
)
from .prepend import (
    LINEAR_MEAN_EPS,
    linear_mean_prepend_block,
    linear_mean_prepend_block_from_covariance,
    normalize_linear_mean,
    prepend_linear_mean,
)
from .spectra import (
    SPARSE_SPECTRA_P,
    covariance_spectra,
    covariance_spectra_from_covariances,
)
from .vector import (
    alternating_vector_eigen,
    alternating_vector_eigen_transverse,
    alternating_vector_eigen_transverse_from_covariance,
)

__all__ = [
    "EIGH_K_RATIO",
    "LINEAR_MEAN_EPS",
    "LINOP_THRESHOLD_P",
    "SPARSE_SPECTRA_P",
    # Private helpers re-exported for back-compat with the former flat module
    # (the eigensolver internals are monkeypatched by name in the test suite).
    "_deflate_symmetric_any",
    "_is_hard_one_hot",
    "_signed_covariance_eigen_dense",
    "_signed_covariance_eigen_linop",
    "_to_sorted_tensors",
    "alternating_vector_eigen",
    "alternating_vector_eigen_transverse",
    "alternating_vector_eigen_transverse_from_covariance",
    "auto_batch_size",
    "covariance_chunk_signed",
    "covariance_chunk_signed_accumulate_",
    "covariance_chunk_signed_vector",
    "covariance_chunk_signed_vector_accumulate_",
    "covariance_spectra",
    "covariance_spectra_from_covariances",
    "deflate_symmetric",
    "deflate_symmetric_subspace",
    "deflated_matvec",
    "flatten_spatial",
    "flatten_spatial_vector",
    "linear_mean_prepend_block",
    "linear_mean_prepend_block_from_covariance",
    "normalize_linear_mean",
    "prepend_linear_mean",
    "signed_covariance_eigen",
    "signed_covariance_full_eigenvalues",
    "signed_covariance_full_eigenvalues_from_covariance",
    "top_k_eig_from_covariance",
    "welford_cov_accumulate_",
]
