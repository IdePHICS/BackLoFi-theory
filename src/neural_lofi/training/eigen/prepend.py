"""Linear-mean (degree-1) direction handling and prepend blocks.

The label-weighted mean ``u = E[y·z]`` defines a degree-1 filter ``v0 = u/‖u‖``
that NLoFi optionally prepends to the eigenvector bank.  These helpers build the
``[v0 | top-k eigenvectors]`` block either from a feature matrix ``(z, y)`` or
from an already-accumulated ``(P, P)`` covariance.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from .deflation import deflate_symmetric
from .dense import signed_covariance_eigen, top_k_eig_from_covariance

log = logging.getLogger(__name__)

# Threshold below which the label-weighted mean direction is treated as
# numerically zero and the linear-mean prepend is skipped in favor of an
# extra eigenvector. Matches float32 epsilon scale.
LINEAR_MEAN_EPS = 1e-7


def normalize_linear_mean(u: Tensor) -> tuple[Tensor, float]:
    """Normalize ``u`` to unit length, returning ``(v0, ||u||)``.

    When ``||u|| < LINEAR_MEAN_EPS`` the direction is degenerate; callers
    should fall back to taking one extra eigenvector instead of prepending.
    """
    norm = float(u.norm())
    if norm < LINEAR_MEAN_EPS:
        return torch.zeros_like(u), norm
    return u / norm, norm


def prepend_linear_mean(v0: Tensor, eigvecs: Tensor) -> Tensor:
    """Stack ``[v0 | eigvecs]`` along the column axis."""
    return torch.cat([v0.unsqueeze(1), eigvecs], dim=1)


def linear_mean_prepend_block(
    z: Tensor,
    y: Tensor,
    *,
    k: int,
    include_linear_mean: bool,
    orthogonalize: bool,
    log_prefix: str = "",
) -> tuple[Tensor, Tensor]:
    """Top-*k* eigenpairs of the signed covariance, with optional ``v0`` prepend.

    Operates directly on a feature tensor ``z`` of shape ``(N, P)``.

    When ``include_linear_mean`` is False this is a plain call to
    :func:`signed_covariance_eigen`. Otherwise:

    1. Compute ``u = E[y · z]`` and ``v0 = u / ||u||``.
    2. If ``||u|| < LINEAR_MEAN_EPS`` (degenerate direction): log a warning
       and fall back to ``signed_covariance_eigen(z, y, k + 1)``.
    3. Otherwise: compute top-*k* eigenpairs of the (optionally deflated)
       signed covariance, prepend ``v0`` as the first eigenvector and its
       Rayleigh quotient ``E[(z @ v0)^2 · y]`` as the first eigenvalue.

    ``log_prefix`` is prepended to the fallback warning so callers can
    identify which layer triggered it (e.g. ``"Layer 3 (conv): "``).
    """
    if not include_linear_mean:
        return signed_covariance_eigen(z, y, k)

    u = (z * y.view(-1, 1)).mean(dim=0)
    v0, u_norm = normalize_linear_mean(u)
    if u_norm < LINEAR_MEAN_EPS:
        log.warning(
            "%s||u||=%.2e is below eps; taking %d eigenvectors of C as a fallback.",
            log_prefix,
            u_norm,
            k + 1,
        )
        return signed_covariance_eigen(z, y, k + 1)

    deflate = v0 if orthogonalize else None
    eigvals_k, eigvecs_k = signed_covariance_eigen(z, y, k, deflate_against=deflate)
    V = prepend_linear_mean(v0, eigvecs_k)
    v0_rq = ((z @ v0) ** 2 * y.view(-1)).mean().to(eigvals_k.dtype)
    eigvals = torch.cat([v0_rq.unsqueeze(0), eigvals_k])
    return eigvals, V


def linear_mean_prepend_block_from_covariance(
    c: Tensor,
    u: Tensor,
    *,
    k: int,
    include_linear_mean: bool,
    orthogonalize: bool,
    target_dtype: torch.dtype,
    target_device: torch.device,
    log_prefix: str = "",
) -> tuple[Tensor, Tensor]:
    """Top-*k* eigenpairs of an accumulated signed covariance, with ``v0`` prepend.

    Symmetric counterpart to :func:`linear_mean_prepend_block` for callers
    that have already accumulated the dense covariance ``c`` (shape
    ``(P, P)``) and label-weighted mean ``u`` (shape ``(P,)``) across
    mini-batches. Both ``c`` and ``u`` are expected on CPU as float32; the
    returned eigenpairs are moved to ``target_device``/``target_dtype``.

    Fallback and prepend semantics match :func:`linear_mean_prepend_block`;
    deflation uses :func:`deflate_symmetric` on ``c`` (not an
    ``eigen_fn``-side kwarg) because the underlying eigensolver here
    consumes a dense matrix rather than ``(z, y)``.
    """
    if not include_linear_mean:
        eigvals_t, eigvecs_t = top_k_eig_from_covariance(c, k)
        return (
            eigvals_t.to(dtype=target_dtype, device=target_device),
            eigvecs_t.to(dtype=target_dtype, device=target_device),
        )

    v0_cpu, u_norm = normalize_linear_mean(u)
    if u_norm < LINEAR_MEAN_EPS:
        log.warning(
            "%s||u||=%.2e is below eps; taking %d eigenvectors of C as a fallback.",
            log_prefix,
            u_norm,
            k + 1,
        )
        eigvals_t, eigvecs_t = top_k_eig_from_covariance(c, k + 1)
        return (
            eigvals_t.to(dtype=target_dtype, device=target_device),
            eigvecs_t.to(dtype=target_dtype, device=target_device),
        )

    c_for_eigsh = deflate_symmetric(c, v0_cpu) if orthogonalize else c
    eigvals_k_t, eigvecs_k_t = top_k_eig_from_covariance(c_for_eigsh, k)

    eigvals_k = eigvals_k_t.to(dtype=target_dtype, device=target_device)
    eigvecs_k = eigvecs_k_t.to(dtype=target_dtype, device=target_device)
    v0_dev = v0_cpu.to(dtype=target_dtype, device=target_device)
    V = prepend_linear_mean(v0_dev, eigvecs_k)
    v0_rq = (v0_cpu @ c @ v0_cpu).to(dtype=target_dtype, device=target_device)
    eigvals = torch.cat([v0_rq.unsqueeze(0), eigvals_k])
    return eigvals, V
