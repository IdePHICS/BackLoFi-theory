"""Subspace-projection algebra for the signed-covariance eigensolvers.

Rank-update deflation of a symmetric matrix against a vector or an
orthonormal basis, plus a numpy matvec wrapper for the matrix-free Lanczos
path.  Leaf module: depends only on torch/numpy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor


def deflate_symmetric(c: Tensor, v0: Tensor) -> Tensor:
    """Project a symmetric matrix onto the orthogonal complement of ``v0``.

    Returns ``C̃ = P_perp C P_perp`` where ``P_perp = I − v0 v0^T``.  Uses
    the rank-2 update form ``C − v0 (v0^T C) − (C v0) v0^T + (v0^T C v0) v0 v0^T``
    which is cheaper and more float32-stable than building ``P_perp`` explicitly.
    Assumes ``v0`` is unit-norm.
    """
    cv = c @ v0
    vcv = v0 @ cv
    return c - torch.outer(v0, cv) - torch.outer(cv, v0) + vcv * torch.outer(v0, v0)


def deflate_symmetric_subspace(c: Tensor, u: Tensor) -> Tensor:
    """Project a symmetric matrix onto the orthogonal complement of ``span(U)``.

    ``u`` has shape ``(P, r)`` with orthonormal columns.  Returns
    ``C̃ = P_perp C P_perp`` with ``P_perp = I − U Uᵀ`` via the update form
    ``C − U(UᵀC) − (CU)Uᵀ + U(UᵀCU)Uᵀ`` (cheaper and more float32-stable than
    building ``P_perp`` explicitly).  Reduces to :func:`deflate_symmetric` when
    ``r = 1``.
    """
    cu = c @ u  # (P, r)
    ucu = u.transpose(0, 1) @ cu  # (r, r)
    return (
        c
        - u @ cu.transpose(0, 1)
        - cu @ u.transpose(0, 1)
        + u @ ucu @ u.transpose(0, 1)
    )


def _deflate_symmetric_any(c: Tensor, u: Tensor) -> Tensor:
    """Deflate ``c`` against a single vector ``(P,)`` or a basis ``(P, r)``."""
    return deflate_symmetric(c, u) if u.ndim == 1 else deflate_symmetric_subspace(c, u)


def deflated_matvec(
    matvec: Callable[[np.ndarray], np.ndarray], u: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a matvec so each application is ``v → P_perp (matvec(P_perp v))``.

    ``u`` is a unit-norm numpy vector ``(P,)`` or an orthonormal basis
    ``(P, r)``; both projections are O(P·r).  Used by the LinearOperator eigen
    path to compute eigenpairs of the deflated signed covariance without ever
    forming ``C`` or ``P_perp`` explicitly.
    """

    if u.ndim == 1:

        def _deflated(v: np.ndarray) -> np.ndarray:
            v_perp = v - u * float(u @ v)
            out = matvec(v_perp)
            return out - u * float(u @ out)

        return _deflated

    def _deflated_subspace(v: np.ndarray) -> np.ndarray:
        v_perp = v - u @ (u.T @ v)
        out = matvec(v_perp)
        return out - u @ (u.T @ out)

    return _deflated_subspace
