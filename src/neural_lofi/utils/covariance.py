"""Signed and unsigned covariance primitives.

Single source of truth for the label-aware covariance
``C = (1/N) Z^T diag(y) Z`` and the plain covariance ``(1/N) Z^T Z`` used
throughout the ridge spectral trainers.  Every eigendecomposition path —
dense, chunked-accumulation, and matrix-free Lanczos — builds on these
functions, so the formula and its symmetrisation live in exactly one place.

The functions are *mechanical*: they operate on whatever device and dtype
the inputs carry.  Policy decisions (when to move large ``P`` to CPU
float32, when to use a matrix-free path) belong to the callers, not here.
"""


from torch import Tensor


def signed_covariance_matrix(
    z: Tensor,
    y: Tensor,
    *,
    normalize: bool = True,
    symmetrize: bool = True,
) -> Tensor:
    """Signed (label-aware) covariance ``C = Z^T diag(y) Z``.

    Parameters
    ----------
    z : Tensor
        Feature matrix ``(N, P)``.
    y : Tensor
        Targets, broadcastable to ``(N,)`` (e.g. ``(N,)`` or ``(N, 1)``).
        Cast to ``z``'s dtype before use.
    normalize : bool, default True
        Divide by ``N``.  Set ``False`` when accumulating partial sums
        across chunks and normalising once at the end.
    symmetrize : bool, default True
        Return ``0.5 (C + C^T)``.  The matrix is symmetric in exact
        arithmetic; this removes the float-rounding asymmetry that would
        otherwise perturb downstream ``eigsh``/``eigh``.

    Returns
    -------
    Tensor
        The ``(P, P)`` covariance, on the same device/dtype as *z*.
    """
    n = z.shape[0]
    y_flat = y.reshape(n).to(z.dtype)
    c = (y_flat.unsqueeze(1) * z).T @ z
    if normalize:
        c = c / n
    if symmetrize:
        c = 0.5 * (c + c.T)
    return c


def covariance_matrix(
    z: Tensor,
    *,
    normalize: bool = True,
    symmetrize: bool = True,
) -> Tensor:
    """Unsigned covariance ``C = Z^T Z`` (``signed`` with ``y = 1``).

    Parameters
    ----------
    z : Tensor
        Feature matrix ``(N, P)``.
    normalize : bool, default True
        Divide by ``N``.
    symmetrize : bool, default True
        Return ``0.5 (C + C^T)``.

    Returns
    -------
    Tensor
        The ``(P, P)`` covariance, on the same device/dtype as *z*.
    """
    n = z.shape[0]
    c = z.T @ z
    if normalize:
        c = c / n
    if symmetrize:
        c = 0.5 * (c + c.T)
    return c


def signed_covariance_matvec(
    z: Tensor,
    y: Tensor,
    v: Tensor,
    *,
    normalize: bool = True,
) -> Tensor:
    """Matrix-free product ``C v = Z^T (y ⊙ (Z v))`` for ``C = Z^T diag(y) Z``.

    Computes the action of :func:`signed_covariance_matrix` on *v* without
    forming the ``(P, P)`` matrix — ``O(N·P)`` instead of ``O(N·P^2)``.
    Used by the LinearOperator (Lanczos) eigendecomposition path.

    Parameters
    ----------
    z : Tensor
        Feature matrix ``(N, P)``.
    y : Tensor
        Targets, broadcastable to ``(N,)``.
    v : Tensor
        Vector ``(P,)`` or block of vectors ``(P, m)``.
    normalize : bool, default True
        Divide by ``N`` (must match the matrix form it stands in for).

    Returns
    -------
    Tensor
        ``C v`` with the same trailing shape as *v*.
    """
    n = z.shape[0]
    y_flat = y.reshape(n).to(z.dtype)
    zv = z @ v
    yzv = y_flat * zv if v.ndim == 1 else y_flat.unsqueeze(1) * zv
    out = z.T @ yzv
    if normalize:
        out = out / n
    return out
