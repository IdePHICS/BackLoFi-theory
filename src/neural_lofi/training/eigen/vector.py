"""Vector-label (multi-class) alternating Rayleigh maximization.

The vector-label NLoFi inner loop alternates between a label direction ``a`` and
the top eigenvector ``v`` of the label-direction-weighted signed covariance
``Ĉ_a``.  Provides the materialised-feature form, the transverse (linear-SVD +
deflation) form, and the covariance-accumulation form used by conv layers.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .deflation import deflate_symmetric_subspace
from .dense import signed_covariance_eigen, top_k_eig_from_covariance
from .robust import safe_svd


def alternating_vector_eigen(
    z: Tensor,
    y: Tensor,
    k: int,
    *,
    max_iter: int = 20,
    tol: float = 1e-4,
    patience: int = 1,
) -> tuple[Tensor, Tensor, Tensor, list[float]]:
    """Alternating Rayleigh maximization for vector (multi-class) labels.

    Implements the per-layer inner loop of the vector-label NLoFi algorithm.
    The label-direction-weighted operator ``Ĉ_a = (1/N) Z^T diag(Y a) Z`` is
    the signed covariance with the scalar weight replaced by ``s_μ = a^T y_μ``,
    so its top eigenpair is obtained directly from
    :func:`signed_covariance_eigen` with ``y → s``.  The label direction is then
    updated in closed form by ``a ← r / ‖r‖`` with
    ``r = (1/N) Σ_μ (v^T z_μ)^2 y_μ``, which ascends the joint bilinear
    objective ``(a, v) ↦ a^T r(v)`` over the two unit spheres.  The scalar
    algorithm is recovered as the special case ``c = 1``, ``a = 1``.

    The "top eigenpair (e.g. by power iteration)" of the algorithm is the
    largest-*magnitude* eigenpair, which is exactly what
    :func:`signed_covariance_eigen` (``which="LM"``) returns; the sign is
    absorbed by the ``a`` update, since ``a`` ranges over the full sphere.

    Parameters
    ----------
    z : Tensor
        Feature matrix ``(N, P)``.
    y : Tensor
        Vector labels ``(N, c)``.
    k : int
        Number of filters (top eigenvectors of ``Ĉ_{a*}``) to return.
    max_iter : int
        Maximum number of alternating iterations ``T``.
    tol : float
        Convergence tolerance on ``|λ^(t) − λ^(t-1)|``.
    patience : int
        Number of *consecutive* below-``tol`` steps required before stopping.
        ``patience=1`` stops as soon as the tolerance is first met; larger
        values run extra iterations past convergence so the plateau is visible
        in the recorded ``lambda_trajectory``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, list[float]]
        ``(a_star, eigvals, V, lambda_trajectory)`` — the optimal label
        direction ``a_star`` ``(c,)``, the top-``k`` eigenvalues ``(k,)`` of
        ``Ĉ_{a*}`` (descending ``|λ|``), the ``(P, k)`` filter matrix ``V``,
        and the list of leading ``|λ^(t)|`` recorded each iteration.

    Notes
    -----
    The scalar-method knobs ``include_linear_mean`` / ``orthogonalize`` are
    deliberately not used here: the vector algorithm extracts the top-``k``
    eigenvectors of ``Ĉ_{a*}`` directly (deflation at fixed ``a*``).
    """
    if y.ndim != 2:
        raise ValueError(f"y must be a 2-D (N, c) tensor, got shape {tuple(y.shape)}")
    y = y.to(device=z.device, dtype=z.dtype)

    # Init a^(0): top left singular vector of M = Y^T Z  (c, P).
    m = y.transpose(0, 1) @ z  # (c, P)
    u, _, _ = safe_svd(m)
    a = u[:, 0].contiguous()  # (c,)

    lambda_trajectory: list[float] = []
    prev_lambda: float | None = None
    stable = 0
    for _ in range(max_iter):
        s = y @ a  # (N,) scalar weights a^T y_μ
        eigvals_t, eigvecs_t = signed_covariance_eigen(z, s, 1)
        v = eigvecs_t[:, 0]  # (P,)
        lam = float(eigvals_t[0].abs())
        lambda_trajectory.append(lam)

        # Closed-form label-direction update r = (1/N) Σ (v^T z_μ)^2 y_μ.
        q = z @ v  # (N,)
        r = (q * q) @ y  # (c,)
        r_norm = float(r.norm())
        if r_norm > 0:
            a = r / r_norm

        if prev_lambda is not None and abs(lam - prev_lambda) < tol:
            stable += 1
            if stable >= patience:
                break
        else:
            stable = 0
        prev_lambda = lam

    eigvals, eigvecs = signed_covariance_eigen(z, y @ a, k)
    return a, eigvals, eigvecs, lambda_trajectory


def alternating_vector_eigen_transverse(
    z: Tensor,
    y: Tensor,
    k: int,
    *,
    linear_k: int,
    max_iter: int = 20,
    tol: float = 1e-4,
    patience: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[float]]:
    """Linear-SVD + transverse alternating Rayleigh maximization (vector labels).

    Generalizes the scalar ``include_linear_mean`` + ``orthogonalize_to_mean``
    recipe to vector labels.  The linear term is the cross-moment
    ``M = (1/N) Zᵀ Y ∈ ℝ^{P×c}``; its top ``linear_k`` left singular vectors
    ``U_lin`` are kept as degree-1 ("linear") filters.  The alternating
    ``(a, v)`` Rayleigh loop of :func:`alternating_vector_eigen` then runs on
    the **full** operator ``Ĉ_a`` (``U_lin`` is *not* deflated out inside the
    loop) to find the optimal label direction ``a*``.  Only the final
    ``k``-direction extraction is deflated against ``U_lin``, so the assembled
    bank ``V = [U_lin | V_perp]`` (width ``linear_k + k``) is orthonormal by
    construction (``V_perp ⟂ U_lin``) — the vector analog of the scalar
    ``orthogonalize_to_mean``.

    The label direction ``a``-init lives in label space ``(c,)``, so it is
    unaffected by the feature-space directions and there is no degeneracy.

    Parameters
    ----------
    z, y, k, max_iter, tol, patience
        As in :func:`alternating_vector_eigen`.
    linear_k : int
        Number of linear (degree-1) directions to prepend.  Must satisfy
        ``1 <= linear_k <= min(c, P)`` and ``k <= P - linear_k``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[float]]
        ``(a_star, U_lin, lin_svals, eigvals_perp, V, lambda_trajectory)`` —
        the optimal label direction ``(c,)``, the linear basis ``U_lin``
        ``(P, linear_k)``, the singular values of ``M`` ``(min(c, P),)``, the
        ``k`` transverse eigenvalues, the full ``(P, linear_k + k)`` filter
        bank, and the inner-loop ``|λ^(t)|`` trajectory.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be a 2-D (N, c) tensor, got shape {tuple(y.shape)}")
    y = y.to(device=z.device, dtype=z.dtype)
    P = z.shape[1]
    c = y.shape[1]
    if linear_k < 1:
        raise ValueError(f"linear_k must be >= 1, got {linear_k}")
    if linear_k > min(c, P):
        raise ValueError(f"linear_k={linear_k} exceeds min(c, P)={min(c, P)}")
    if k > P - linear_k:
        raise ValueError(f"k={k} must satisfy k <= P - linear_k = {P - linear_k}")

    # SVD of M = Yᵀ Z (c, P): label-space left singular vectors u (a-init) and
    # feature-space right singular vectors vh (the linear directions U_lin).
    m = y.transpose(0, 1) @ z  # (c, P)
    u, s, vh = safe_svd(m)
    a = u[:, 0].contiguous()  # (c,)
    u_lin = vh[:linear_k].transpose(0, 1).contiguous()  # (P, linear_k)
    lin_svals = s.contiguous()

    lambda_trajectory: list[float] = []
    prev_lambda: float | None = None
    stable = 0
    for _ in range(max_iter):
        sw = y @ a  # (N,) scalar weights aᵀ y_μ
        # Leading signed eigenpair of the FULL operator Ĉ_a (the alternating
        # Rayleigh loop runs on Ĉ_a itself — the linear directions U_lin are NOT
        # deflated out here; they are removed only at the final k-extraction).
        eigvals_t, eigvecs_t = signed_covariance_eigen(z, sw, 1)
        v = eigvecs_t[:, 0]  # (P,)
        lam = float(eigvals_t[0].abs())
        lambda_trajectory.append(lam)

        # Closed-form label-direction update r = (1/N) Σ (vᵀz_μ)² y_μ.
        q = z @ v  # (N,)
        r = (q * q) @ y  # (c,)
        r_norm = float(r.norm())
        if r_norm > 0:
            a = r / r_norm

        if prev_lambda is not None and abs(lam - prev_lambda) < tol:
            stable += 1
            if stable >= patience:
                break
        else:
            stable = 0
        prev_lambda = lam

    eigvals_perp, v_perp = signed_covariance_eigen(z, y @ a, k, deflate_against=u_lin)
    v_full = torch.cat([u_lin, v_perp], dim=1)  # (P, linear_k + k)
    return a, u_lin, lin_svals, eigvals_perp, v_full, lambda_trajectory


def alternating_vector_eigen_transverse_from_covariance(
    c_stack: Tensor,
    m: Tensor,
    k: int,
    *,
    linear_k: int,
    max_iter: int = 20,
    tol: float = 1e-4,
    patience: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[float]]:
    """Covariance-form of :func:`alternating_vector_eigen_transverse`.

    Runs the linear-SVD + transverse alternating Rayleigh maximization on
    *pre-accumulated* per-class signed covariances, so conv layers never need
    the materialised ``(N*H*W, P)`` patch-feature matrix.  Algebraically
    equivalent to the materialised path: the ``(a, v)`` loop runs on the full
    ``Ĉ_a = Σ_c a_c C_c`` and only the final ``k``-extraction is deflated
    against ``U_lin``; here both use the small ``(C, P, P)`` stack
    (``Ĉ_a`` via ``einsum``, ``r_c = vᵀ C_c v`` via ``einsum``).

    Parameters
    ----------
    c_stack : Tensor
        Per-class signed covariances ``(C, P, P)`` where
        ``c_stack[c] = (1/M) Σ_patch y_{·,c} z zᵀ`` — already divided by the
        total patch count *once* (one global denominator, never per-class).
    m : Tensor
        Cross-moment ``M = Zᵀ Y`` of shape ``(P, C)`` (raw; the SVD that uses
        it is scale-invariant).  ``c_stack`` and ``m`` must share device/dtype.
    k : int
        Number of transverse eigenvectors to return.
    linear_k : int
        Number of linear (degree-1) directions to prepend.  Must satisfy
        ``1 <= linear_k <= min(C, P)`` and ``k <= P - linear_k``.
    max_iter, tol, patience
        Inner-loop controls, as in :func:`alternating_vector_eigen_transverse`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[float]]
        ``(a_star, U_lin, lin_svals, eigvals_perp, V, lambda_trajectory)`` with
        the same shapes/semantics as :func:`alternating_vector_eigen_transverse`;
        ``V`` is ``(P, linear_k + k)``.
    """
    p, c = m.shape
    if linear_k < 1:
        raise ValueError(f"linear_k must be >= 1, got {linear_k}")
    if linear_k > min(c, p):
        raise ValueError(f"linear_k={linear_k} exceeds min(C, P)={min(c, p)}")
    if k > p - linear_k:
        raise ValueError(f"k={k} must satisfy k <= P - linear_k = {p - linear_k}")

    # SVD of M = Zᵀ Y (P, C): left singular vectors are the feature-space linear
    # directions U_lin; the leading right singular vector (Vh[0]) lives in label
    # space and seeds the alternating direction a.
    u, s, vh = safe_svd(m)
    u_lin = u[:, :linear_k].contiguous()  # (P, linear_k)
    a = vh[0].contiguous()  # (C,)
    lin_svals = s.contiguous()

    def _weighted(a_vec: Tensor) -> Tensor:
        # Ĉ_a = Σ_c a_c C_c (identity i), re-symmetrised (C-2) before each solve.
        a_dev = a_vec.to(dtype=c_stack.dtype, device=c_stack.device)
        c_hat = torch.einsum("c,cij->ij", a_dev, c_stack)
        return 0.5 * (c_hat + c_hat.transpose(0, 1))

    lambda_trajectory: list[float] = []
    prev_lambda: float | None = None
    stable = 0
    for _ in range(max_iter):
        # Leading signed eigenpair of the FULL operator Ĉ_a = Σ_c a_c C_c (the
        # Rayleigh loop runs on Ĉ_a itself; U_lin is deflated out only at the
        # final k-extraction below, not inside the loop).
        c_hat = _weighted(a)
        eigvals_t, eigvecs_t = top_k_eig_from_covariance(c_hat, 1)
        v = eigvecs_t[:, 0].to(device=c_stack.device)
        lam = float(eigvals_t[0].abs())
        lambda_trajectory.append(lam)

        # Closed-form label-direction update r_c = vᵀ C_c v (identity ii).
        r = torch.einsum("cij,i,j->c", c_stack, v, v)
        r_norm = float(r.norm())
        if r_norm > 0:
            a = r / r_norm

        if prev_lambda is not None and abs(lam - prev_lambda) < tol:
            stable += 1
            if stable >= patience:
                break
        else:
            stable = 0
        prev_lambda = lam

    c_final = deflate_symmetric_subspace(_weighted(a), u_lin)
    eigvals_perp, v_perp = top_k_eig_from_covariance(c_final, k)
    v_perp = v_perp.to(device=u_lin.device)
    v_full = torch.cat([u_lin, v_perp], dim=1)  # (P, linear_k + k)
    return a, u_lin, lin_svals, eigvals_perp, v_full, lambda_trajectory
