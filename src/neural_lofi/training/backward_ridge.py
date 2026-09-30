"""Joint ridge fit of the paired row factors: Algorithm 1's column half.

This is the ``C*`` fit of the ICLR draft, ``backward_correction.tex``
lines 382--438, used by the merged trainer in place of the lifted right
singular vectors (``implemented_algorithm.tex``, Algorithm 3 line 91).

Objective (eq:label-fit).  With features ``Z ∈ R^{n×p}`` of the block below
the boundary, scores ``S ∈ R^{n×r}`` (the backward signal projected on the
installed column directions) and a dictionary ``P ∈ R^{p×d}`` of neuron
patterns, put ``X = Z P`` and solve

    C* = argmin_C  (1/2n) Σ_μ [ y_μ − Σ_i S_{μi} (X C)_{μi} ]²
                   + (λ_eff/2) ‖C‖_F²,            Q̂ = P C* ∈ R^{p×r}.

Normal equations (eq:normal-equations):

    (1/n) Xᵀ [ S ⊙ (ŷ_C 1ᵀ) ] + λ_eff C = (1/n) Xᵀ [ diag(y) S ],
    ŷ_C = rowsum((X C) ⊙ S),

solved matrix-free by preconditioned conjugate gradients with the Kronecker
preconditioner ``(XᵀX/n) ⊗ (SᵀS/n) + λ_eff I``.  ``λ_eff = λ·tr(JᵀJ/n)/(d r)``
with ``J_{μ,(j,i)} = X_{μj} S_{μi}`` the implicit design matrix, i.e. ``λ``
is relative to the mean diagonal of the Gram.

What is deliberately *not* here: the trust radius (eq:trust), the
training-MSE backtracking and the per-channel normalisation of the draft.
``Q̂`` is returned with the scale the fit gives it, which is a linearisation
artefact (the scores are small, so the fitted coordinates come out an order
of magnitude above the forward ones); the caller scales the block by one
common factor and interpolates toward it with its damping ``α``.  ``λ`` is
chosen on a held-out split of the training set rather than fixed.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor

log = logging.getLogger(__name__)

DEFAULT_LAMBDAS: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@torch.no_grad()
def build_dictionary(cols: Tensor) -> Tensor:
    """Orthonormal basis ``P`` (p × d) of the span of ``cols`` (p × d0).

    Columns are normalised before the SVD and directions with singular value
    at most ``max(p, d0)·ε_mach·σ_max`` are discarded (the draft's convention,
    ``backward_subroutines.tex`` lines 32--33 and 60--63).  Zero columns are
    dropped first so a dead lift column cannot poison the scaling.
    """
    c = cols.double()
    norms = c.norm(dim=0)
    alive = norms > 0
    if not bool(alive.any()):
        return torch.zeros(cols.shape[0], 0, dtype=cols.dtype, device=cols.device)
    c = c[:, alive] / norms[alive]
    try:
        u, s, _vh = torch.linalg.svd(c, full_matrices=False)
    except Exception:  # divide-and-conquer failure on a degenerate block
        q, _r = torch.linalg.qr(c)
        return q.to(cols.dtype)
    eps = torch.finfo(torch.float64).eps
    cut = max(c.shape) * eps * float(s[0]) if s.numel() else 0.0
    keep = int((s > cut).sum().item())
    return u[:, :keep].to(cols.dtype)


def _forward(x: Tensor, s: Tensor, c: Tensor) -> Tensor:
    """``ŷ_C = rowsum((X C) ⊙ S)``."""
    return ((x @ c) * s).sum(dim=1)


def _normal_op(x: Tensor, s: Tensor, c: Tensor, lam: float) -> Tensor:
    """Left-hand side of the normal equations applied to ``C``."""
    n = x.shape[0]
    yhat = _forward(x, s, c)
    return x.transpose(0, 1) @ (s * yhat.unsqueeze(1)) / n + lam * c


class _KroneckerPreconditioner:
    """``(A ⊗ B + λI)⁻¹`` for ``A = XᵀX/n``, ``B = SᵀS/n`` in matrix form.

    For ``C`` (d × r) the operator is ``C ↦ A C B + λ C``; with
    ``A = E_a Λ_a E_aᵀ`` and ``B = E_b Λ_b E_bᵀ`` its inverse is
    ``E_a [ (E_aᵀ R E_b) / (λ_a λ_bᵀ + λ) ] E_bᵀ``.
    """

    def __init__(self, x: Tensor, s: Tensor, lam: float) -> None:
        n = x.shape[0]
        a = (x.transpose(0, 1).double() @ x.double()) / n
        b = (s.transpose(0, 1).double() @ s.double()) / n
        la, ea = torch.linalg.eigh(a)
        lb, eb = torch.linalg.eigh(b)
        self.ea, self.eb = ea.to(x.dtype), eb.to(x.dtype)
        denom = la.clamp_min(0).unsqueeze(1) * lb.clamp_min(0).unsqueeze(0) + lam
        self.inv = (1.0 / denom).to(x.dtype)

    def __call__(self, r: Tensor) -> Tensor:
        t = self.ea.transpose(0, 1) @ r @ self.eb
        return self.ea @ (t * self.inv) @ self.eb.transpose(0, 1)


@torch.no_grad()
def _solve(
    x: Tensor,
    s: Tensor,
    y: Tensor,
    lam: float,
    *,
    c0: Tensor | None,
    max_iter: int,
    tol: float,
) -> tuple[Tensor, dict[str, Any]]:
    """Preconditioned CG on the normal equations; returns ``(C, info)``."""
    n = x.shape[0]
    rhs = x.transpose(0, 1) @ (s * y.unsqueeze(1)) / n
    c = torch.zeros_like(rhs) if c0 is None else c0.clone()
    pre = _KroneckerPreconditioner(x, s, lam)
    r = rhs - _normal_op(x, s, c, lam)
    z = pre(r)
    p = z.clone()
    rz = float((r * z).sum())
    rhs_norm = max(float(rhs.norm()), 1e-30)
    it, res = 0, float(r.norm()) / rhs_norm
    for it in range(1, max_iter + 1):
        ap = _normal_op(x, s, p, lam)
        pap = float((p * ap).sum())
        if not math.isfinite(pap) or pap <= 0.0:
            log.warning("ridge CG: non-positive curvature at iteration %d.", it)
            break
        step = rz / pap
        c = c + step * p
        r = r - step * ap
        res = float(r.norm()) / rhs_norm
        if res <= tol:
            break
        z = pre(r)
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    return c, {"iterations": it, "residual": res}


@torch.no_grad()
def fit_paired_rows(
    z: Tensor,
    y: Tensor,
    s: Tensor,
    dictionary: Tensor,
    *,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    holdout: float = 0.10,
    select_n: int | None = 10_000,
    cg_iters: int = 240,
    cg_tol: float = 1e-4,
    seed: int = 0,
) -> tuple[Tensor, dict[str, Any]]:
    """``Q̂ = P C*`` for the paired rows of one boundary.

    Parameters
    ----------
    z, y, s
        Features of the lower block ``(n, p)``, centred scalar labels ``(n,)``
        and scores ``(n, r)``.
    dictionary
        ``P`` ``(p, d)`` from :func:`build_dictionary`.
    lambdas
        Relative ridge penalties, each scaled by the mean Gram diagonal.
    holdout, select_n
        The penalty is picked by the held-out linearised MSE on a random
        ``holdout`` fraction of (at most ``select_n`` of) the training samples,
        then the winner is refitted on every sample.  ``select_n=None`` selects
        on the full set.
    """
    n, p = z.shape
    r = s.shape[1]
    d = dictionary.shape[1]
    dtype = z.dtype
    if d == 0 or r == 0:
        return torch.zeros(p, r, dtype=dtype, device=z.device), {"empty": True}
    x = z @ dictionary.to(dtype)  # (n, d)
    yv = y.to(dtype).reshape(n)
    sv = s.to(dtype)

    # Mean diagonal of the implicit Gram: tr(JᵀJ/n)/(d r).
    diag = float((x.pow(2).sum(dim=1) * sv.pow(2).sum(dim=1)).sum()) / (n * d * r)
    diag = max(diag, torch.finfo(torch.float64).tiny)

    # --- penalty selection on a held-out split ------------------------------
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_sel = n if select_n is None else min(n, int(select_n))
    perm = torch.randperm(n, generator=gen)[:n_sel].to(z.device)
    n_ho = max(1, int(round(holdout * n_sel)))
    ho, tr = perm[:n_ho], perm[n_ho:]
    scores: list[tuple[float, float]] = []
    c_prev: Tensor | None = None
    for lam_rel in sorted(lambdas, reverse=True):  # warm start, damped first
        lam = lam_rel * diag
        c_tr, _ = _solve(
            x[tr], sv[tr], yv[tr], lam, c0=c_prev, max_iter=cg_iters, tol=cg_tol
        )
        c_prev = c_tr
        mse = float((_forward(x[ho], sv[ho], c_tr) - yv[ho]).pow(2).mean())
        scores.append((lam_rel, mse))
    best_rel = min(scores, key=lambda t: t[1])[0]

    # --- refit on every sample with the selected penalty --------------------
    c_star, info = _solve(
        x, sv, yv, best_rel * diag, c0=None, max_iter=cg_iters, tol=cg_tol
    )
    if not bool(torch.isfinite(c_star).all()):
        log.warning("ridge: non-finite solution, returning a zero proposal.")
        c_star = torch.zeros_like(c_star)
    q_hat = dictionary.to(dtype) @ c_star  # (p, r)
    train_mse = float((_forward(x, sv, c_star) - yv).pow(2).mean())
    return q_hat, {
        "lambda_rel": best_rel,
        "lambda_eff": best_rel * diag,
        "gram_diag": diag,
        "dictionary_dim": d,
        "holdout_mse": {f"{lr:g}": m for lr, m in scores},
        "train_mse": train_mse,
        **info,
    }
