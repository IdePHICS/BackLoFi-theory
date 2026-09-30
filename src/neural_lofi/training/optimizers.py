"""Optimizers for the GD twin — one registry, model-agnostic.

``torch.optim`` ships SGD / Adam / AdamW but **no** layer-wise adaptive optimizer,
so the two trust-ratio methods live here:

* :class:`LARS` (You et al., 2017) — SGD+momentum with a per-tensor trust ratio.
* :class:`LAMB` (You et al., 2019) — the Adam counterpart of the same idea.

Both take the effective step to ``lr * trust * ‖w‖`` per weight tensor regardless
of the gradient scale, which is what makes them usable on the folded lofi
warm-start (whose per-layer weight norms span ~3 orders of magnitude).  Bias and
other 1-D parameters are excluded from BOTH the trust adaptation and weight decay
— the standard carve-out that keeps normalization/bias terms on a plain step.

:func:`build_optimizer` resolves any of ``sgd | adam | adamw | lars | lamb`` from
a name and forwards only the kwargs that optimizer accepts, so callers can pass a
single flat config.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = ["LAMB", "LARS", "OPTIMIZERS", "build_optimizer"]


def _trust_ratio(
    w_norm: Tensor, u_norm: Tensor, coefficient: float, eps: float
) -> Tensor | float:
    """``coefficient * ‖w‖ / (‖u‖ + eps)``; 1.0 when either norm vanishes."""
    if w_norm > 0 and u_norm > 0:
        return coefficient * w_norm / (u_norm + eps)
    return 1.0


class LARS(torch.optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling (You et al., 2017), SGD+momentum base.

    Per weight tensor the step is ``lr * local_lr * (g + wd*w)`` with
    ``local_lr = trust * ‖w‖ / (‖g + wd*w‖ + eps)``.
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                trust_coefficient=trust_coefficient,
                eps=eps,
            ),
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            wd, eps = group["weight_decay"], group["eps"]
            trust = group["trust_coefficient"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                excluded = p.ndim <= 1  # bias / 1-D: plain SGD, no wd, no trust
                if excluded:
                    local_lr: Tensor | float = 1.0
                else:
                    if wd != 0.0:
                        g = g.add(p, alpha=wd)
                    local_lr = _trust_ratio(
                        torch.linalg.vector_norm(p),
                        torch.linalg.vector_norm(g),
                        trust,
                        eps,
                    )
                    g = g * local_lr
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(p)
                buf.mul_(mom).add_(g)
                p.add_(buf, alpha=-lr)
        return loss


class LAMB(torch.optim.Optimizer):
    """Layer-wise Adaptive Moments (You et al., 2019) — LARS with an Adam base.

    The Adam update ``u = m̂ / (√v̂ + eps) + wd*w`` is rescaled by the same trust
    ratio ``trust * ‖w‖ / ‖u‖``, so the step is again ``~ lr * trust * ‖w‖``.
    ``trust_coefficient`` defaults to 1.0 (the paper's plain ratio) — unlike LARS,
    whose ``u`` is a raw gradient and needs the 1e-3 damping.
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        trust_coefficient: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                trust_coefficient=trust_coefficient,
                eps=eps,
            ),
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2) = group["lr"], group["betas"]
            wd, eps = group["weight_decay"], group["eps"]
            trust = group["trust_coefficient"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                # bias-corrected Adam direction
                u = (m / (1.0 - b1**t)) / ((v / (1.0 - b2**t)).sqrt() + eps)
                excluded = p.ndim <= 1  # bias / 1-D: plain Adam, no wd, no trust
                if excluded:
                    ratio: Tensor | float = 1.0
                else:
                    if wd != 0.0:
                        u = u.add(p, alpha=wd)  # decoupled: applied to the direction
                    ratio = _trust_ratio(
                        torch.linalg.vector_norm(p),
                        torch.linalg.vector_norm(u),
                        trust,
                        eps,
                    )
                p.add_(u * ratio if not excluded else u, alpha=-lr)
        return loss


OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "lars": LARS,
    "lamb": LAMB,
}

# Which of the flat kwargs each optimizer actually accepts.
_ACCEPTS: dict[str, frozenset[str]] = {
    "sgd": frozenset({"momentum", "weight_decay"}),
    "adam": frozenset({"betas", "weight_decay", "eps"}),
    "adamw": frozenset({"betas", "weight_decay", "eps"}),
    "lars": frozenset({"momentum", "weight_decay", "trust_coefficient", "eps"}),
    "lamb": frozenset({"betas", "weight_decay", "trust_coefficient", "eps"}),
}


def build_optimizer(
    params: Any,
    name: str,
    *,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    trust_coefficient: float | None = None,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float | None = None,
) -> torch.optim.Optimizer:
    """Build ``name`` (``sgd|adam|adamw|lars|lamb``) from a flat kwarg set.

    Kwargs the chosen optimizer does not accept are dropped, so one config block
    can drive any of them.  ``params`` is anything ``torch.optim`` takes (an
    iterable of tensors or a list of param-group dicts).  ``trust_coefficient`` /
    ``eps`` default to each optimizer's own default when left ``None``.
    """
    key = name.strip().lower()
    if key not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer: {name!r} (one of {sorted(OPTIMIZERS)})")
    offered: dict[str, Any] = {
        "momentum": momentum,
        "weight_decay": weight_decay,
        "betas": betas,
    }
    if trust_coefficient is not None:
        offered["trust_coefficient"] = trust_coefficient
    if eps is not None:
        offered["eps"] = eps
    kwargs = {k: v for k, v in offered.items() if k in _ACCEPTS[key]}
    cls: Any = OPTIMIZERS[key]  # torch.optim.Optimizer's base __init__ has no `lr`
    opt: torch.optim.Optimizer = cls(params, lr=lr, **kwargs)
    return opt
