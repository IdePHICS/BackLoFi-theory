"""General-purpose helper functions."""


import logging
import math
import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

_SQRT2 = math.sqrt(2.0)
_PI_OVER_4 = math.pi / 4.0

# ---------------------------------------------------------------------------
# Activation registries
# ---------------------------------------------------------------------------


class _Erf(nn.Module):
    """Element-wise error function (no built-in ``nn.Erf`` in PyTorch)."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.erf(x)


def _step(x: Tensor) -> Tensor:
    """Heaviside step: 1 if ``x > 0`` else 0, cast to the input dtype."""
    return (x > 0).to(x.dtype)


class _Step(nn.Module):
    """Heaviside step: 1 if x>0 else 0.  Gradient-free; ridge-spectral only."""

    def forward(self, x: Tensor) -> Tensor:
        return (x > 0).to(x.dtype)


def _capped_relu(x: Tensor) -> Tensor:
    """ReLU saturated at 1: ``clamp(x, 0, 1)``."""
    return x.clamp(0.0, 1.0)


class _CappedReLU(nn.Module):
    """ReLU saturated at 1: max(0, min(x, 1))."""

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp(0.0, 1.0)


def _relu_normalized(x: Tensor) -> Tensor:
    """ReLU then per-sample L2 normalize on the last dim (FFN-only)."""
    # Spherical arc-cosine kernel feature map. Assumes 2D input (N, P);
    # higher-rank tensors (e.g. conv outputs) are not what the kernel intends.
    z = F.relu(x)
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class _ReLUNormalized(nn.Module):
    """ReLU then per-sample L2 normalize on the last dim (FFN-only)."""

    def forward(self, x: Tensor) -> Tensor:
        return _relu_normalized(x)


def _poly2(x: Tensor) -> Tensor:
    """Polynomial activation x²."""
    return x * x


class _Poly2(nn.Module):
    """Polynomial activation x²."""

    def forward(self, x: Tensor) -> Tensor:
        return x * x


def _poly3(x: Tensor) -> Tensor:
    """Polynomial activation x³."""
    return x * x * x


class _Poly3(nn.Module):
    """Polynomial activation x³."""

    def forward(self, x: Tensor) -> Tensor:
        return x * x * x


class _Cos(nn.Module):
    """Cosine activation; with sin/cos features, recovers the RBF kernel."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(x)


class _Sin(nn.Module):
    """Sine activation; with sin/cos features, recovers the RBF kernel."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


def _rbf(x: Tensor) -> Tensor:
    """Han et al. single-activation RBF: ``√2 sin(√2 t + π/4)``."""
    # √2 sin(θ + π/4) = sin θ + cos θ, so one activation reproduces the
    # (sin, cos) RFF construction with a deterministic phase shift.
    return _SQRT2 * torch.sin(_SQRT2 * x + _PI_OVER_4)


class _RBF(nn.Module):
    """Han et al. single-activation RBF: √2 sin(√2 t + π/4)."""

    def forward(self, x: Tensor) -> Tensor:
        return _SQRT2 * torch.sin(_SQRT2 * x + _PI_OVER_4)


# NOTE: ``torch.erf`` is used because ``F.erf`` does not exist in PyTorch.
# ``F.gelu`` defaults to the exact (erf-based) GELU, which is the form with
# a closed-form NNGP kernel.
ACTIVATIONS: dict[str, Callable[..., Tensor]] = {
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "relu": F.relu,
    "relu6": F.relu6,
    "erf": torch.erf,
    "gelu": F.gelu,
    "step": _step,
    "capped_relu": _capped_relu,
    "relu_normalized": _relu_normalized,
    "poly2": _poly2,
    "poly3": _poly3,
    "cos": torch.cos,
    "sin": torch.sin,
    "rbf": _rbf,
    "identity": lambda x: x,
}

ACTIVATION_MODULES: dict[str, nn.Module] = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "erf": _Erf(),
    "gelu": nn.GELU(),
    "step": _Step(),
    "capped_relu": _CappedReLU(),
    "relu_normalized": _ReLUNormalized(),
    "poly2": _Poly2(),
    "poly3": _Poly3(),
    "cos": _Cos(),
    "sin": _Sin(),
    "rbf": _RBF(),
    "identity": nn.Identity(),
}

# Canonical-name alias (Phase C of the post-unify-spectral naming cleanup).
# The plural mismatch between the pre-existing ``ACTIVATIONS`` (functions)
# and ``ACTIVATION_MODULES`` (``nn.Module`` instances) obscured the parallel
# structure of the two registries.  ``ACTIVATION_FUNCTIONS`` is the canonical
# name; ``ACTIVATIONS`` remains importable as an alias.
ACTIVATION_FUNCTIONS = ACTIVATIONS


# ---------------------------------------------------------------------------
# Activation derivatives (backward Neural LoFi)
# ---------------------------------------------------------------------------
#
# Backward Neural LoFi replaces the activation feature map ``σ(h)`` by the
# *derivative feature map* ``σ'(h)`` of the layer above (the gate that
# backpropagation applies to the readout signal).  Each entry below is the
# exact elementwise derivative of the same-named entry in ``ACTIVATIONS``.
#
# Deliberately absent:
#
# - ``step``: its derivative is 0 almost everywhere, so the derivative
#   feature map is identically zero and the backward correction is
#   degenerate.  Callers should reject it with a clear error rather than
#   silently produce zero effective labels.
# - ``relu_normalized``: the per-sample L2 normalization makes it
#   non-elementwise, so it has no diagonal derivative map.


_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _d_sigmoid(x: Tensor) -> Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s)


def _d_tanh(x: Tensor) -> Tensor:
    return 1.0 - torch.tanh(x) ** 2


def _d_relu6(x: Tensor) -> Tensor:
    return ((x > 0) & (x < 6)).to(x.dtype)


def _d_erf(x: Tensor) -> Tensor:
    return _TWO_OVER_SQRT_PI * torch.exp(-x * x)


def _d_gelu(x: Tensor) -> Tensor:
    # Exact (erf-based) GELU: d/dx [x Φ(x)] = Φ(x) + x φ(x).
    phi_cdf = 0.5 * (1.0 + torch.erf(x * _INV_SQRT2))
    phi_pdf = _INV_SQRT_2PI * torch.exp(-0.5 * x * x)
    return phi_cdf + x * phi_pdf


def _d_capped_relu(x: Tensor) -> Tensor:
    return ((x > 0) & (x < 1)).to(x.dtype)


def _d_rbf(x: Tensor) -> Tensor:
    # d/dx [√2 sin(√2 x + π/4)] = 2 cos(√2 x + π/4).
    return 2.0 * torch.cos(_SQRT2 * x + _PI_OVER_4)


ACTIVATION_DERIVATIVES: dict[str, Callable[..., Tensor]] = {
    "sigmoid": _d_sigmoid,
    "tanh": _d_tanh,
    "relu": _step,
    "relu6": _d_relu6,
    "erf": _d_erf,
    "gelu": _d_gelu,
    "capped_relu": _d_capped_relu,
    "poly2": lambda x: 2.0 * x,
    "poly3": lambda x: 3.0 * x * x,
    "cos": lambda x: -torch.sin(x),
    "sin": torch.cos,
    "rbf": _d_rbf,
    "identity": torch.ones_like,
}


def resolve_device(device: str) -> str:
    """Resolve a device string, expanding ``"auto"`` to the best available."""
    logger = logging.getLogger(__name__)
    if device == "auto":
        if torch.backends.mps.is_available():  # pragma: no cover - MPS only
            logger.info("Device resolved to 'mps'")
            return "mps"
        if torch.cuda.is_available():  # pragma: no cover - CUDA only
            logger.info("Device resolved to 'cuda'")
            return "cuda"
        logger.info(
            "Device resolved to 'cpu'"
        )  # pragma: no cover - reached only on machines without MPS or CUDA
        return "cpu"
    logger.info("Using device '%s'", device)
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy and stdlib."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA only
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
