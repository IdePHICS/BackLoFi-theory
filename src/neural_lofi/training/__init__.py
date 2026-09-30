"""neural_lofi.training — Model training orchestration.

Provides :class:`SpectralTrainer` as the unified entry point for spectral
training: it dispatches on a single axis — the label type (scalar ``(N,)`` vs
vector ``(N, c)``, from the rank of the targets) — while the model's
auto-detected layer composition handles feedforward vs convolutional, so one
trainer + config covers every case.  :class:`BackpropTrainer` provides the
gradient-based twin.

Vector (multi-class) targets are handled by passing ``(N, c)`` labels to the
same :class:`SpectralTrainer` — there is no separate vector trainer class.
"""

from .backprop import (
    BackpropConfig,
    BackpropTrainer,
)
from .backward_coupled import CoupledBackwardConfig, CoupledBackwardTrainer
from .backward_lofi import (
    BackwardSpectralConfig,
    BackwardSpectralTrainer,
    widen_spectral_model,
)
from .base import (
    BaseTrainer,
    BaseTrainerConfig,
    fit_linear_readout,
    fit_ridge_readout_from_covariance,
)
from .optimizers import LAMB, LARS, OPTIMIZERS, build_optimizer
from .spectral import SpectralTrainer, SpectralTrainerConfig

__all__ = [
    "LAMB",
    "LARS",
    "OPTIMIZERS",
    "BackpropConfig",
    "BackpropTrainer",
    "BackwardSpectralConfig",
    "BackwardSpectralTrainer",
    "BaseTrainer",
    "BaseTrainerConfig",
    "CoupledBackwardConfig",
    "CoupledBackwardTrainer",
    "SpectralTrainer",
    "SpectralTrainerConfig",
    "build_optimizer",
    "fit_linear_readout",
    "fit_ridge_readout_from_covariance",
    "widen_spectral_model",
]
