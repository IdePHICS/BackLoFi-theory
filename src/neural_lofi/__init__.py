"""neural_lofi — PyTorch implementation of NLoFi.

NLoFi (Nonlinear Layerwise Feature learning) trains deep networks one
layer at a time using signed-covariance eigenreduction on top of random
features.  This package provides the canonical spectral trainer plus a
parallel gradient-based trainer for comparison studies.

Subpackages
-----------
datasets       : Dataset loading and registry (torchvision + HuggingFace).
models         : ``SpectralModel`` (FFN + CNN), ``BackpropModel`` (FFN + CNN).
training       : ``SpectralTrainer``, ``BackpropTrainer``, kernel variants.
utils          : Shared helpers (activations, metrics, layer-config).
visualization  : Feature visualization (Fourier maximizer, Jacobian importance).

The package was named ``train_hierarchically`` before v0.3 and the
shim was removed in v0.4.
"""

__version__ = "0.5.0"
