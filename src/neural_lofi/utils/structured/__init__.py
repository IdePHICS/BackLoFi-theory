"""Structured-transform compute backends.

This subpackage holds the *pure functional* compute primitives behind the
structured random transforms in :mod:`neural_lofi.utils.transforms` — the
Walsh–Hadamard transform and the fused SORF operator — together with their
optional accelerated backends (the Dao-AILab fused Hadamard kernel and the
first-party fused Triton SORF kernel).  It deliberately contains no
``nn.Module`` and no model knowledge: tensors in, tensors out.  The
``nn.Module`` wrappers that own buffers, conv geometry and serialization live
in :mod:`neural_lofi.utils.transforms` and call into here.
"""

from __future__ import annotations

from .hadamard import fwht, is_pow2, next_pow2, pad_to
from .sorf import conv_sorf_project, fc_sorf_project, sorf_project

__all__ = [
    "conv_sorf_project",
    "fc_sorf_project",
    "fwht",
    "is_pow2",
    "next_pow2",
    "pad_to",
    "sorf_project",
]
