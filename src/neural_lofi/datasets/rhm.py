"""Random Hierarchy Model (RHM) — synthetic hierarchical classification dataset.

Re-implementation of the Wyart-group generative model (Cagnetta, Petrini,
Tomasini, Favero, Wyart — "How Deep Neural Networks Learn Compositional Data:
The Random Hierarchy Model", PRX 14, 031001 (2024); reference code:
https://github.com/pcsl-epfl/random-hierarchy-model).

The model is an ``L``-level ``s``-ary tree grammar over a vocabulary of ``v``
symbols.  The root is one of ``n_c`` class labels; every symbol at every level
expands into one of its ``m`` synonymic ``s``-tuples, down to the ``d = s**L``
leaves.  The input is the leaf string, one-hot encoded to ``(v, d)``
(channels-first); the label is the root class.  The number of distinct data
points is ``Pmax = n_c * m ** ((s**L - 1) // (s - 1))``.

Sampling never enumerates the dataset: following the reference code, a data
point is an integer in ``[0, Pmax)`` whose mixed-radix (base-``m``) digits are
the per-node rule choices, decoded level-by-level with vectorized torch ops —
O(N * s**L) regardless of ``Pmax``.  Rules are drawn with a torch generator,
so datasets are reproducible here but not bit-identical to the reference
(which uses Python's ``random``); the distribution is the same.

Split handling (this file's deviation from the reference, which draws train
and test jointly in one call): each split is a deterministic disjoint region
of a seeded sample stream, so the registry can build splits independently.

- ``Pmax <= 2**26`` (dense): one seeded ``randperm(Pmax)``; ``train`` reads
  from the head, ``test``/``val`` from the tail.  Exactly disjoint provided
  ``n_train + n_test <= Pmax`` (the builder cannot check this across calls).
  Supports whole-dataset training (``n_samples = Pmax``).
- ``Pmax <= 2**62`` (sparse): sequential rejection sampling from one seeded
  stream; the first ``test_pool`` (default 100k) distinct indices are
  reserved for ``test``/``val``, ``train`` continues the same stream.
  Exactly disjoint at any size.
- larger (huge): int64 indices would overflow, so sample generatively
  (random label + random rule choice per node, separate per-split seeds);
  collision probability ~ ``N**2 / Pmax`` is astronomically small.

``val`` is an alias for ``test`` (same region — the RHM has no third split).
``config.transform`` is ignored; ``config.seed`` selects the sample stream
(the grammar itself is fixed by ``params["seed_rules"]``).
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import DatasetConfig
from .factory import register

_DENSE_MAX = 1 << 26  # randperm cap: 512 MiB of int64 indices
_INDEX_MAX = 1 << 62  # above this, int64 index arithmetic is unsafe

_DEFAULTS: dict[str, Any] = {
    "num_features": 8,  # v — vocabulary size
    "num_synonyms": 2,  # m — synonymic tuples per symbol
    "tuple_size": 2,  # s — children per node
    "num_layers": 2,  # L — tree depth
    "seed_rules": 0,  # grammar seed (shared across splits = same task)
    "whitening": False,  # centre/scale the one-hot channels
    "test_pool": 100_000,  # sparse regime: stream prefix reserved for test
}


def _dec2base(idx: torch.Tensor, base: int, length: int) -> torch.Tensor:
    """Base-``base`` digits of *idx*, most-significant first — ``(*, length)``."""
    digits = []
    for _ in range(length):
        digits.append(idx % base)
        idx = idx // base
    return torch.stack(digits[::-1], dim=-1)


def _sample_rules(
    v: int, n_c: int, m: int, s: int, num_layers: int, seed: int
) -> list[torch.Tensor]:
    """Per-level rule tensors: level 0 is ``(n_c, m, s)``, deeper ``(v, m, s)``."""
    if v**s > _DENSE_MAX:
        raise ValueError(f"v**s = {v**s} too large to sample rules from")
    g = torch.Generator().manual_seed(seed)
    rules = []
    for level in range(num_layers):
        n_parents = n_c if level == 0 else v
        if n_parents * m > v**s:
            raise ValueError(
                f"RHM level {level} needs {n_parents}*{m} distinct s-tuples "
                f"but only v**s = {v**s} exist"
            )
        chosen = torch.randperm(v**s, generator=g)[: n_parents * m]
        rules.append(_dec2base(chosen, v, s).reshape(n_parents, m, s))
    return rules


def _split_indices(
    split: str, n: int, max_data: int, seed: int, test_pool: int
) -> torch.Tensor:
    """Deterministic disjoint sample indices for *split* (see module docstring)."""
    is_test = split != "train"
    if max_data <= _DENSE_MAX:
        if n > max_data:
            raise ValueError(f"n_samples={n} exceeds the {max_data} distinct data")
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(max_data, generator=g)
        return perm[max_data - n :] if is_test else perm[:n]

    if is_test and n > test_pool:
        raise ValueError(
            f"test n_samples={n} exceeds test_pool={test_pool}; "
            "raise params['test_pool'] (identically for the train run!)"
        )
    # ponytail: python-loop sequential rejection sampling, ~1s per 1e6 draws;
    # switch to a Feistel index permutation if this ever becomes the bottleneck.
    rng = random.Random(seed)
    need = n if is_test else test_pool + n
    seen: set[int] = set()
    stream: list[int] = []
    while len(stream) < need:
        i = rng.randrange(max_data)
        if i not in seen:
            seen.add(i)
            stream.append(i)
    return torch.tensor(stream[:n] if is_test else stream[test_pool:])


def _decode_choices(
    idx: torch.Tensor, n_c: int, m: int, s: int, num_layers: int, max_data: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Mixed-radix decode: index -> (labels, per-level rule-choice tensors)."""
    per_label = max_data // n_c
    labels = idx // per_label
    rem = idx % per_label
    choices = []
    size = 1
    for _ in range(num_layers):
        per_label //= m**size
        choices.append(_dec2base(rem // per_label, m, size))  # (N, size) in [0, m)
        rem = rem % per_label
        size *= s
    return labels, choices


def _generative_choices(
    split: str, n: int, n_c: int, m: int, s: int, num_layers: int, seed: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Huge-``Pmax`` fallback: sample label and rule choices directly."""
    g = torch.Generator().manual_seed(seed + (1_000_003 if split != "train" else 0))
    labels = torch.randint(n_c, (n,), generator=g)
    choices = [
        torch.randint(m, (n, s**level), generator=g) for level in range(num_layers)
    ]
    return labels, choices


def _expand(
    labels: torch.Tensor, choices: list[torch.Tensor], rules: list[torch.Tensor]
) -> torch.Tensor:
    """Run the grammar down the tree: ``(N,)`` labels -> ``(N, s**L)`` leaves."""
    x = labels.unsqueeze(1)
    for rule, choice in zip(rules, choices):
        x = rule[x, choice].flatten(start_dim=1)
    return x


class _RHMDataset(Dataset):
    """In-memory ``(features, label)`` pairs with a ``.classes`` attribute."""

    def __init__(
        self, features: torch.Tensor, labels: torch.Tensor, classes: list[int]
    ) -> None:
        self.features = features
        self.labels = labels
        self.classes = classes

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


@register("rhm")
def build_rhm(config: DatasetConfig) -> Dataset:
    """Build a Random Hierarchy Model split.

    Grammar hyperparameters come from ``config.params`` (see ``_DEFAULTS``);
    the class count from ``config.num_classes`` (default 2).  Features are
    one-hot ``(v, s**L)`` float32 (flattened when ``config.flatten``), labels
    are integer classes — use the ``"binary"`` preset or ``classes=[0, 1]``
    for {-1, +1} readout targets, or ``label_encoding="one_hot"`` for
    multi-class.
    """
    params = {**_DEFAULTS, **dict(config.params or {})}
    unknown = set(params) - set(_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown RHM params {sorted(unknown)}; use {sorted(_DEFAULTS)}"
        )
    v, m = params["num_features"], params["num_synonyms"]
    s, num_layers = params["tuple_size"], params["num_layers"]
    n_c = config.num_classes or 2
    if s < 2 or num_layers < 1 or m < 1 or v < 2:
        raise ValueError(f"Invalid RHM geometry: v={v}, m={m}, s={s}, L={num_layers}")

    rules = _sample_rules(v, n_c, m, s, num_layers, params["seed_rules"])
    max_data = n_c * m ** ((s**num_layers - 1) // (s - 1))

    if max_data > _INDEX_MAX:
        labels, choices = _generative_choices(
            config.split, config.n_samples, n_c, m, s, num_layers, config.seed
        )
    else:
        idx = _split_indices(
            config.split, config.n_samples, max_data, config.seed, params["test_pool"]
        )
        labels, choices = _decode_choices(idx, n_c, m, s, num_layers, max_data)

    leaves = _expand(labels, choices, rules)  # (N, s**L) symbols in [0, v)
    features = F.one_hot(leaves, num_classes=v).float()  # (N, d, v)
    if params["whitening"]:
        features = (features - 1.0 / v) * (1.0 - 1.0 / v) ** -0.5
    features = features.permute(0, 2, 1)  # (N, v, d) channels-first
    if config.flatten:
        features = features.flatten(start_dim=1)

    return _RHMDataset(features.contiguous(), labels, classes=list(range(n_c)))
