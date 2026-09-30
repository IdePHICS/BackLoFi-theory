"""Sub-sampling and class-filtering wrappers used by ``build_dataset``."""


from collections.abc import Sized
from typing import cast

import torch
from torch.utils.data import Dataset, Subset

# ------------------------------------------------------------------
# Size Liminting
# ------------------------------------------------------------------


def limit_dataset(
    ds: Dataset, n: int, *, seed: int = 0, shuffle: bool = True
) -> Dataset:
    """Sub-sample *ds* to at most *n* elements."""
    if n <= 0:
        return ds

    total = len(cast(Sized, ds))
    if n >= total:
        return ds

    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(total, generator=g)[:n].tolist()
    else:
        idx = list(range(n))

    return Subset(ds, idx)


# ------------------------------------------------------------------
# Class filtering
# ------------------------------------------------------------------


class _MappedSubset(Dataset):
    """Subset that remaps targets via ``label_map``."""

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        label_map: dict[int, int] | dict[int, int | float],
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map

    def __getitem__(self, idx: int) -> tuple:
        x, y = self.dataset[self.indices[idx]]
        return x, self.label_map[int(y)]

    def __len__(self) -> int:
        return len(self.indices)


def filter_classes(
    ds: Dataset,
    classes: list[int],
    *,
    remap: bool = True,
    label_map: dict[int, int] | dict[int, int | float] | None = None,
) -> Dataset:
    """Keep only samples whose target is in *classes* and optionally relabel them.

    Parameters
    ----------
    ds : Dataset
        A PyTorch dataset returning ``(input, target)`` tuples.
    classes : list[int]
        Class indices to retain.
    remap : bool
        If ``True`` (default) and ``label_map`` is not provided, relabel kept
        classes to consecutive integers ``0 … len(classes) - 1``
        (sorted by original label).
    label_map : dict[int, int | float] | None
        Explicit mapping for target relabelling
        (e.g. ``{3: -1.0, 7: 1.0}``).  If provided, it takes
        precedence over ``remap`` and is applied to the kept classes.

    Returns
    -------
    Dataset
        Filtered dataset, optionally relabelled.

    Notes
    -----
    - Returned targets are scalar labels (not one-hot vectors).
    - If both ``remap`` and ``label_map`` are set, ``label_map`` is used.
    """
    keep = set(classes)
    indices: list[int] = []

    for i in range(len(cast(Sized, ds))):
        _, y = ds[i]
        if int(y) in keep:
            indices.append(i)

    if not indices:
        raise ValueError(
            f"No samples found for classes {sorted(keep)}. "
            "Check that the class indices match the dataset."
        )

    if label_map is None and not remap:
        return Subset(ds, indices)

    if label_map is not None:
        missing = keep.difference(label_map.keys())
        if missing:
            raise ValueError(
                f"label_map is missing mappings for classes {sorted(missing)}. "
                f"Provided keys: {sorted(label_map.keys())}"
            )
        return _MappedSubset(ds, indices, label_map=label_map)

    remap_map = {c: new for new, c in enumerate(sorted(keep))}
    return _MappedSubset(ds, indices, label_map=remap_map)


# ------------------------------------------------------------------
# One-hot encoding
# ------------------------------------------------------------------


class _OneHotDataset(Dataset):
    """Wrap a dataset, emitting one-hot ``float32`` targets.

    The wrapped dataset must return integer targets in ``[0, num_classes)``.
    """

    def __init__(
        self, dataset: Dataset, num_classes: int, *, centered: bool = False
    ) -> None:
        self.dataset = dataset
        self.num_classes = num_classes
        self.centered = centered

    def __getitem__(self, idx: int) -> tuple:
        x, y = self.dataset[idx]
        cls = int(y)
        if not 0 <= cls < self.num_classes:
            raise ValueError(
                f"target {cls} out of range for num_classes={self.num_classes}"
            )
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        vec[cls] = 1.0
        if self.centered:
            vec -= 1.0 / self.num_classes
        return x, vec

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))


def one_hot_targets(
    ds: Dataset, num_classes: int, *, centered: bool = False
) -> Dataset:
    """Wrap *ds* so its integer targets become one-hot ``float32`` vectors.

    Parameters
    ----------
    ds : Dataset
        A dataset returning ``(input, target)`` with integer targets in
        ``[0, num_classes)``.
    num_classes : int
        Length of the emitted one-hot vector.
    centered : bool
        If ``True``, subtract ``1 / num_classes`` from every entry so each
        vector sums to zero per sample.

    Returns
    -------
    Dataset
        Dataset emitting ``(input, one_hot_vector)`` tuples.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    return _OneHotDataset(ds, num_classes, centered=centered)


# ------------------------------------------------------------------
# Self-supervised target (target_mode="input")
# ------------------------------------------------------------------


class _InputAsTargetDataset(Dataset):
    """Wrap a dataset so each sample's target is replaced by its input.

    Implements the ``target_mode="input"`` contract: ``(x, y) -> (x, x)``.  The
    original target is discarded, making any dataset self-supervised.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, idx: int) -> tuple:
        x, _ = self.dataset[idx]
        return x, x

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))


def input_as_target(ds: Dataset) -> Dataset:
    """Wrap *ds* so it returns ``(x, x)`` — the self-supervised target contract.

    Parameters
    ----------
    ds : Dataset
        A dataset returning ``(input, target)`` tuples; the target is dropped.

    Returns
    -------
    Dataset
        Dataset emitting ``(input, input)`` tuples.
    """
    return _InputAsTargetDataset(ds)
