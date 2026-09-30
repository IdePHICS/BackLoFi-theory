"""Dataset registry, ``register`` decorator, and the ``build_dataset`` entry point."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .config import DatasetConfig
from .wrappers import (
    filter_classes,
    input_as_target,
    limit_dataset,
    one_hot_targets,
)

Builder = Callable[[DatasetConfig], Dataset]
_REGISTRY: dict[str, Builder] = {}


def register(dataset_type: str):
    """
    Decorator to register dataset builders.
    """

    def _wrap(fn: Builder) -> Builder:
        key = dataset_type.lower().strip()
        if key in _REGISTRY:
            raise KeyError(f"Dataset builder already registered for {key!r}")
        _REGISTRY[key] = fn
        return fn

    return _wrap


def _infer_num_classes(ds: Dataset) -> int:
    """Read the class count from a dataset's ``.classes`` attribute.

    Torchvision MNIST / FashionMNIST / CIFAR10 and ``ImageFolder`` (CINIC-10)
    all expose ``.classes``.  Raises ``ValueError`` when it is unavailable so
    the caller knows to pass ``num_classes`` explicitly.
    """
    classes = getattr(ds, "classes", None)
    if classes is None:
        raise ValueError(
            "Cannot infer the number of classes for one-hot encoding: the "
            "dataset has no '.classes' attribute. Set 'num_classes' explicitly "
            "on DatasetConfig."
        )
    return len(classes)


def _build_raw_dataset(config: DatasetConfig) -> Dataset:
    """Build the raw dataset for ``config.split``, concatenating ``+``-joined splits.

    A single split delegates straight to the registered builder.  A combined
    split (e.g. ``"train+val"``) builds each part and wraps them in a
    ``ConcatDataset``, preserving the ``.classes`` attribute (when present) so
    one-hot inference / preset filtering still works downstream.
    """
    builder = _REGISTRY[config.dataset_type]
    parts = config.split.split("+")
    if len(parts) == 1:
        return builder(config)

    # Builders read ``config.split``; drive each part in turn, then restore.
    saved_split = config.split
    subsets: list[Dataset] = []
    try:
        for part in parts:
            config.split = part
            subsets.append(builder(config))
    finally:
        config.split = saved_split

    combined = ConcatDataset(subsets)
    classes = getattr(subsets[0], "classes", None)
    if classes is not None:
        combined.classes = classes  # ConcatDataset has no slots; attribute sticks
    return combined


def build_dataset(config: DatasetConfig) -> Dataset:
    """
    Main entry point: returns a ``torch.utils.data.Dataset``.

    Processing order:
      1. Build the raw dataset via its registered builder (``+``-joined splits
         such as ``"train+val"`` are built per-part and concatenated).
      2. If ``config.target_mode == "input"``: replace the target with the input
         (``(x, x)``) and skip all label encoding.
      3. Otherwise encode targets per ``config.label_encoding``:
         - ``"binary_pm1"``: filter to ``config.classes`` (if set), scalar labels.
         - one-hot modes: filter to ``config.classes`` (if set, remapped to
           ``0…K-1``) then emit one-hot vectors of dimension ``K``.
      4. Sub-sample to ``config.n_samples``.

    Each sample is a ``(input, target)`` tuple following standard PyTorch
    conventions.
    """
    key = config.dataset_type
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset_type {key!r}. "
            f"Registered types: {sorted(_REGISTRY.keys())}"
        )

    ds = _build_raw_dataset(config)

    # --- self-supervised target: (x, y) -> (x, x), no label encoding ---
    if config.target_mode == "input":
        ds = input_as_target(ds)
        return limit_dataset(ds, config.n_samples, seed=config.seed, shuffle=True)

    # --- target encoding (before n_samples limit) ---
    if config.label_encoding != "binary_pm1":
        centered = config.label_encoding == "one_hot_centered"
        if config.classes is not None:
            # Remap the selected subset to consecutive 0…K-1, then one-hot it.
            ds = filter_classes(ds, config.classes, remap=True, label_map=None)
            k = len(set(config.classes))
        else:
            k = config.num_classes or _infer_num_classes(ds)
        ds = one_hot_targets(ds, k, centered=centered)
    elif config.classes is not None:
        ds = filter_classes(
            ds, config.classes, remap=config.remap_labels, label_map=config.label_map
        )

    # --- sub-sample ---
    ds = limit_dataset(ds, config.n_samples, seed=config.seed, shuffle=True)

    return ds


def load_tensors(
    *,
    dataset_type: str,
    split: str,
    n_samples: int,
    device: torch.device | str,
    seed: int = 0,
    root: str | Path = "./data",
    flatten: bool = True,
    class_preset: str | None = None,
    classes: list[int] | None = None,
    remap_labels: bool = True,
    label_encoding: str = "binary_pm1",
    num_classes: int | None = None,
    batch_size: int = 512,
    max_seq_length: int | None = None,
    target_mode: str = "label",
    x_dtype: torch.dtype | None = None,
    y_dtype: torch.dtype | None = None,
    transform: Any | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a dataset as ``(X, y)`` tensors on *device*.

    Convenience wrapper around :func:`build_dataset` that materialises
    the full dataset into two tensors.

    Dtype handling is **dtype-aware**, not dtype-forcing: by default ``X``
    keeps the dataset's native dtype (floating inputs are normalised to
    ``float32``; **integer token ids stay ``long``**) and ``y`` is cast to
    ``float32`` (the ridge-readout convention).  Pass ``x_dtype`` / ``y_dtype``
    to override either — e.g. ``y_dtype=torch.long`` for self-supervised token
    targets under ``target_mode="input"``.

    Parameters
    ----------
    dataset_type : str
        Registered dataset name (e.g. ``"mnist"``, ``"cifar10"``).
    split : str
        ``"train"``, ``"val"``, or ``"test"`` — or several joined with ``+``
        (e.g. ``"train+val"``) to concatenate them into one pool.
    n_samples : int
        Number of samples to load.
    device : torch.device | str
        Target device for the returned tensors.
    seed : int
        Random seed for sub-sampling.
    root : str | Path
        Root data directory.
    flatten : bool
        Whether to flatten spatial dims (default ``True``).
    class_preset : str | None
        Named class-grouping preset.
    classes : list[int] | None
        Explicit class filter.
    remap_labels : bool
        Whether to remap labels (default ``True``).
    label_encoding : str
        Target encoding: ``"binary_pm1"`` (default, scalar) or one of the
        one-hot variants (``"one_hot"``, ``"one_hot_centered"``).
    num_classes : int | None
        One-hot dimension override when ``classes`` is ``None`` (see
        :class:`DatasetConfig`).
    batch_size : int
        Batch size for the internal DataLoader (default 512).
    target_mode : str
        ``"label"`` (default) or ``"input"`` (self-supervised ``(x, x)``); see
        :class:`DatasetConfig`.
    x_dtype : torch.dtype | None
        Explicit dtype for ``X``.  ``None`` (default) preserves the native dtype
        (floating inputs → ``float32``, integer token ids stay ``long``).
    y_dtype : torch.dtype | None
        Explicit dtype for ``y``.  ``None`` (default) casts to ``float32``.
    params : dict[str, Any] | None
        Builder-specific extra parameters (e.g. the RHM grammar); see
        :class:`DatasetConfig`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(X, y)``.  By default ``y`` is ``float32`` (shape ``(N, K)`` in the
        one-hot modes, else ``(N,)``); ``X`` keeps its native dtype unless
        ``x_dtype`` is given.
    """
    ds_cfg = DatasetConfig(
        dataset_type=dataset_type,
        split=split,
        root=root,
        n_samples=n_samples,
        seed=seed,
        flatten=flatten,
        max_seq_length=max_seq_length,
        class_preset=class_preset,
        classes=classes,
        remap_labels=remap_labels,
        label_encoding=label_encoding,
        num_classes=num_classes,
        target_mode=target_mode,
        transform=transform,
        params=params,
    )
    ds = build_dataset(ds_cfg)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    xs, ys = zip(*list(loader))

    x = torch.cat(xs, dim=0)
    if x_dtype is not None:
        x = x.to(x_dtype)
    elif x.is_floating_point():
        x = x.float()
    # else: integer inputs (e.g. token ids) keep their native dtype.

    y = torch.cat(ys, dim=0)
    y = y.to(y_dtype) if y_dtype is not None else y.float()

    return x.to(device), y.to(device)


def available_datasets() -> list[str]:
    """Return sorted list of registered dataset type names."""
    return sorted(_REGISTRY.keys())
