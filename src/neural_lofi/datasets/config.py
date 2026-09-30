"""Dataset configuration dataclass shared by every dataset builder."""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torchvision import transforms

_VALID_TYPES = {
    "mnist",
    "rhm",
    "fashion-mnist",
    "cifar10",
    "cinic10",
    "imagenet-100",
    "pcam",
    "celeba",
    "amazon-polarity",
    "tiny-stories",
    "wikitext",
}

_VALID_SPLITS = {"train", "val", "test"}

#: Target encodings.  ``"binary_pm1"`` keeps the scalar {-1, +1} default;
#: the one-hot variants emit a float32 vector of length = number of classes.
_VALID_LABEL_ENCODINGS = {"binary_pm1", "one_hot", "one_hot_centered"}

#: Target-slot modes.  ``"label"`` (default) returns the dataset's supervised
#: target; ``"input"`` returns the input itself as the target — the
#: self-supervised ``(x, x)`` contract used by the unsupervised text datasets,
#: which any *text* builder (e.g. ``amazon-polarity``) may opt into.
_VALID_TARGET_MODES = {"label", "input"}


@dataclass(slots=True)
class DatasetConfig:
    """
    Unified configuration for dataset creation.

    dataset_type:
      One of the registered vision dataset names (e.g. "mnist", "cifar10", "celeba", …).

    split: "train" | "val" | "test"
      Used to select the appropriate dataset split.  Several splits may be
      joined with ``+`` (e.g. ``"train+val"``) to concatenate them into a single
      pool — handy when a dataset's val/test splits are large and you want more
      training data (e.g. CINIC-10).  Only meaningful where the splits are
      genuinely distinct; on datasets that alias ``val`` to ``test`` (mnist,
      fashion-mnist, cifar10, amazon-polarity) combining them double-counts.

    class_preset : str | None
      Named grouping shortcut (e.g. ``"even_odd"``, ``"upper_lower"``).
      When set, populates ``classes`` and ``label_map`` automatically.
      Cannot be combined with explicit ``classes`` or ``label_map``.
      See :mod:`neural_lofi.datasets.presets` for available names.

    classes : list[int] | None
      If given, only samples whose target is in this list are kept.
      Applied **before** the ``n_samples`` limit.

    remap_labels : bool
      When ``classes`` is set and ``remap_labels`` is True (default),
      the original class indices are remapped to consecutive integers
      ``0 … len(classes) - 1`` (sorted order).

    label_encoding : str
      Output target format, standardized across datasets:

      - ``"binary_pm1"`` (default): scalar {-1, +1} label (legacy behaviour).
      - ``"one_hot"``: ``float32`` one-hot vector with ``1.0`` at the true
        class and ``0.0`` elsewhere.
      - ``"one_hot_centered"``: the same vector minus ``1/K`` so it sums to
        zero per sample.

      The one-hot vector length ``K`` is the number of classes in play:
      ``len(classes)`` when ``classes`` is set, otherwise the dataset's full
      natural class count (or ``num_classes`` if given).  The one-hot
      encodings are incompatible with ``class_preset`` / ``label_map`` (both
      imply a scalar mapping).

    num_classes : int | None
      Optional override for the one-hot dimension when ``classes`` is ``None``.
      If omitted, it is inferred from the raw dataset's ``.classes`` attribute.

    target_mode : str
      Which value occupies the target slot:

      - ``"label"`` (default): the dataset's supervised target.
      - ``"input"``: the input itself, i.e. the self-supervised ``(x, x)``
        contract.  For text builders (e.g. ``amazon-polarity``) this makes the
        same corpus usable without labels.  Incompatible with class filtering
        / one-hot encoding (``classes``, ``class_preset``, ``label_map``, and
        non-``binary_pm1`` ``label_encoding``), which all imply a label.

      The unsupervised text datasets (``tiny-stories``, ``wikitext``) carry no
      label and always behave as ``"input"`` regardless of this setting.

    Each returned sample is an ``(input, target)`` tuple following standard
    PyTorch Dataset conventions.
    """

    dataset_type: str
    split: str = "train"
    root: str | Path = "./data"

    n_samples: int = 20_000
    seed: int = 0
    flatten: bool = False
    max_seq_length: int | None = None

    class_preset: str | None = None
    classes: list[int] | None = None
    remap_labels: bool = True
    label_map: dict[int, int | float] | None = None

    label_encoding: str = "binary_pm1"
    num_classes: int | None = None

    target_mode: str = "label"

    transform: transforms.Compose | None = None

    #: Extra builder-specific parameters (synthetic datasets — e.g. the RHM
    #: grammar: num_features, num_synonyms, tuple_size, num_layers, seed_rules).
    #: Ignored by builders that take none; unknown keys raise in the builder.
    params: dict[str, Any] | None = None

    root_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_type = self.dataset_type.lower().strip()
        if self.dataset_type not in _VALID_TYPES:
            raise ValueError(
                f"dataset_type must be one of {sorted(_VALID_TYPES)}, "
                f"got {self.dataset_type!r}"
            )

        # ``split`` may be a single split or several joined by ``+`` (e.g.
        # "train+val") to concatenate them into one pool — useful when a
        # dataset's val/test splits are large and you want more training data.
        self.split = self.split.lower().strip()
        split_parts = [p.strip() for p in self.split.split("+")]
        if not all(p in _VALID_SPLITS for p in split_parts):
            raise ValueError(
                f"split must be one of {sorted(_VALID_SPLITS)}, optionally several "
                f"joined by '+' (e.g. 'train+val'); got {self.split!r}"
            )
        self.split = "+".join(split_parts)

        self.label_encoding = self.label_encoding.lower().strip()
        self.target_mode = self.target_mode.lower().strip()

        self.root_path = Path(self.root).expanduser()

        self._resolve_class_preset()
        self._maybe_autoset_binary_label_map()

        self.validate()

    def _resolve_class_preset(self) -> None:
        """Populate ``classes`` and ``label_map`` from a named preset.

        Raises ``ValueError`` if ``class_preset`` is set alongside explicit
        ``classes`` or ``label_map``, as those combinations are ambiguous.
        """
        if self.class_preset is None:
            return
        if self.classes is not None:
            raise ValueError(
                "Cannot set both 'class_preset' and 'classes'. "
                "Use 'class_preset' alone or specify 'classes' directly."
            )
        if self.label_map is not None:
            raise ValueError(
                "Cannot set both 'class_preset' and 'label_map'. "
                "Use 'class_preset' alone or specify 'label_map' directly."
            )
        from .presets import resolve_preset  # local import to avoid circular deps

        preset = resolve_preset(self.dataset_type, self.class_preset)
        self.classes = list(preset.classes)
        self.label_map = preset.label_map

    def _maybe_autoset_binary_label_map(self) -> None:
        """If binary class filtering is requested, auto-map to {-1, +1}.

        Mapping rule (deterministic): sorted(classes)[0] -> -1, sorted(classes)[1] -> +1
        """
        if self.label_encoding != "binary_pm1":
            return  # one-hot modes never produce a scalar ±1 mapping
        if not self.remap_labels:
            return
        if self.label_map is not None:
            return
        if self.classes is None:
            return

        unique_classes = sorted(set(self.classes))
        if len(unique_classes) == 2:
            self.label_map = {
                unique_classes[0]: -1.0,
                unique_classes[1]: +1.0,
            }

    def validate(self) -> None:
        """Sanity-check fields after ``__post_init__`` resolves presets."""
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if self.max_seq_length is not None and self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be > 0 when set")
        if self.label_encoding not in _VALID_LABEL_ENCODINGS:
            raise ValueError(
                f"label_encoding must be one of {sorted(_VALID_LABEL_ENCODINGS)}, "
                f"got {self.label_encoding!r}"
            )
        if self.target_mode not in _VALID_TARGET_MODES:
            raise ValueError(
                f"target_mode must be one of {sorted(_VALID_TARGET_MODES)}, "
                f"got {self.target_mode!r}"
            )
        if self.target_mode == "input" and (
            self.classes is not None
            or self.class_preset is not None
            or self.label_map is not None
            or self.label_encoding != "binary_pm1"
        ):
            raise ValueError(
                "target_mode='input' is self-supervised and carries no label, so "
                "it is incompatible with 'classes', 'class_preset', 'label_map', "
                "and non-'binary_pm1' label_encoding."
            )
        if self.label_encoding != "binary_pm1":
            if self.class_preset is not None or self.label_map is not None:
                raise ValueError(
                    "One-hot label_encoding is incompatible with 'class_preset' "
                    "and 'label_map' (both imply a scalar ±1 mapping). "
                    "Use plain 'classes' to select a subset, or leave it unset "
                    "to use all natural classes."
                )
        if self.num_classes is not None and self.num_classes < 2:
            raise ValueError("num_classes must be >= 2 when set")
        if self.classes is not None and len(self.classes) == 0:
            raise ValueError("classes must be None or a non-empty list")
        if self.label_map is not None and len(self.label_map) == 0:
            raise ValueError("label_map must be None or a non-empty dict")
        if self.label_map is not None and self.classes is not None:
            class_set = set(self.classes)
            map_keys = set(self.label_map.keys())
            if map_keys != class_set:
                raise ValueError(
                    "label_map keys must match exactly the selected classes. "
                    f"classes={sorted(class_set)}, label_map keys={sorted(map_keys)}"
                )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"DatasetConfig(type={self.dataset_type!r}, split={self.split!r}, "
            f"n_samples={self.n_samples}, root={str(self.root_path)!r})"
        )
