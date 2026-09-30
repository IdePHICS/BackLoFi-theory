"""ImageNet-100 builder, registered as ``"imagenet-100"``.

ImageNet-100 is a 100-class subset of ILSVRC-2012 ImageNet (the standard subset
popularised by Tian et al., "Contrastive Multiview Coding") — 100 synsets of
variable-resolution RGB images.  It is *not* distributed via torchvision.
Download and extract it once, then point ``DatasetConfig.root`` at its parent
directory; the expected on-disk layout is a torchvision ``ImageFolder`` tree::

    <root>/imagenet-100/{train,val}/<wnid>/*.JPEG

There is no separate test split — the ``val`` folder is the held-out evaluation
set, so both ``split="val"`` and ``split="test"`` map to it.  ``ImageFolder``
assigns class indices in alphabetical (sorted-WNID) folder order; using the same
WNID folders for train and val keeps the indices aligned across splits.

The popular Kaggle distribution (``ambityga/imagenet100``) ships the training
set pre-sharded into ``train.X1`` … ``train.X4`` sub-directories — consolidate
those WNID folders into a single ``train/`` directory before pointing the
builder at it.
"""

from torch.utils.data import Dataset
from torchvision import datasets

from ..config import DatasetConfig
from ..factory import register
from .utils import default_imagenet100_transform

# DatasetConfig split -> on-disk ImageNet-100 directory name.  There is no
# dedicated test split, so ``test`` aliases the ``val`` evaluation folder.
_SPLIT_DIRS = {"train": "train", "val": "val", "test": "val"}


@register("imagenet-100")
def build_imagenet100(config: DatasetConfig) -> Dataset:
    """Build the ImageNet-100 dataset for ``config.split`` from an ImageFolder tree."""
    tfm = config.transform or default_imagenet100_transform(flatten=config.flatten)

    split_dir = _SPLIT_DIRS.get(config.split)
    if split_dir is None:
        raise ValueError(f"Unsupported split for ImageNet-100: {config.split!r}")

    root = config.root_path / "imagenet-100" / split_dir
    if not root.is_dir():
        raise FileNotFoundError(
            f"ImageNet-100 directory not found at {root}. ImageNet-100 is not "
            "auto-downloaded; download and extract it so that the layout is "
            f"{config.root_path / 'imagenet-100'}/{{train,val}}/<wnid>/*.JPEG "
            "(see e.g. the Kaggle 'ambityga/imagenet100' release)."
        )

    return datasets.ImageFolder(  # pragma: no cover - requires downloaded data
        root=str(root), transform=tfm
    )
