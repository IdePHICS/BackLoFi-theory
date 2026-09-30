"""CINIC-10 builder, registered as ``"cinic10"``.

CINIC-10 is a CIFAR-10-compatible benchmark (10 classes, 32×32 RGB, 270k
images) that is *not* distributed via torchvision.  Download and extract the
archive once, then point ``DatasetConfig.root`` at its parent directory; the
expected on-disk layout is::

    <root>/cinic-10/{train,valid,test}/<class_name>/*.png

ImageFolder assigns class indices in alphabetical folder order, which matches
CIFAR-10's canonical class ordering, so class indices line up with CIFAR-10.
"""


from torch.utils.data import Dataset
from torchvision import datasets

from ..config import DatasetConfig
from ..factory import register
from .utils import default_cinic10_transform

# DatasetConfig split -> on-disk CINIC-10 directory name.
_SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


@register("cinic10")
def build_cinic10(config: DatasetConfig) -> Dataset:
    """Build the CINIC-10 dataset for ``config.split`` from an ImageFolder tree."""
    tfm = config.transform or default_cinic10_transform(flatten=config.flatten)

    split_dir = _SPLIT_DIRS.get(config.split)
    if split_dir is None:
        raise ValueError(f"Unsupported split for CINIC-10: {config.split!r}")

    root = config.root_path / "cinic-10" / split_dir
    if not root.is_dir():
        raise FileNotFoundError(
            f"CINIC-10 directory not found at {root}. CINIC-10 is not "
            "auto-downloaded; download and extract it so that the layout is "
            f"{config.root_path / 'cinic-10'}/{{train,valid,test}}/<class>/*.png "
            "(see https://github.com/BayesWatch/cinic-10)."
        )

    return datasets.ImageFolder(  # pragma: no cover - requires downloaded data
        root=str(root), transform=tfm
    )
