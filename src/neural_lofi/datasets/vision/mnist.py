"""MNIST builder, registered as ``"mnist"``."""


from torch.utils.data import Dataset
from torchvision import datasets

from ..config import DatasetConfig
from ..factory import register
from .utils import default_mnist_transform


@register("mnist")
def build_mnist(config: DatasetConfig) -> Dataset:
    """Build the torchvision MNIST dataset honouring ``config.split`` and ``config.flatten``."""
    tfm = config.transform or default_mnist_transform(flatten=config.flatten)

    if config.split == "train":
        train = True
    elif config.split in {"val", "test"}:
        train = False
    else:
        raise ValueError(f"Unsupported split for MNIST: {config.split!r}")

    ds = datasets.MNIST(
        root=str(config.root_path),
        train=train,
        download=True,
        transform=tfm,
    )

    return ds
