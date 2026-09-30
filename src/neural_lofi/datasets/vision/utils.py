"""Per-dataset default transforms and empirical normalisation constants."""


import torch
from torchvision import transforms

# Per-dataset empirical channel statistics (computed on training sets).
_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)

_FASHION_MNIST_MEAN = (0.2860,)
_FASHION_MNIST_STD = (0.3530,)

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Official CINIC-10 channel statistics.
_CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
_CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)

_PCAM_MEAN = (0.7008, 0.5384, 0.6916)
_PCAM_STD = (0.2350, 0.2774, 0.2129)

_CELEBA_MEAN = (0.5063, 0.4258, 0.3832)
_CELEBA_STD = (0.3107, 0.2904, 0.2897)

# Standard ImageNet channel statistics, reused for the ImageNet-100 subset
# (the 100-class subset's per-channel stats are within ~1e-3 of full ImageNet,
# so the canonical constants are used by convention).
_IMAGENET100_MEAN = (0.485, 0.456, 0.406)
_IMAGENET100_STD = (0.229, 0.224, 0.225)


def _maybe_flatten(steps: list, *, flatten: bool) -> transforms.Compose:
    """Append a Flatten step when *flatten* is True, then compose."""
    if flatten:
        steps.append(torch.nn.Flatten(start_dim=0))
    return transforms.Compose(steps)


def default_mnist_transform(*, flatten: bool = False) -> transforms.Compose:
    """MNIST: 28×28 grayscale, normalised with empirical stats."""
    return _maybe_flatten(
        [
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
        ],
        flatten=flatten,
    )


def default_fashion_mnist_transform(*, flatten: bool = False) -> transforms.Compose:
    """Fashion-MNIST: 28×28 grayscale, normalised with empirical stats."""
    return _maybe_flatten(
        [
            transforms.ToTensor(),
            transforms.Normalize(_FASHION_MNIST_MEAN, _FASHION_MNIST_STD),
        ],
        flatten=flatten,
    )


def default_cifar10_transform(*, flatten: bool = False) -> transforms.Compose:
    """CIFAR-10: 32×32 RGB, normalised with empirical stats."""
    return _maybe_flatten(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ],
        flatten=flatten,
    )


def default_cinic10_transform(*, flatten: bool = False) -> transforms.Compose:
    """CINIC-10: 32×32 RGB, normalised with official CINIC-10 stats."""
    return _maybe_flatten(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CINIC10_MEAN, _CINIC10_STD),
        ],
        flatten=flatten,
    )


def default_pcam_transform(*, flatten: bool = False) -> transforms.Compose:
    """PCAM: 96×96 RGB, normalised with empirical stats."""
    return _maybe_flatten(
        [
            transforms.ToTensor(),
            transforms.Normalize(_PCAM_MEAN, _PCAM_STD),
        ],
        flatten=flatten,
    )


def default_celeba_transform(*, flatten: bool = False) -> transforms.Compose:
    """CelebA: center-cropped to 178×178, resized to 64×64 RGB, normalised."""
    return _maybe_flatten(
        [
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(_CELEBA_MEAN, _CELEBA_STD),
        ],
        flatten=flatten,
    )


def default_imagenet100_transform(*, flatten: bool = False) -> transforms.Compose:
    """ImageNet-100: variable-size RGB resized to 256 then center-cropped to
    224×224, normalised with the standard ImageNet stats (the canonical
    ImageNet preprocessing)."""
    return _maybe_flatten(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET100_MEAN, _IMAGENET100_STD),
        ],
        flatten=flatten,
    )
