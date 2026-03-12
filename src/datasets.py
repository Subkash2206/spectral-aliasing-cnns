"""
datasets.py

Boring but necessary plumbing for loading real images in later phases.
The actual interesting stimuli for Phase 1 are in synthetic.py. This
file is for when we start running the spectral analysis on actual
ImageNet images to see whether the AVR numbers from resnet50 on real
data look anything like what we measured on synthetic inputs.

Important note: the ImageNet normalisation constants (mean/std) are not
arbitrary. They were computed from the full ImageNet training set and are
now a de-facto standard. Using different values would cause the pretrained
model to see out-of-distribution pixel intensities, which would corrupt
the activations we are trying to measure.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dsets


# These are the per-channel mean and std of the ImageNet training set.
# If you use a pretrained model you must use exactly these values --
# the model's internal statistics were calibrated against them.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_transform(
    image_size: int = 224,
    augment: bool = False,
) -> T.Compose:
    """
    Returns the standard eval or train preprocessing pipeline for ImageNet.
    The augment flag controls which one to return.

    For spectral analysis we almost always want augment=False because we
    are measuring properties of the model's representation of the image,
    not trying to train anything. Random crops and flips would change the
    spatial frequency content of the input in unpredictable ways, making
    the AVR measurements noisy and hard to interpret.

    The 1.14x resize before the centre crop is standard practice -- resize
    slightly larger than 224 so the centre crop is not blurring the image
    by upsampling. The ratio comes from 256/224 ~ 1.14 which is what the
    original ResNet papers used.

    Parameters
    image_size : int
        Target resolution for the shorter spatial dimension. Default 224
        matches what most ImageNet models were trained on.
    augment : bool
        True for a training pipeline with random crops and colour jitter.
        False (default) for a clean eval pipeline.

    Returns
    torchvision.transforms.Compose
    """
    if augment:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_imagefolder_loader(
    root: str,
    batch_size: int = 32,
    image_size: int = 224,
    augment: bool = False,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    """
    Builds a DataLoader that reads from an ImageFolder-structured directory
    (one sub-folder per class). This is the standard torchvision layout and
    matches what a typical ImageNet validation set download looks like.

    pin_memory is set automatically based on whether CUDA is available --
    it pre-pins the tensors in page-locked host memory so GPU transfer is
    faster. On CPU-only machines it does nothing.

    Parameters
    root : str
        Path to the top-level directory containing class sub-folders.
    batch_size : int
        Number of images per batch.
    image_size : int
        Passed to imagenet_transform.
    augment : bool
        True to use the training augmentation pipeline.
    num_workers : int
        Number of parallel data loading processes. 4 is a reasonable
        default for most machines; reduce to 0 if you get weird errors.
    shuffle : bool
        Whether to shuffle the dataset each epoch. For spectral analysis
        we usually want False so results are reproducible.

    Returns
    DataLoader
    """
    dataset = dsets.ImageFolder(
        root=root,
        transform=imagenet_transform(image_size=image_size, augment=augment),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_cifar10_loader(
    n_images: int = 1,
    batch_size: int = 1,
    image_size: int = 224,
    data_root: str = "data",
) -> DataLoader:
    """
    Returns a DataLoader over the first n_images of the CIFAR-10 test set,
    resized to image_size x image_size and normalised with ImageNet stats.

    Why CIFAR-10 and not ImageNet? Because CIFAR-10 is freely downloadable
    (only ~170 MB) while ImageNet requires a manual licence. The images are
    32x32 originally, so we upscale to 224 -- the quality is low but the
    spectral structure is real and sufficient for a sanity check on the hook
    and AVR pipeline.

    Why ImageNet normalisation? Because we are feeding these images into a
    pretrained ResNet50 that was trained with those exact mean/std values.
    Using the wrong normalisation would shift the pixel distribution and the
    model's activations would look out-of-distribution.

    Parameters
    n_images : int
        How many images to include. Wraps a Subset so you do not have to
        iterate over the whole test set (10k images) to get one sample.
    batch_size : int
        Batch size for the DataLoader.
    image_size : int
        Target resolution after resize. Default 224 matches ResNet50 input.
    data_root : str
        Directory where CIFAR-10 will be downloaded if not already present.

    Returns
    DataLoader
    """
    transform = T.Compose([
        # Upsample the tiny 32x32 CIFAR images to 224 so ResNet50 sees the
        # expected receptive field sizes. Bilinear is used by default.
        T.Resize(image_size),
        T.ToTensor(),
        # Apply ImageNet normalisation so the pretrained model is not confused
        # by unusually bright or dark pixel values.
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = dsets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )
    # Take only the first n_images to keep things fast in Phase 1.
    subset = torch.utils.data.Subset(dataset, list(range(n_images)))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


def get_stl10_loader(
    n_images: int = 1000,
    batch_size: int = 32,
    image_size: int = 224,
    data_root: str = "data",
) -> DataLoader:
    """
    Returns a DataLoader over the first n_images of the STL10 test set,
    resized to image_size x image_size and normalised with ImageNet stats.

    STL10 images are 96x96 -- far less destructive to upsample to 224
    than CIFAR-10's 32x32. Bilinear upsampling from 96 to 224 preserves
    genuine mid-to-high frequency content that CIFAR-10 upsampling destroys.
    This makes STL10 a much better proxy for ImageNet for spectral analysis.

    Parameters
    n_images : int
        How many images to use from the test split (13,000 available).
    batch_size : int
        Batch size for the DataLoader.
    image_size : int
        Target resolution after resize. Default 224.
    data_root : str
        Directory where STL10 will be downloaded (~2.5GB).

    Returns
    DataLoader
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = dsets.STL10(
        root=data_root,
        split='test',
        download=True,
        transform=transform,
    )
    subset = torch.utils.data.Subset(dataset, list(range(n_images)))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

