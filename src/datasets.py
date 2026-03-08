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
