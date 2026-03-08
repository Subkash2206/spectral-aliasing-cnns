"""
models.py

Loading and inspecting CNN models. The two main things I need here are:
  1. A way to get a standard model (from timm) and an anti-aliased version
     (from antialiased_cnns) so I can compare their AVR profiles.
  2. A way to automatically find which layers in any given model actually
     do strided convolutions -- those are the spots where aliasing can occur
     and where I need to measure AVR just before.

The antialiased_cnns package is Richard Zhang's implementation of the
BlurPool idea: replace stride-2 convolutions with a low-pass blur followed
by stride-2 subsampling. The blur is supposed to attenuate high frequencies
before the Nyquist violation happens. Whether it actually reduces AVR in
practice is exactly what this experiment is trying to measure.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm


def get_model(
    name: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 1000,
) -> nn.Module:
    """
    Loads a model from the timm registry and puts it in eval mode.
    We always put it in eval mode because we are only running inference
    -- having dropout or batch norm in training mode would add random
    noise to the feature maps and mess up the spectral measurements.

    Parameters
    name : str
        timm model identifier, e.g. 'resnet50', 'vgg16', 'efficientnet_b0'.
        The full list is at timm.list_models().
    pretrained : bool
        Whether to load ImageNet-pretrained weights. For measuring aliasing
        in real-world activations we want pretrained. Only skip this for
        quick structural checks.
    num_classes : int
        Number of output classes (1000 = ImageNet default).

    Returns
    nn.Module
        Model in eval mode on CPU.
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    model.eval()
    return model


def get_antialiased_model(name: str = "resnet50", pretrained: bool = False) -> nn.Module:
    """
    Loads the BlurPool variant of a ResNet from the antialiased_cnns package.
    The idea behind anti-aliased CNNs is to insert a learnable low-pass filter
    before every strided operation so that high-frequency content is attenuated
    before the stride throws away samples. If AVR(antialiased) < AVR(standard)
    then the blur is doing what it is supposed to.

    The getattr lookup lets us support different architectures from the same
    package without a big if-elif chain. If the requested name is not in the
    package we raise a clear error rather than returning None silently.

    Parameters
    name : str
        Architecture name as exposed by antialiased_cnns, e.g. 'resnet50'.
    pretrained : bool
        Whether to load pre-trained weights if the package provides them.

    Returns
    nn.Module
        Anti-aliased model in eval mode on CPU.
    """
    import antialiased_cnns  # type: ignore

    model_fn = getattr(antialiased_cnns, name, None)
    if model_fn is None:
        raise ValueError(
            f"antialiased_cnns does not expose model '{name}'. "
            "Try 'resnet50', 'resnet18', etc."
        )
    model = model_fn(pretrained=pretrained)
    model.eval()
    return model


def find_strided_layers(model: nn.Module) -> List[Tuple[str, nn.Module, int]]:
    """
    Walks every module in the model and collects the Conv2d layers that
    have stride > 1. These are exactly the layers where Nyquist violations
    can happen -- the feature map going *into* such a layer needs to have
    its AVR measured.

    The stride is stored as a tuple (stride_h, stride_w) by PyTorch but
    sometimes as a plain integer. Taking max(sy, sx) handles both cases and
    picks up any layer that is strided in at least one spatial dimension.

    Parameters
    model : nn.Module
        Any PyTorch model. Walks the full module tree in depth-first order.

    Returns
    list of (str, nn.Module, int)
        Each entry is (layer_name, module, stride). Ordered the same way
        as named_modules(), which is depth-first left-to-right.
    """
    strided: List[Tuple[str, nn.Module, int]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            sy, sx = (
                module.stride
                if isinstance(module.stride, tuple)
                else (module.stride, module.stride)
            )
            s = max(sy, sx)
            if s > 1:
                strided.append((name, module, s))
    return strided


def build_layer_stride_map(model: nn.Module) -> Dict[str, int]:
    """
    Condenses find_strided_layers into a plain dict for easy lookup.
    I use this to build the layer_strides argument for avr_per_layer
    without having to unpack tuples everywhere in the experiment code.

    Parameters
    model : nn.Module

    Returns
    dict[str, int]
        Maps each strided Conv2d layer name to its stride value.
    """
    return {name: stride for name, _mod, stride in find_strided_layers(model)}
