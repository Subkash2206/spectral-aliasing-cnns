"""
metrics.py

The three functions in spectral.py work on a single 2-D feature map.
In reality a CNN produces batches of multi-channel tensors, so we need
glue code that iterates over (batch, channel) pairs and aggregates the
individual AVR values into something useful for a whole layer or a whole
network.

That is all this file does -- no new math, just plumbing. The reason to
keep it separate from spectral.py is that spectral.py should stay pure
and easy to test; this file is allowed to be messy about loops and dicts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from src.spectral import compute_avr, compute_power_spectrum, compute_radial_profile


def avr_for_feature_map(
    feature_map: torch.Tensor,
    stride: int,
) -> float:
    """
    Runs the full three-step spectral pipeline on a single 2-D slice and
    returns one AVR number. Exists mostly for clarity in avr_for_batch --
    pulling the three-step chain into its own function makes the nested
    loop readable.

    If you skipped this wrapper and inlined the three calls everywhere,
    you would end up copy-pasting the same three lines repeatedly, which
    makes it easy to accidentally call them in the wrong order or forget
    one step.

    Parameters
    feature_map : torch.Tensor, shape (H, W)
        Single-channel spatial activation from one sample.
    stride : int
        Stride of the convolution that follows this layer -- we measure
        how vulnerable this feature map is to *that* operation.

    Returns
    float
        AVR in [0, 1].
    """
    ps = compute_power_spectrum(feature_map)
    return compute_avr(ps, stride)


def avr_for_batch(
    feature_maps: torch.Tensor,
    stride: int,
) -> float:
    """
    Computes AVR independently for every (sample, channel) pair in a
    batch and returns the mean. The mean is a reasonable summary statistic
    because we want a single number per layer -- variance across channels
    is an interesting second-order measure but we are not reporting it yet.

    Each channel of each sample is a separate 2-D spatial signal and
    deserves its own power spectrum. Averaging the raw tensors before
    computing AVR would completely destroy the spectral information.

    Parameters
    feature_maps : torch.Tensor, shape (B, C, H, W)
        Batch of feature maps from a single layer, as captured by the hook.
    stride : int
        Stride of the subsequent strided convolution.

    Returns
    float
        Mean AVR across all B*C individual channel maps.
    """
    if feature_maps.dim() != 4:
        raise ValueError(
            f"Expected shape (B, C, H, W), got {tuple(feature_maps.shape)}"
        )

    avrs: List[float] = []
    B, C, H, W = feature_maps.shape
    for b in range(B):
        for c in range(C):
            avrs.append(avr_for_feature_map(feature_maps[b, c], stride))
    return float(np.mean(avrs))


def avr_per_layer(
    layer_features: Dict[str, torch.Tensor],
    layer_strides: Dict[str, int],
) -> Dict[str, float]:
    """
    Given the full dict of captured feature maps (one entry per hooked
    layer) and a matching dict of strides, returns a dict of AVR values
    keyed by layer name. That output is what we plot and compare across
    models.

    The stride dict has to be provided separately because PyTorch hooks
    give you the output tensor of a layer, but not the stride of the
    *next* layer. The caller has to look that up from the model definition
    (see find_strided_layers in models.py) and pass it in explicitly.

    If a layer is missing from layer_strides we raise immediately rather
    than silently producing an AVR using the wrong stride number.

    Parameters
    layer_features : dict[str, Tensor]
        Layer name -> captured feature map batch (B, C, H, W).
    layer_strides : dict[str, int]
        Layer name -> stride of the operation that follows it.

    Returns
    dict[str, float]
        Layer name -> mean AVR for that layer.

    Raises
    KeyError
        If any key in layer_features has no corresponding entry in
        layer_strides. A silent wrong-stride answer would be worse than
        a crash here.
    """
    results: Dict[str, float] = {}
    for name, feat in layer_features.items():
        if name not in layer_strides:
            raise KeyError(
                f"No stride specified for layer '{name}'. "
                "Please update layer_strides dict."
            )
        results[name] = avr_for_batch(feat, layer_strides[name])
    return results
