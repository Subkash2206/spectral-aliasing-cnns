"""
hooks.py

The whole spectral analysis pipeline works on feature maps -- the
intermediate tensors a CNN produces inside its layers. But PyTorch's
forward pass does not hand those to you by default: you get the final
output and nothing else. Forward hooks are the standard way to intercept
and save whatever you want from inside the network without modifying the
model's source code.

The big gotcha is that naive hook code often silently fails if you
mistype a layer name -- you just get an empty dict and no error, which
is a nightmare to debug. FeatureExtractor is strict by design: it raises
immediately if a requested layer name does not exist in the model.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class FeatureExtractor:
    """
    Attaches PyTorch forward hooks to specific named layers and collects
    the output tensors emitted at each hook site during a forward pass.

    The reason to wrap this in a class rather than just calling
    register_forward_hook inline is cleanup -- hooks hold a reference to
    a closure which holds a reference to the feature dict, which means
    the model will keep accumulating tensors forever if you forget to
    remove them. The class tracks all handles and removes them either
    manually via remove_hooks() or automatically in __del__.

    Parameters
    model : nn.Module
        The network to instrument. It is not modified -- hooks are
        non-destructive and can be removed at any time.
    layer_names : list[str]
        Dot-separated module paths as produced by dict(model.named_modules()).
        For ResNet50 these look like 'layer1', 'layer2.0.conv1', etc.

    Raises
    KeyError
        Raised immediately at construction time if any name in layer_names
        does not exist in the model. Better to crash loudly here than to
        silently collect nothing and produce wrong AVR numbers later.

    Example
    >>> extractor = FeatureExtractor(model, ['layer1', 'layer2.0.conv1'])
    >>> outputs = extractor(x)  # dict[str, Tensor]
    """

    def __init__(self, model: nn.Module, layer_names: List[str]) -> None:
        self.model = model
        self.layer_names = layer_names
        self._features: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHook] = []

        named = dict(model.named_modules())
        for name in layer_names:
            if name not in named:
                raise KeyError(
                    f"Layer '{name}' not found in model. "
                    f"Available layers: {list(named.keys())[:20]} ..."
                )
            self._handles.append(
                named[name].register_forward_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str):
        """
        Returns a closure (not a method) so that each hook captures its
        own specific name string. If I used a single hook function with
        name as an argument passed at call time I would hit the classic
        Python late-binding problem and all hooks would record under the
        same name. The closure freezes the value of name at registration.
        """

        def hook(_module: nn.Module, _input, output: torch.Tensor) -> None:
            # detach() so the tensor is not part of the computation graph,
            # cpu() so we are not holding GPU memory after the forward pass.
            self._features[name] = output.detach().cpu()

        return hook

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Runs a full forward pass under no_grad (we only need activations,
        not gradients) and returns whatever the hooks collected.

        By clearing _features before the pass we avoid accidentally
        returning stale activations from a previous call if a hook
        somehow did not fire.

        Parameters
        x : Tensor
            Input batch, shape (B, C, H, W).

        Returns
        dict[str, Tensor]
            Layer name -> feature map tensor captured at that layer.
        """
        self._features.clear()
        with torch.no_grad():
            self.model(x)
        return dict(self._features)

    def remove_hooks(self) -> None:
        """
        Detaches every registered hook from the model. Should be called
        when you are done extracting features. If you forget, the closures
        keep a reference to _features which keeps everything alive in
        memory -- not catastrophic for small networks but gets bad fast
        with something like a ViT on many batches.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __del__(self) -> None:
        self.remove_hooks()


def register_hooks(
    model: nn.Module,
    layer_names: List[str],
) -> FeatureExtractor:
    """
    One-liner wrapper around FeatureExtractor for callers who just want
    to get set up quickly without thinking about the class API. Returns
    the extractor so the caller can still call .remove_hooks() later.

    Parameters
    model : nn.Module
        Target model.
    layer_names : list[str]
        Layers to instrument.

    Returns
    FeatureExtractor
    """
    return FeatureExtractor(model, layer_names)
