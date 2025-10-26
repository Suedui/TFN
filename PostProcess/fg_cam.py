"""Frequency-domain Grad-CAM utilities for TFN models."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class FrequencyGradCAM:
    """Compute FG-CAM maps for 1D signals in the frequency domain."""

    def __init__(self, model: nn.Module, target_layer: str) -> None:
        self.model = model
        self.target_layer = self._get_layer(target_layer)
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _get_layer(self, name: str) -> nn.Module:
        module = self.model
        for attr in name.split('.'):
            module = getattr(module, attr)
        if not isinstance(module, nn.Module):
            raise ValueError(f"Attribute {name} is not a module")
        return module

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def generate(self, inputs: Tensor, target_class: Optional[int] = None) -> Tensor:
        """Generate a normalized FG-CAM heatmap in the frequency domain."""

        self.model.eval()
        inputs = inputs.requires_grad_(True)
        outputs = self.model(inputs)

        if target_class is None:
            target_indices = outputs.argmax(dim=1)
        else:
            target_indices = torch.full((outputs.size(0),), int(target_class), device=outputs.device)

        selected = outputs.gather(1, target_indices.view(-1, 1)).sum()
        self.model.zero_grad()
        selected.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients")

        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        freq_map = torch.fft.rfft(cam, dim=-1)
        magnitude = freq_map.abs()
        magnitude -= magnitude.amin(dim=-1, keepdim=True)
        magnitude /= magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return magnitude.detach().cpu()


__all__ = ["FrequencyGradCAM"]

