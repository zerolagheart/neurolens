"""Grad-CAM (Selvaraju et al., 2017) computed from a trained model.

For a target class ``c`` and feature maps ``A^k`` of the chosen layer::

    alpha_k = mean_{i,j} d y_c / d A^k_{ij}
    CAM     = ReLU( sum_k alpha_k A^k )

The map is upsampled to the input resolution and min-max normalised to ``[0, 1]`` per image.
It shows which regions most increased the class score *in the model's computation*. It is not
a segmentation, does not establish medical correctness and does not prove a tumour was found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neurolens.models.factory import UnsupportedArchitectureError, gradcam_target_layer


@dataclass
class GradCAMResult:
    cam: np.ndarray  # (H, W) float32 in [0, 1]
    target_class: int
    probabilities: np.ndarray  # raw softmax of the forward pass used for the CAM
    logits: np.ndarray
    degenerate: bool  # True if the CAM was all zeros (no positive evidence at this layer)


class GradCAM:
    """Context manager that hooks one layer and computes Grad-CAM maps.

    Hooks are removed on ``close()`` / context exit, even if the computation fails.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    @classmethod
    def for_architecture(cls, model: nn.Module, architecture: str) -> GradCAM:
        try:
            layer = gradcam_target_layer(model, architecture)
        except (UnsupportedArchitectureError, AttributeError, IndexError) as exc:
            raise UnsupportedArchitectureError(
                f"Grad-CAM is not configured for architecture '{architecture}'"
            ) from exc
        return cls(model, layer)

    def _forward_hook(self, _module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise UnsupportedArchitectureError("Grad-CAM target layer must output (B, C, H, W)")
        self._activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    def close(self) -> None:
        self._handle.remove()
        self._activations = None
        self._gradients = None

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __call__(self, inputs: torch.Tensor, target_class: int | None = None) -> GradCAMResult:
        """Compute a CAM for a single image tensor of shape ``(1, 3, H, W)``.

        ``target_class`` defaults to the predicted class.
        """
        if inputs.ndim != 4 or inputs.shape[0] != 1:
            raise ValueError("GradCAM expects a single image tensor of shape (1, 3, H, W)")
        was_training = self.model.training
        self.model.eval()
        try:
            # Gradients are required even if the caller is inside no_grad/inference_mode.
            with torch.inference_mode(False), torch.enable_grad():
                x = inputs.detach().clone().requires_grad_(True)
                logits = self.model(x)
                probs = torch.softmax(logits.detach(), dim=1)
                target = int(probs.argmax(1).item()) if target_class is None else int(target_class)
                if not 0 <= target < logits.shape[1]:
                    raise ValueError(f"target_class {target} out of range")
                self.model.zero_grad(set_to_none=True)
                logits[0, target].backward()
            if self._activations is None or self._gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
            activations = self._activations.detach()[0]
            gradients = self._gradients.detach()[0]
            weights = gradients.mean(dim=(1, 2), keepdim=True)
            cam = F.relu((weights * activations).sum(dim=0, keepdim=True))
            cam = F.interpolate(
                cam[None], size=inputs.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]
            cam_np = cam.cpu().numpy().astype(np.float32)
            peak = float(cam_np.max())
            degenerate = peak <= 0 or not np.isfinite(peak)
            cam_np = (
                np.zeros_like(cam_np)
                if degenerate
                else (cam_np - cam_np.min()) / (peak - cam_np.min() + 1e-12)
            )
            return GradCAMResult(
                cam=cam_np,
                target_class=target,
                probabilities=probs[0].cpu().numpy(),
                logits=logits.detach()[0].cpu().numpy(),
                degenerate=degenerate,
            )
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
            self._activations = None
            self._gradients = None
