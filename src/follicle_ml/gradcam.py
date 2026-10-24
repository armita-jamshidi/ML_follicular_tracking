"""Grad-CAM, to check the model attends to follicles rather than text, calipers or borders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .preprocess import fan_mask


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self._acts = self._grads = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "_acts", o))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grads", go[0]))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """Return a [0, 1] heatmap the size of ``x`` (1xCxHxW) and the explained class."""
        x = x.requires_grad_(True)
        logits = self.model(x)
        class_idx = int(logits.argmax(1)) if class_idx is None else class_idx
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.detach().cpu().numpy(), class_idx


def energy_inside_fan(cam: np.ndarray, gray: np.ndarray) -> float:
    """Share of Grad-CAM mass that falls inside the ultrasound fan (1.0 = none on text/borders)."""
    mask = fan_mask(gray)
    return float(cam[mask].sum() / max(cam.sum(), 1e-8))


def save_overlay_grid(items: list[dict], path: str | Path, cols: int = 4) -> None:
    """``items``: dicts with ``gray`` (HxW uint8), ``cam`` (HxW), ``title``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = max(1, int(np.ceil(len(items) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, it in zip(np.atleast_1d(axes).ravel(), items):
        ax.imshow(it["gray"], cmap="gray")
        ax.imshow(it["cam"], cmap="jet", alpha=0.35)
        ax.set_title(it["title"], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
