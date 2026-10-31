"""CNN backbones (torchvision, ImageNet-pretrained) and embedding extraction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision import models as tvm

SUPPORTED = ("resnet18", "efficientnet_b0")


def build_model(name: str = "resnet18", num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    if name == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {SUPPORTED}")
    return model


def head(model: nn.Module) -> nn.Module:
    return model.fc if hasattr(model, "fc") else model.classifier[-1]


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Target layer for Grad-CAM."""
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    return model.features[-1]


def feature_extractor(name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Frozen backbone that returns pooled embeddings (head replaced by identity)."""
    model = build_model(name, num_classes=1, pretrained=pretrained)
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    else:
        model.classifier = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader, device: str = "cpu") -> np.ndarray:
    model.to(device).eval()
    feats = [model(x.to(device)).cpu().numpy() for x, _ in loader]
    return np.concatenate(feats)
