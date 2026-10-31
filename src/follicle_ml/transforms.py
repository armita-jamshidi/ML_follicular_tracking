"""Image transforms, including ultrasound-style speckle noise."""

from __future__ import annotations

import torch
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SpeckleNoise:
    """Multiplicative Gaussian noise, a simple model of ultrasound speckle."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        return (x * (1 + self.std * torch.randn_like(x))).clamp(0, 1)


class GrayToRGB:
    """Repeat a 1-channel tensor to 3 channels for ImageNet-pretrained backbones."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


def train_transform(aug: dict) -> T.Compose:
    steps = []
    if aug.get("hflip", True):
        steps.append(T.RandomHorizontalFlip())
    if aug.get("vflip", False):
        steps.append(T.RandomVerticalFlip())
    if aug.get("rotation_deg", 0):
        steps.append(T.RandomRotation(aug["rotation_deg"]))
    steps.append(T.ColorJitter(brightness=aug.get("brightness", 0), contrast=aug.get("contrast", 0)))
    steps += [T.ToTensor(), SpeckleNoise(aug.get("speckle_std", 0)), GrayToRGB(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


def eval_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), GrayToRGB(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
