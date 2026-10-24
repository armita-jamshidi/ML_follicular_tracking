"""PyTorch dataset over preprocessed images held in memory (the main set is ~300 images)."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import load_gray
from .preprocess import crop_to_fan, letterbox


def prepare_image(path: str, image_size: int, crop: bool = True) -> Image.Image:
    img = load_gray(path)
    if crop:
        img = crop_to_fan(img)
    return letterbox(img, image_size)


class ImageCache:
    """Preprocess each image once and reuse it across folds and epochs."""

    def __init__(self, image_size: int, crop: bool = True):
        self.image_size, self.crop, self._store = image_size, crop, {}

    def __call__(self, path: str) -> Image.Image:
        if path not in self._store:
            self._store[path] = prepare_image(path, self.image_size, self.crop)
        return self._store[path]


class UltrasoundDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache: ImageCache, transform: Callable, label_col: str = "label_idx"):
        self.paths = df["path"].tolist()
        self.labels = df[label_col].tolist() if label_col in df else [-1] * len(df)
        self.cache, self.transform = cache, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.transform(self.cache(self.paths[i])), torch.tensor(self.labels[i])
