"""Synthetic stand-in for the main dataset. Tests never touch real (patient) images."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import ndimage

from follicle_ml import CLASS_NAMES


def synthetic_scan(seed: int, n_follicles: int, big: bool, size=(300, 360)) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    img = Image.new("L", size, 0)
    d = ImageDraw.Draw(img)
    d.pieslice([-w // 2, -h // 3, w * 3 // 2, h], start=55, end=125, fill=110)  # the fan
    arr = np.asarray(img).astype(float)
    # Smooth random texture so every synthetic scan has a distinct perceptual hash.
    texture = ndimage.gaussian_filter(rng.normal(0, 1, arr.shape), 12)
    arr += (arr > 0) * (texture / texture.std() * 35 + rng.normal(0, 10, arr.shape))
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(n_follicles):
        r = 28 if big else 7
        x, y = rng.integers(w // 3, 2 * w // 3), rng.integers(h // 3, 2 * h // 3)
        d.ellipse([x - r, y - r, x + r, y + r], fill=10)
    d.text((4, 4), "C9-4v 19Hz", fill=255)  # on-screen text outside the fan
    return img


@pytest.fixture(scope="session")
def main_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("main")
    spec = {"Normal": (2, False), "Dominant_Follicle": (1, True), "PCO": (12, False)}
    for cls in CLASS_NAMES:
        (root / cls).mkdir()
        n_follicles, big = spec[cls]
        for i in range(10):
            synthetic_scan(100 * CLASS_NAMES.index(cls) + i, n_follicles, big).save(root / cls / f"{cls}_{i}.jpg", quality=95)
    # An exact duplicate pair inside PCO, as found in the real data.
    Image.open(root / "PCO" / "PCO_0.jpg").save(root / "PCO" / "PCO_dup.jpg", quality=90)
    return root


@pytest.fixture(scope="session")
def indexed(main_root):
    from follicle_ml.data import index_main
    from follicle_ml.dedup import annotate_duplicates
    from follicle_ml.splits import make_folds

    df = index_main(main_root)
    df, pairs = annotate_duplicates(df, threshold=4)
    df["fold"] = make_folds(df, n_splits=3, seed=0)
    return df, pairs
