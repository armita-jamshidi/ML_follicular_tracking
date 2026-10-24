"""Index the datasets on disk as DataFrames (one row per image)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# Every Dominant Follicle and PCO image is a 728-pixel-tall scanner export from the same
# machine, but most Normal images are not (phone photos of a monitor, other scanners).
# We record this so evaluation can check the model is not just learning the source.
STANDARD_EXPORT_HEIGHT = 728


def _color_fraction(img: Image.Image) -> float:
    """Fraction of pixels with visible color (UI overlays, photo color casts)."""
    a = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.int16)
    spread = np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])
    return float((spread > 30).mean())


def index_main(root: str | Path) -> pd.DataFrame:
    """Return path, label, label_idx, size and acquisition metadata for the main dataset."""
    root = Path(root)
    rows = []
    for idx, cls in enumerate(CLASS_NAMES):
        for p in sorted((root / cls).iterdir()) if (root / cls).exists() else []:
            if p.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(p) as img:
                w, h = img.size
                rows.append(
                    {
                        "path": str(p),
                        "image_id": f"{cls}/{p.name}",
                        "label": cls,
                        "label_idx": idx,
                        "width": w,
                        "height": h,
                        "mode": img.mode,
                        "color_fraction": _color_fraction(img),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No images found under {root}. Run `follicle-ml download` first.")
    df["acquisition"] = np.where(df["height"] == STANDARD_EXPORT_HEIGHT, "standard_export", "other")
    return df


def index_folder_dataset(root: str | Path) -> pd.DataFrame:
    """Index a ``<split>/<label>/<image>`` tree (used for the BTX24 export)."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            rel = p.relative_to(root).parts
            rows.append({"path": str(p), "split": rel[0], "label": rel[1], "image_id": "/".join(rel)})
    return pd.DataFrame(rows)


def load_gray(path: str | Path) -> Image.Image:
    """Load an image as 8-bit grayscale (the scanners' color overlays carry no anatomy)."""
    with Image.open(path) as img:
        return img.convert("L")
