"""Cross-dataset test: train on the full main dataset, test PCO-vs-rest on deduplicated BTX24.

BTX24 is never used for training here. Main-dataset classes map to binary as
PCO -> positive, Normal + Dominant Follicle -> negative. The positive score is P(PCO).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from . import CLASS_NAMES
from .splits import inner_split
from .torch_data import ImageCache, UltrasoundDataset
from .train import default_device, predict, seed_everything, train_one
from .transforms import eval_transform

BTX24_LABELS = {"infected": 1, "notinfected": 0}
PCO = CLASS_NAMES.index("PCO")


def btx24_binary_labels(labels: pd.Series) -> np.ndarray:
    norm = labels.str.lower().str.replace(r"[^a-z]", "", regex=True)
    unknown = set(norm) - set(BTX24_LABELS)
    if unknown:
        raise ValueError(f"Unexpected BTX24 labels {unknown}; update BTX24_LABELS")
    return norm.map(BTX24_LABELS).to_numpy()


def binary_metrics(y_true: np.ndarray, p_pos: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (p_pos >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "auroc": float(roc_auc_score(y_true, p_pos)) if len(set(y_true)) == 2 else None,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def evaluate_binary_external(cfg: dict, main_df: pd.DataFrame, ext_df: pd.DataFrame, out_dir: str | Path) -> dict:
    seed_everything(cfg["seed"])
    device = default_device()
    cache = ImageCache(cfg["data"]["image_size"], cfg["data"]["crop_to_fan"])
    fit_df, val_df = inner_split(main_df, cfg["cv"]["inner_val_fraction"], cfg["seed"])
    model = train_one(cfg, fit_df, val_df, cache, device)

    ext_df = ext_df.copy()
    ext_df["binary"] = btx24_binary_labels(ext_df["label"])
    loader = DataLoader(UltrasoundDataset(ext_df, cache, eval_transform(), label_col="binary"), batch_size=32)
    proba = predict(model, loader, device)
    ext_df["p_pco"] = proba[:, PCO]
    result = binary_metrics(ext_df["binary"].to_numpy(), ext_df["p_pco"].to_numpy())
    result["by_split"] = {
        s: binary_metrics(g["binary"].to_numpy(), g["p_pco"].to_numpy()) for s, g in ext_df.groupby("split")
    }
    out_dir = Path(out_dir)
    ext_df.drop(columns=["path"]).to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
