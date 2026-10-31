"""Fine-tune a pretrained CNN with stratified, duplicate-aware k-fold cross-validation."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from . import CLASS_NAMES
from .models import build_model
from .splits import inner_split
from .torch_data import ImageCache, UltrasoundDataset
from .transforms import eval_transform, train_transform


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def class_weights(labels, n_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    return torch.tensor(counts.sum() / (n_classes * counts.clip(min=1)), dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.softmax(model(x.to(device)), 1).cpu().numpy() for x, _ in loader])


def train_one(cfg: dict, fit_df: pd.DataFrame, val_df: pd.DataFrame, cache: ImageCache, device: str, log=print):
    """Train on ``fit_df``, early-stop on macro-F1 of ``val_df``. Returns the best model."""
    t = cfg["train"]
    n_classes = len(CLASS_NAMES)
    model = build_model(cfg["model"]["name"], n_classes, cfg["model"]["pretrained"]).to(device)
    weights = class_weights(fit_df["label_idx"], n_classes).to(device) if t["class_weights"] else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    loader_args = {"batch_size": t["batch_size"], "num_workers": t.get("num_workers", 0)}
    fit_loader = DataLoader(UltrasoundDataset(fit_df, cache, train_transform(cfg["augment"])), shuffle=True, **loader_args)
    val_loader = DataLoader(UltrasoundDataset(val_df, cache, eval_transform()), **loader_args)

    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(t["epochs"]):
        model.train()
        total = 0.0
        for x, y in fit_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        val_pred = predict(model, val_loader, device).argmax(1)
        val_f1 = f1_score(val_df["label_idx"], val_pred, average="macro", labels=list(range(n_classes)), zero_division=0)
        log(f"  epoch {epoch + 1:02d} loss {total / len(fit_df):.4f} val macro-F1 {val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1, best_state, stale = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= t["patience"]:
                break
    model.load_state_dict(best_state)
    return model


def cross_validate_cnn(cfg: dict, df: pd.DataFrame, out_dir: str | Path | None = None, device: str | None = None,
                       log=print) -> pd.DataFrame:
    """Return out-of-fold predictions for every image. Optionally saves each fold's weights."""
    device = device or default_device()
    seed_everything(cfg["seed"])
    cache = ImageCache(cfg["data"]["image_size"], cfg["data"]["crop_to_fan"])
    rows = []
    for k in sorted(df["fold"].unique()):
        log(f"Fold {k}")
        train_df, test_df = df[df["fold"] != k], df[df["fold"] == k]
        fit_df, val_df = inner_split(train_df, cfg["cv"]["inner_val_fraction"], cfg["seed"] + int(k))
        model = train_one(cfg, fit_df, val_df, cache, device, log)
        loader = DataLoader(UltrasoundDataset(test_df, cache, eval_transform()), batch_size=cfg["train"]["batch_size"])
        proba = predict(model, loader, device)
        part = test_df[["image_id", "path", "label", "label_idx", "fold", "acquisition"]].copy()
        part["pred"] = proba.argmax(1)
        for c, name in enumerate(CLASS_NAMES):
            part[f"prob_{name}"] = proba[:, c]
        rows.append(part)
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(out_dir) / f"model_fold{k}.pt")
    return pd.concat(rows).sort_index()
