"""Classical baselines evaluated with the same folds as the CNN.

* kNN on raw pixels (weak baseline)
* kNN and logistic regression on frozen ImageNet CNN embeddings
* a "shortcut" model that sees only image size and colour statistics, never anatomy.
  If it scores well, the dataset has a non-anatomical signal the CNN could exploit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import CLASS_NAMES
from .torch_data import prepare_image

SHORTCUT_FEATURES = ["width", "height", "aspect", "color_fraction"]


def raw_pixel_features(df: pd.DataFrame, size: int = 64, crop: bool = True) -> np.ndarray:
    return np.stack(
        [np.asarray(prepare_image(p, size, crop), dtype=np.float32).ravel() / 255.0 for p in df["path"]]
    )


def shortcut_features(df: pd.DataFrame) -> np.ndarray:
    feats = df.assign(aspect=df["width"] / df["height"])
    return feats[SHORTCUT_FEATURES].to_numpy(dtype=np.float32)


def knn_model(k_values, metric: str = "euclidean"):
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(metric=metric, weights="distance"))
    return pipe, {"kneighborsclassifier__n_neighbors": list(k_values)}


def logreg_model():
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    return pipe, {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]}


def cross_validate_sklearn(X: np.ndarray, df: pd.DataFrame, make_model, seed: int = 42) -> pd.DataFrame:
    """Outer loop over ``df['fold']``; hyperparameters picked by inner grouped CV on the training fold only."""
    rows = []
    y, groups = df["label_idx"].to_numpy(), df["dup_cluster"].to_numpy()
    for k in sorted(df["fold"].unique()):
        tr, te = (df["fold"] != k).to_numpy(), (df["fold"] == k).to_numpy()
        pipe, grid = make_model()
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner)
        search.fit(X[tr], y[tr], groups=groups[tr])
        proba = search.predict_proba(X[te])
        part = df.loc[te, ["image_id", "label", "label_idx", "fold"]].copy()
        if "acquisition" in df:
            part["acquisition"] = df.loc[te, "acquisition"]
        part["pred"] = search.classes_[proba.argmax(1)]
        for c, col in enumerate(proba.T):
            part[f"prob_{CLASS_NAMES[search.classes_[c]]}"] = col
        part["best_params"] = str(search.best_params_)
        rows.append(part)
    return pd.concat(rows).sort_index()
