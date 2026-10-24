"""Stratified, duplicate-aware cross-validation folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.Series:
    """Assign each row a test fold. Rows sharing a ``dup_cluster`` share a fold."""
    groups = df["dup_cluster"] if "dup_cluster" in df else np.arange(len(df))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, test_idx) in enumerate(sgkf.split(df, df["label_idx"], groups)):
        fold[test_idx] = k
    return pd.Series(fold, index=df.index, name="fold")


def inner_split(train_df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training fold into fit and early-stopping parts, keeping duplicate clusters together.

    The outer test fold is never used for model selection.
    """
    clusters = train_df.drop_duplicates("dup_cluster")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    _, val_c = next(sss.split(clusters, clusters["label_idx"]))
    val_ids = set(clusters["dup_cluster"].iloc[val_c])
    is_val = train_df["dup_cluster"].isin(val_ids)
    return train_df[~is_val], train_df[is_val]
