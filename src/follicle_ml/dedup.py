"""Perceptual-hash duplicate detection.

Run before any training. Near-duplicates are grouped into clusters and the cluster id is
used as the group in cross-validation, so copies of the same scan never land on both
sides of a train/test split.
"""

from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image


def compute_hashes(paths: list[str], hash_size: int = 8) -> list[imagehash.ImageHash]:
    hashes = []
    for p in paths:
        with Image.open(p) as img:
            hashes.append(imagehash.phash(img.convert("L"), hash_size=hash_size))
    return hashes


def duplicate_pairs(df: pd.DataFrame, threshold: int = 4, hashes=None) -> pd.DataFrame:
    """All pairs of rows whose pHash Hamming distance is <= threshold."""
    hashes = hashes if hashes is not None else compute_hashes(df["path"].tolist())
    bits = np.stack([h.hash.flatten() for h in hashes]).astype(np.uint8)
    rows = []
    for start in range(0, len(bits), 256):
        block = bits[start : start + 256]
        dist = (block[:, None, :] != bits[None, :, :]).sum(-1)
        for a, j in zip(*np.nonzero(dist <= threshold)):
            i = start + int(a)
            if i < j:
                rows.append({"i": i, "j": int(j), "distance": int(dist[a, j])})
    pairs = pd.DataFrame(rows, columns=["i", "j", "distance"])
    for side in ("i", "j"):
        for col in ("image_id", "label"):
            if col in df.columns:
                pairs[f"{col}_{side}"] = df[col].to_numpy()[pairs[side]] if len(pairs) else []
        if "split" in df.columns:
            pairs[f"split_{side}"] = df["split"].to_numpy()[pairs[side]] if len(pairs) else []
    return pairs


def cluster_ids(n: int, pairs: pd.DataFrame) -> np.ndarray:
    """Union-find over duplicate pairs; returns a cluster id per row."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(pairs["i"], pairs["j"]):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def annotate_duplicates(df: pd.DataFrame, threshold: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``dup_cluster`` and ``is_dup_keeper`` columns. Returns (df, pairs)."""
    df = df.reset_index(drop=True).copy()
    pairs = duplicate_pairs(df, threshold)
    df["dup_cluster"] = cluster_ids(len(df), pairs)
    # Keep the first image of each cluster when a strictly deduplicated set is needed.
    df["is_dup_keeper"] = ~df.duplicated("dup_cluster", keep="first")
    return df, pairs


def summarize(df: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    sizes = df["dup_cluster"].value_counts()
    summary = {
        "n_images": int(len(df)),
        "n_unique_after_dedup": int(df["is_dup_keeper"].sum()),
        "n_duplicate_pairs": int(len(pairs)),
        "n_clusters_with_duplicates": int((sizes > 1).sum()),
    }
    if len(pairs) and "label_i" in pairs:
        summary["n_cross_label_pairs"] = int((pairs["label_i"] != pairs["label_j"]).sum())
    if len(pairs) and "split_i" in pairs:
        summary["n_cross_split_pairs"] = int((pairs["split_i"] != pairs["split_j"]).sum())
    return summary


def write_report(df: pd.DataFrame, pairs: pd.DataFrame, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "duplicate_pairs.csv", index=False)
    df.to_csv(out_dir / "index_with_clusters.csv", index=False)
    summary = summarize(df, pairs)
    pd.Series(summary).to_json(out_dir / "dedup_summary.json", indent=2)
    return summary
