# Spec: data pipeline

## Download
- `follicle-ml download` fetches the main archive from a pinned commit, verifies SHA-256,
  extracts with `unrar`, `7z` or `bsdtar` (first one found), deletes `desktop.ini`.
- `--secondary` exports BTX24 via `datasets.load_dataset` to `data/btx24/<split>/<label>/`.
- Everything lands in `data/`, which is git-ignored. Re-running is a no-op unless `--force`.

## Audit (required before any model)
`follicle-ml audit` writes `data/index.csv` with one row per image:
`path, image_id, label, label_idx, width, height, mode, color_fraction, acquisition,
dup_cluster, is_dup_keeper, fold`, plus `outputs/audit/` (duplicate pairs, summary,
acquisition-by-class table).

- **Dedup:** 64-bit pHash on the grayscale full image; pairs with Hamming distance
  ≤ `data.dup_threshold` (default 4) are duplicates; union-find turns pairs into clusters.
- **Acquisition:** `standard_export` if height is 728 px, else `other`. This heuristic fits
  the current archive only; revisit if the data changes.
- **Folds:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`, stratified
  by class, grouped by `dup_cluster`. Fold assignment is deterministic given the seed.

## Preprocessing (per image, cached in memory)
1. Convert to 8-bit grayscale.
2. `crop_to_fan` (if `data.crop_to_fan`): threshold > 20, 9×9 opening to remove thin text,
   keep the largest component, close and fill holes, fill each row between its outermost
   pixels, blank everything outside, crop to the bounding box.
3. `letterbox` to `data.image_size` (224) without changing aspect ratio.
4. Repeat to 3 channels and normalise with ImageNet statistics.

## Acceptance
- Planted duplicates are found; distinct synthetic scans are not merged.
- No duplicate cluster spans two folds; class counts per fold differ by at most 2.
- Text outside the fan is removed (tested on synthetic scans).
