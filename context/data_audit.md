# Main dataset audit (v0.1, 2026-10-24)

Produced by `follicle-ml audit`. Re-run it whenever the data or the dedup threshold changes
and update this file (see `routines/data_audit.md`).

## Duplicates

pHash (64-bit) Hamming distance ≤ 4 on the full image. Pairs at distance 0–2 were checked
by eye and are the same scan; pairs at distance 6–8 were checked and are different scans
with similar scanner layout, which is why the threshold is 4.

| Image A | Image B | Distance |
|---|---|---|
| Dominant_Follicle_1 | Dominant_Follicle_128 | 0 |
| Dominant_Follicle_60 | Dominant_Follicle_61 | 0 |
| Dominant_Follicle_85 | Dominant_Follicle_86 | 0 |
| PCO_111 | PCO_2 | 0 |
| PCO_1 | PCO_55 | 2 |
| PCO_59 | PCO_60 | 2 |

295 unique images out of 301. No pair crosses classes. Each pair is a duplicate cluster, and
clusters are kept in the same CV fold, so a scan is never in both train and test.

## Acquisition-source confound

| Class | Standard scanner export (728 px tall) | Other source |
|---|---|---|
| Normal | 10 | 31 |
| Dominant Follicle | 144 | 0 |
| PCO | 116 | 0 |

The "other" Normal images are mostly phone photos of a monitor or exports from other
scanners: different size, framing, colour cast and on-screen UI. A model can therefore
separate most Normal images by **how they were captured**, not by anatomy.

Evidence: a logistic regression on only width, height, aspect ratio and colour fraction
reaches pooled macro-F1 0.635 and Normal recall 0.73 (0.97 on the "other" group) under the
same 5-fold CV. Restricted to standard exports, the same model falls to 0.348 (chance).

Consequences for the pipeline:
- Every run reports metrics per acquisition group (`by_acquisition` in `metrics.json`).
- `-s data.subset=standard_export` re-runs any model on the 270 standard exports
  (only 10 Normal), as a sensitivity analysis.
- The shortcut baseline is the bar a model must beat on the full set.
- Fan cropping works on standard exports but not on photos of monitors (the whole photo is
  bright), so crops of the "other" group still contain screen borders. Tracked in ROADMAP v0.3.

## On-screen text and calipers

Standard exports carry the probe name, frequency, depth ruler and a date stamp outside the
fan. `preprocess.crop_to_fan` blanks everything outside the fan. Rulers touching the fan edge
can survive; Grad-CAM energy inside the fan is reported to catch the model relying on them.
