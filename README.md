# ML follicular tracking

Classifying transvaginal ovarian ultrasound images as **Normal**, **Dominant Follicle** or
**PCO** (polycystic ovarian morphology), with a fine-tuned CNN compared against kNN baselines.

> **Research and education only. This is not a diagnostic tool**, has not been clinically
> validated, and must not be used to make decisions about anyone's health.

## Why this repo exists

Many published PCOS-ultrasound models report near-perfect accuracy on datasets with
duplicated images across train and test splits. This project aims for the opposite: a small,
reproducible pipeline whose numbers you can trust, including the uncomfortable ones.

What the pipeline does before any model is trained:

* **Duplicate check.** Perceptual hashing found 6 duplicate pairs among the 301 images
  (295 unique). Duplicates are kept in the same cross-validation fold.
* **Acquisition audit.** 31 of the 41 Normal images come from a different source (phone
  photos of a monitor, other scanners) than every Dominant Follicle and PCO image. A
  model that sees only image size and colour, never anatomy, gets Normal recall of 0.97.
  Results are therefore reported per acquisition group, plus a sensitivity analysis on
  standard scanner exports only. See [context/data_audit.md](context/data_audit.md).
* **On-screen text removal.** Images are cropped to the ultrasound fan, and everything
  outside it (text, rulers, scanner UI) is blanked.

## Quick start

On a free GPU, open [notebooks/run_pipeline_colab.ipynb](notebooks/run_pipeline_colab.ipynb)
in Google Colab or Kaggle. Locally:

```bash
pip install -e '.[dev]'          # add ,secondary for the BTX24 cross-dataset test
follicle-ml download             # needs unrar, 7z or bsdtar to unpack the archive
follicle-ml audit                # dedup check + acquisition audit + folds (required)
follicle-ml baselines            # shortcut model, kNN (raw pixels, embeddings), logreg
follicle-ml train-cnn            # ResNet18, stratified 5-fold CV (GPU recommended)
follicle-ml gradcam
follicle-ml report               # writes outputs/RESULTS.md
```

Override config values with `-s`, e.g. `follicle-ml -s model.name=efficientnet_b0 train-cnn`
or `follicle-ml -s data.subset=standard_export baselines`. Defaults are in
[configs/default.yaml](configs/default.yaml).

## Results so far

CPU baselines on the main dataset, stratified duplicate-aware 5-fold CV, pooled
out-of-fold macro-F1 with a 95% bootstrap interval:

| Model | All images | Standard exports only |
|---|---|---|
| Shortcut: image size + colour only (no anatomy) | 0.635 (0.578–0.685) | 0.348 (0.296–0.400) |
| kNN on raw pixels (64×64) | 0.596 (0.533–0.652) | 0.347 (0.310–0.387) |
| kNN / logreg on frozen ResNet18 embeddings | run on Colab | run on Colab |
| Fine-tuned ResNet18 | run on Colab | run on Colab |

Chance macro-F1 for three classes is about 0.33. The shortcut row is the bar a real model
has to clear on the full set. Both baselines fall to chance once the acquisition shortcut
is removed, which is why the CNN must be reported on both columns.

## Data

* **Main:** Borna, Saadat et al. (2025) ovarian ultrasound dataset, 301 images described as one
  per patient (our audit found 6 duplicate pairs), radiologist labels, CC BY 4.0. Downloaded by script from a pinned commit and
  checked against a SHA-256. Please cite:
  > Borna M-R, Saadat H, Sepehri MM, Torkashvand H, Torkashvand L, Pilehvari S.
  > AI-powered diagnosis of ovarian conditions: insights from a newly introduced
  > ultrasound dataset. *Frontiers in Physiology*. 2025. doi:10.3389/fphys.2025.1520898
* **Secondary (optional):** Hugging Face `BTX24/PCOS-ultrasound-dataset`, binary labels,
  no stated source or licence. Used only as a deduplicated cross-dataset test, never
  merged into training.

No images are ever committed (see `.gitignore`), including the author's own scans.

## Repository map

| Path | What it holds |
|---|---|
| `src/follicle_ml/` | The pipeline package (CLI in `cli.py`) |
| `tests/` | Tests on synthetic images only |
| `configs/` | Experiment config |
| `notebooks/` | Thin Colab/Kaggle runner |
| `context/` | Background: datasets, data audit, clinical context, related work |
| `customers/` | Who this is for and what they need to trust it |
| `specs/` | Specs for the data pipeline, evaluation protocol, models, cross-dataset test |
| `demo/` | Plan for the results demo |
| `routines/` | Repeatable procedures (running experiments, audits, reviews) |
| `ROADMAP.md`, `REVIEW.md`, `CLAUDE.md` | Plan, review checklist, instructions for AI agents |

## Licence

Code: MIT. The dataset keeps its own licence (CC BY 4.0) and is not redistributed here.
