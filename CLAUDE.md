# CLAUDE.md

Instructions for AI coding agents (and humans) working in this repository.

## What this project is

An educational and research ML project that classifies transvaginal ovarian ultrasound
images into Normal, Dominant Follicle and PCO, connected to how follicular tracking works.
The audience is researchers in AI and medicine, so the bar is a rigorous, honest,
reproducible evaluation, not a high headline number.

**It is not a diagnostic tool.** Every README, report, figure caption, demo and model card
must say "research/education only, not a diagnostic tool". Never write copy that implies
clinical use ("detects PCOS", "diagnoses"). Say "classifies images labelled as PCO
morphology". PCO morphology on ultrasound is not a PCOS diagnosis (see
`context/follicular_tracking.md`).

## Hard rules

1. **No images in git, ever.** That covers datasets, the owner's own scans, screenshots of
   scans and Grad-CAM figures made from real images. `data/` and `outputs/` are ignored.
   Datasets are downloaded by `follicle-ml download`. Tests use synthetic images only.
2. **Run the audit before training.** `follicle-ml audit` (perceptual-hash dedup,
   acquisition audit, fold assignment) must run before any model. Training commands read
   `data/index.csv`, which only the audit writes.
3. **Never merge BTX24 with the main dataset for training.** BTX24 is only a deduplicated
   cross-dataset test (PCO vs. Normal + Dominant Follicle), or optional pre-training
   followed by fine-tuning on the main dataset. Deduplicate it within the set and across
   its splits first.
4. **No test-fold leakage.** Early stopping, hyperparameter search and threshold choice use
   only the training fold (`splits.inner_split`, inner grouped CV). Duplicate clusters stay
   in one fold.
5. **Report macro-F1, per-class recall and confusion matrices**, never accuracy alone.
   Report per acquisition group (`by_acquisition` in `metrics.json`) and the
   `standard_export` sensitivity analysis next to every headline number.
6. **Cite the main dataset** (Borna, Saadat et al., Front. Physiol. 2025,
   doi:10.3389/fphys.2025.1520898) wherever results are shown.
7. **Free compute only:** Colab T4 or Kaggle. Keep a training run to minutes.

## Layout

```
src/follicle_ml/   package: download, data, dedup, splits, preprocess, transforms,
                   torch_data, models, train, baselines, metrics, gradcam,
                   evaluate_external, cli
configs/           default.yaml (override with -s key=value)
tests/             pytest on synthetic images
notebooks/         thin Colab/Kaggle runner, no logic
context/           datasets, data audit, clinical background, related work
customers/         audience and what they need
specs/             what each part must do; update the spec when behaviour changes
demo/              results demo plan
routines/          repeatable procedures
```

## Commands

```bash
pip install -e '.[dev]'
pytest -q                                   # must pass before every PR
follicle-ml download | audit | baselines | train-cnn | gradcam | cross-dataset | report
follicle-ml -s data.subset=standard_export train-cnn
```

## Conventions

* Python 3.10+, type hints, short module docstrings explaining *why*. Keep logic in the
  package; notebooks only call the CLI.
* Every new experiment option goes in `configs/default.yaml` with a comment.
* New behaviour gets a spec update in `specs/` and a synthetic-data test.
* Each run writes `metrics.json`, `oof_predictions.csv` and `confusion_matrix.png` to
  `outputs/<run>/`. Don't commit them; copy headline numbers into the README table by hand
  with the run config.
* Before opening a PR, go through `REVIEW.md`.
* Track work in the owner's Notion board (Derek): add a task when starting something; when
  it's ready, write a resume-style summary on the task and set it to "Ready for Review".
  The owner moves it to "Done".
