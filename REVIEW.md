# Review checklist

Use for every PR and before sharing any result. Researchers in AI and medicine will look for
exactly these failure modes.

## Leakage and evaluation

- [ ] `follicle-ml audit` ran on the current data and `data/index.csv` is fresh
- [ ] No duplicate cluster spans train and test (folds are assigned by `StratifiedGroupKFold`)
- [ ] Early stopping, k, C and any threshold were chosen without the outer test fold
- [ ] BTX24 was not used for training except as declared pre-training, and was deduplicated
- [ ] Metrics: macro-F1 (with CI), per-class recall, confusion matrix; accuracy never alone
- [ ] `by_acquisition` metrics and the `standard_export` run are reported next to headline numbers
- [ ] The result beats the shortcut baseline (size + colour only) on the same folds
- [ ] Grad-CAM looked at for every class; attention is on ovarian tissue, not borders or text

## Privacy and data

- [ ] `git diff --stat` shows no images, archives, model weights or outputs
- [ ] No figure made from real images was committed
- [ ] The dataset is cited wherever results appear

## Framing

- [ ] "Research/education only, not a diagnostic tool" appears on anything user-facing
- [ ] Claims say "PCO morphology", not "PCOS diagnosis"
- [ ] Limitations are stated (small dataset, single centre, source confound in Normal)

## Code

- [ ] `pytest -q` passes; new behaviour has a synthetic-data test
- [ ] New options are in `configs/default.yaml` with a comment
- [ ] The relevant spec in `specs/` is updated

## Open questions for the owner

- Should the 31 non-standard Normal images be dropped from the headline result, or kept
  with the per-acquisition breakdown? (v0.1 keeps them and reports both.)
- Code licence: v0.1 uses MIT. Change if you prefer another.
