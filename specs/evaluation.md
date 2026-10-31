# Spec: evaluation protocol

## Cross-validation
- Outer loop: the 5 folds from the audit. Each image is predicted exactly once, by a model
  that never saw it or its duplicates.
- Inner selection uses only the training fold:
  - CNN: a stratified 15% of duplicate clusters (`cv.inner_val_fraction`) for early
    stopping on macro-F1.
  - kNN / logreg: grid search with 3-fold `StratifiedGroupKFold` on the training fold.

## Metrics (all in `outputs/<run>/metrics.json`)
- **Primary:** macro-F1, pooled over out-of-fold predictions, with a 95% percentile
  bootstrap CI (1000 resamples over images).
- Per-fold mean ± std of macro-F1, balanced accuracy and per-class recall. Pooled numbers
  are preferred for small classes, because a fold can hold as few as 2 Normal images.
- Per-class recall and precision, and the confusion matrix (`confusion_matrix.png`, counts
  and row-normalised).
- `by_acquisition`: accuracy and per-class recall within each acquisition group.
- Accuracy is recorded but never reported alone.

## Shortcut checks
- **Shortcut baseline** (`shortcut_logreg`): logistic regression on width, height, aspect
  ratio and colour fraction. Any model must be compared with it on the same folds.
- **Sensitivity analysis:** every model is also run with `-s data.subset=standard_export`.
- **Grad-CAM energy inside the fan** (see `explainability.md`).

## Reporting
- README table: pooled macro-F1 with CI for each model, for all images and for standard
  exports only, with the commit and config used.
- Always state n per class and that results are research/education only.
