# Researchers in AI and medicine

## Who they are

ML researchers working on medical imaging, and clinician-researchers (reproductive
medicine, radiology) who read ML papers. They review, cite, reuse code, or decide whether a
dataset or method is worth building on.

## What they need

- **Evidence there is no leakage.** Dedup report, how duplicates were handled, and that
  model selection never touched the test fold.
- **Evidence the model learned anatomy, not artefacts.** Shortcut baseline, per-acquisition
  metrics, Grad-CAM with a quantitative check.
- **Metrics that respect imbalance.** Macro-F1 with confidence intervals, per-class recall,
  confusion matrices. Accuracy alone is a red flag to them.
- **Reproducibility.** Pinned data version with checksum, one config file, one command per
  step, seeds, runs that fit on free GPUs.
- **Correct clinical framing.** PCO morphology vs. PCOS, single-image vs. serial tracking,
  explicit "not a diagnostic tool".
- **Proper credit.** Dataset cited, licences respected.

## What would make them stop reading

- Near-perfect accuracy on 301 images with no CI.
- Training on BTX24/Kaggle PCOS images without deduplication.
- Claims of "diagnosing PCOS".
- Results that can't be regenerated from the repo.

## How we serve them

`README.md` results table (both columns), `context/data_audit.md`, `specs/evaluation.md`,
`REVIEW.md`, and the results page planned in `demo/`.
