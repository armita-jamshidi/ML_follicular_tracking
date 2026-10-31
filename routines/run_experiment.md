# Routine: run an experiment

1. Open `notebooks/run_pipeline_colab.ipynb` in Colab (Runtime → T4 GPU) or Kaggle (GPU on).
   For CPU-only work (kNN, shortcut), run locally.
2. `follicle-ml download` then `follicle-ml audit`. Confirm the audit prints
   301 images, 295 unique, and the same acquisition table as `context/data_audit.md`.
   If anything differs, stop and run `routines/data_audit.md`.
3. Decide the change you are testing and set it with `-s key=value` (or edit
   `configs/default.yaml` in a PR if it becomes the new default). Change one thing at a time.
4. Run the model on all images **and** with `-s data.subset=standard_export`.
5. Run `follicle-ml baselines` on the same commit if it hasn't been run, so the shortcut and
   kNN numbers share folds with the new model.
6. For CNN runs, `follicle-ml gradcam` and look at every panel. Note the energy-inside-fan value.
7. `follicle-ml report`. Copy headline numbers (pooled macro-F1 + CI, per-class recall) into
   the README table with the commit hash and any non-default config.
8. Download `outputs/` to your own drive if you want to keep it. Never commit it.
9. Go through `REVIEW.md` before sharing.
