# Specs

What each part of the pipeline must do. Update the spec in the same PR as any behaviour change.

| Spec | Covers | Code |
|---|---|---|
| [data_pipeline.md](data_pipeline.md) | Download, indexing, dedup, preprocessing, folds | `download.py`, `data.py`, `dedup.py`, `preprocess.py`, `splits.py` |
| [evaluation.md](evaluation.md) | CV protocol, metrics, shortcut checks, reporting | `metrics.py`, `baselines.py`, `train.py` |
| [models.md](models.md) | CNN fine-tuning and kNN/logreg baselines | `models.py`, `train.py`, `baselines.py`, `transforms.py` |
| [explainability.md](explainability.md) | Grad-CAM | `gradcam.py` |
| [cross_dataset.md](cross_dataset.md) | BTX24 dedup and binary external test | `evaluate_external.py` |
