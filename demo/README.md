# Demo

Research/education only, not a diagnostic tool. Nothing in this folder may contain real
ultrasound images.

## v0.1: reproduce the results (available now)

The demo for researchers is the pipeline itself: open
[`notebooks/run_pipeline_colab.ipynb`](../notebooks/run_pipeline_colab.ipynb) in Colab, run
all cells (about 15 minutes on a T4), and you get:

- `outputs/RESULTS.md`: every model's macro-F1 with CI and per-class recall
- `outputs/<run>/confusion_matrix.png` for each model
- `outputs/cnn_resnet18/gradcam.png`: where the CNN looks, per class
- `outputs/audit/`: the duplicate pairs and the acquisition-by-class table

Suggested walkthrough for a 5-minute presentation:
1. The data audit: 6 duplicate pairs, and Normal images coming from a different source.
2. The shortcut baseline: what a model gets without looking at anatomy.
3. The CNN vs. kNN results, on all images and on standard exports only.
4. Grad-CAM: where the model looks, and the share of attention inside the fan.
5. The cross-dataset test, and what it says about generalisation.

## Planned (ROADMAP v0.4)

- **Static results page:** the README table, confusion matrices and the audit findings on
  one page, with the dataset citation. Grad-CAM shown only on images whose licence permits
  redistribution with attribution.
- **Optional Gradio app:** upload an image and see the class probabilities and a Grad-CAM
  overlay, with a permanent "research/education only, not a diagnostic tool" banner, no
  logging or storage of uploads, and example images from the public dataset only.
