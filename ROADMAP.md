# Roadmap

Research/education project, not a diagnostic tool. Ordered by what most improves trust in
the results.

## v0.1: clean, honest pipeline (this release)

- [x] Scripted download from a pinned commit with SHA-256 check; nothing committed
- [x] Perceptual-hash dedup (6 duplicate pairs found), duplicate-aware stratified 5-fold CV
- [x] Acquisition audit, with the source confound in the Normal class documented
- [x] Fan cropping that blanks on-screen text, rulers and scanner UI
- [x] Shortcut baseline (image size + colour only) and raw-pixel kNN, run on real data
- [x] kNN and logistic regression on frozen ResNet18 / EfficientNet-B0 embeddings
- [x] Fine-tuned CNN with augmentation (flips, rotation, brightness/contrast, speckle),
      class weights and inner-split early stopping
- [x] Macro-F1 with bootstrap CI, per-class recall, confusion matrices, per-acquisition metrics
- [x] Grad-CAM overlays plus the share of attention inside the fan
- [x] BTX24 cross-dataset test harness (dedup, then PCO vs. rest)
- [x] Colab runner, CI on synthetic images

## v0.2: first full results

- [ ] Run embeddings baselines and ResNet18 on Colab; fill in the README results table
- [ ] Run EfficientNet-B0; compare with ResNet18
- [ ] Run the `standard_export` sensitivity analysis for every model
- [ ] Inspect Grad-CAM for each class and write up what the model attends to
- [ ] Run the BTX24 dedup and report how many near-duplicates cross its train/test split
- [ ] Run the cross-dataset test and report AUROC, sensitivity and specificity

## v0.3: tighten the evaluation

- [ ] Repeat 5-fold CV over 3 to 5 seeds and report the spread
- [ ] Better handling for the 31 non-standard Normal images: a fan detector that works on
      photos of monitors, or drop them and report the result on 10 Normal images
- [ ] Calibration (reliability diagram, ECE)
- [ ] Ablations: no fan crop, no class weights, no speckle augmentation
- [ ] Optional: pre-train on deduplicated BTX24, fine-tune on the main dataset, and
      compare with ImageNet-only

## v0.4: share it

- [ ] Static results page for researchers (see `demo/README.md`)
- [ ] Optional Gradio demo with a "research only" banner, on public example images only
- [ ] Model card and datasheet
- [ ] Short write-up connecting the classes to follicular tracking

## Not planned

- Clinical use, PCOS diagnosis, or anything that needs patient data beyond public datasets
- Segmentation or follicle counting (the main dataset has no masks)
