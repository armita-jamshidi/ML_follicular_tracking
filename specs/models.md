# Spec: models

## Fine-tuned CNN (main model)
- Backbone: torchvision `resnet18` (default) or `efficientnet_b0`, ImageNet weights, head
  replaced with a 3-class linear layer. All layers trainable.
- Input: 224×224 letterboxed fan crop, grayscale repeated to 3 channels.
- Augmentation (training only): horizontal flip, rotation ±10°, brightness/contrast jitter
  0.2, multiplicative Gaussian speckle (std 0.1). No vertical flip: the probe is always at
  the top of the image.
- Loss: cross-entropy with inverse-frequency class weights from the fit split.
- Optimiser: AdamW (lr 3e-4, weight decay 1e-4), cosine schedule, up to 25 epochs,
  early stopping with patience 6 on inner-validation macro-F1; the best epoch is restored.
- Budget: about a minute per fold on a Colab T4.

## Baselines
- **kNN on raw pixels:** 64×64 fan crops, flattened, standardised, Euclidean,
  distance-weighted; k from {1, 3, 5, 7, 9, 15} by inner CV. A deliberately weak baseline.
- **kNN on frozen embeddings:** pooled penultimate features of the ImageNet backbone (no
  fine-tuning), standardised, cosine distance; same k grid.
- **Logistic regression on frozen embeddings:** balanced class weights, C from
  {0.001, 0.01, 0.1, 1}.
- **Shortcut logistic regression:** metadata only (see `evaluation.md`).

Embeddings are computed once for all images. That is not leakage, because the extractor
is frozen and never sees labels.
