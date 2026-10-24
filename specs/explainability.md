# Spec: explainability (Grad-CAM)

- `follicle-ml gradcam` picks `--per-class` (default 4) images per class at random (seeded)
  and explains each with **the fold model that did not train on it**.
- Target layer: last block of `layer4` (ResNet) or the last `features` block (EfficientNet).
- Explained class: the predicted class.
- Output: `outputs/<run>/gradcam.png` (overlay grid titled with true and predicted class)
  and the mean **Grad-CAM energy inside the fan**: the share of heatmap mass inside the
  fan mask, printed next to the fan's share of the image (what uniform attention would
  give). Energy close to the uniform value, or well below 1, means attention on borders,
  rulers or padding and should block sharing the result until explained.
- Grad-CAM figures are made from real images, so they are never committed. Share them only
  in places where the dataset licence (CC BY 4.0, with attribution) is respected.
