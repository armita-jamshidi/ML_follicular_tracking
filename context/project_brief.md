# Project brief

**Product.** An educational/research ML project that classifies transvaginal ovarian
ultrasound images into Normal, Dominant Follicle and PCO, connected to how follicular
tracking works. Not a diagnostic tool; always framed as research/education.

**Owner and buyer.** Armita Jamshidi, building hands-on experience applying neural networks
and kNN to medical imaging.

**Audience.** Researchers in AI and medicine (see `customers/`).

**Pain.** Most public PCOS-ultrasound projects report results that don't survive scrutiny
(train/test duplicates, shortcuts, accuracy-only reporting).

**Promise.** A reproducible, well-documented pipeline with clear results that avoids the
data-leakage problems common in published PCOS projects.

**Constraints.**
- Free compute only: Colab T4 or Kaggle for CNNs, laptop CPU for kNN. Minutes per run.
- Privacy: no personal or patient-identifiable images in the repo, including the owner's
  own scans. Datasets are downloaded by script, not committed.
- Main dataset has class labels only (no masks), so no segmentation or follicle counts.
