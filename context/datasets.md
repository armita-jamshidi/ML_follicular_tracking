# Datasets

## Main: Borna, Saadat et al. ovarian ultrasound dataset

- Source: https://github.com/HananSaadat/ovarian_ultrasound_dataset (pinned to commit
  `20061a1d`, archive SHA-256 in `src/follicle_ml/download.py`)
- Paper: Borna M-R, Saadat H, Sepehri MM, Torkashvand H, Torkashvand L, Pilehvari S.
  *AI-powered diagnosis of ovarian conditions: insights from a newly introduced ultrasound
  dataset.* Frontiers in Physiology, 2025. doi:10.3389/fphys.2025.1520898
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC12279744). **Cite it wherever results appear.**
- Licence: CC BY 4.0 (per the dataset README).
- Contents: 301 transvaginal images described as one per patient, radiologist class labels,
  no masks. Normal 41, Dominant Follicle 144, PCO 116.
- Format as downloaded: a RAR5 archive of JPEGs in `Normal/`, `Dominant_Follicle/`, `PCO/`
  (plus Windows `desktop.ini` files, which the download step deletes). Note the files are
  **not** uniform 1024×1024 grayscale: 270 are 728-pixel-tall RGB scanner exports
  (~500–550 px wide), the rest vary up to 1024 px on the long side. The pipeline converts
  everything to grayscale and letterboxes to 224×224.
- Role: the only training data. Evaluated with stratified, duplicate-aware 5-fold CV.

## Secondary: Hugging Face BTX24/PCOS-ultrasound-dataset

- Source: https://huggingface.co/datasets/BTX24/PCOS-ultrasound-dataset
- About 1.92k images, binary labels (`infected` / `notinfected`), no stated source or licence.
- Likely the same images as the Kaggle PCOS set that the PCONet paper (arXiv:2210.00407)
  found to have near-identical train and test images.
- Rules:
  - Deduplicate with perceptual hashing within the set and across its splits before any use.
  - **Never merge it with the main dataset for training.**
  - Use it only as a cross-dataset test (main-dataset PCO vs. Normal + Dominant Follicle
    mapped to binary) and, optionally, for pre-training before fine-tuning on the main set.
- Status: download and evaluation code exists (`follicle-ml download --secondary`,
  `follicle-ml cross-dataset`) but has not yet been run, because Hugging Face was not
  reachable from the build environment. The label mapping in `evaluate_external.py`
  fails loudly if the label names differ from what is expected.
