# Spec: cross-dataset test on BTX24

Purpose: check whether what the model learned transfers to images from a different source.

1. `follicle-ml download --secondary` exports BTX24 to `data/btx24/<split>/<label>/`.
2. `follicle-ml cross-dataset`:
   - indexes BTX24 and runs the same pHash dedup **within the set and across its splits**,
     writing `btx24_duplicate_pairs.csv` and printing how many pairs cross splits;
   - keeps one image per duplicate cluster;
   - trains one CNN on the **whole main dataset** (early stopping on an inner split);
   - scores every unique BTX24 image with P(PCO); labels map `infected` → 1,
     `notinfected` → 0 (anything else raises an error);
   - reports AUROC, sensitivity and specificity at 0.5, macro-F1, the confusion matrix,
     overall and per original split.
3. BTX24 is never added to the main training set. Optional pre-training on BTX24 followed by
   fine-tuning on the main dataset is a separate, later experiment (ROADMAP v0.3) and must be
   labelled as such.

Caveat to report with any result: BTX24 has no stated source, licence or label provenance,
and its "infected" label may not match the main dataset's radiologist PCO label.
