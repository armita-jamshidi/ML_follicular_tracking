# Related work

- **Borna M-R, Saadat H, et al. (2025).** *AI-powered diagnosis of ovarian conditions:
  insights from a newly introduced ultrasound dataset.* Frontiers in Physiology.
  doi:10.3389/fphys.2025.1520898. Introduces the main dataset. Our audit adds two things to
  read alongside it: 6 duplicate pairs, and the acquisition-source imbalance in the Normal
  class (see `data_audit.md`). Compare any number we report with theirs only after checking
  their split protocol.
- **PCONet (arXiv:2210.00407).** CNN for PCOS detection from ovarian ultrasound. Reported
  that the widely used Kaggle PCOS ultrasound set has near-identical images in its train and
  test folders. That set likely overlaps with BTX24, which is why BTX24 is deduplicated and
  never used for training here.
- **Grad-CAM:** Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization*, ICCV 2017.
- **Shortcut learning:** Geirhos et al., *Shortcut learning in deep neural networks*, Nature
  Machine Intelligence 2020. DeGrave et al., *AI for radiographic COVID-19 detection selects
  shortcuts over signal*, Nature Machine Intelligence 2021. Both motivate the shortcut
  baseline and per-acquisition reporting.
- **Rotterdam criteria (2003)** and the **2018 international evidence-based guideline for
  PCOS**: definitions of PCOS and PCO morphology used in `follicular_tracking.md`.
