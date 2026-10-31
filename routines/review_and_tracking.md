# Routine: finish a piece of work

1. `pytest -q` passes. `git status` shows no images, archives, weights or `outputs/`.
2. Walk through `REVIEW.md` and fix or note anything unchecked.
3. Update the relevant `specs/` file, `ROADMAP.md` checkboxes and, if results changed, the
   README table.
4. Open a PR with: what changed, why, results (both columns), and anything left open.
5. On the Notion board (Derek): the task for this work should already exist (it is added
   when work starts). Add a resume-style summary to the task page, for example:
   - *Built a leakage-aware evaluation pipeline for ovarian ultrasound classification
     (PyTorch, scikit-learn): perceptual-hash dedup, duplicate-grouped stratified 5-fold CV,
     bootstrap CIs.*
   - *Identified an acquisition-source confound affecting 76% of one class; added a
     metadata-only shortcut baseline and per-source reporting.*
   Then set the status to **Ready for Review**. The owner moves it to **Done**.
