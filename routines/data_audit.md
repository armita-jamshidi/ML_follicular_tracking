# Routine: data audit

Run when the dataset version, the dedup threshold or the preprocessing changes, and before
sharing any result.

1. `follicle-ml audit`.
2. Open `outputs/audit/duplicate_pairs.csv`. For every pair, and for a sample of pairs just
   above the threshold (`-s data.dup_threshold=8`), view both images side by side and
   confirm the threshold separates true duplicates from look-alikes.
3. Check `outputs/audit/acquisition_by_class.csv`. If any class is dominated by one source,
   the shortcut baseline and per-acquisition metrics must stay in every report.
4. View 10 random fan crops per class and per acquisition group (`prepare_image`) and make
   sure no text is left inside the crop and no anatomy was cut off.
5. Update `context/data_audit.md` with the date, the numbers and anything new.
6. If the archive changed, update `MAIN_COMMIT` and `MAIN_SHA256` in `download.py` in a PR.
