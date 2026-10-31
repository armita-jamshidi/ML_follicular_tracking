import shutil

import pandas as pd

from follicle_ml.cli import main


def test_audit_and_shortcut_baseline_end_to_end(main_root, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(main_root, data / "main")
    common = ["-s", f"data_dir={data}", "-s", f"output_dir={tmp_path / 'out'}", "-s", "cv.n_splits=3"]
    main([*common, "audit"])
    index = pd.read_csv(data / "index.csv")
    assert {"fold", "dup_cluster", "acquisition"} <= set(index.columns)
    assert (tmp_path / "out" / "audit" / "duplicate_pairs.csv").exists()
    main([*common, "-s", "knn.raw_pixel_size=32", "-s", "knn.k_values=[1,3]", "baselines", "--skip-embeddings"])
    assert (tmp_path / "out" / "knn_raw_pixels" / "metrics.json").exists()
    main([*common, "report"])
    assert "shortcut_logreg" in (tmp_path / "out" / "RESULTS.md").read_text()
