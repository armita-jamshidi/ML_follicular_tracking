"""Command-line entry point: ``follicle-ml [-s key=value ...] <command>``.

Typical order: download -> audit -> baselines -> train-cnn -> gradcam -> cross-dataset -> report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASS_NAMES
from .config import load_config

DISCLAIMER = "Research/education only. Not a diagnostic tool."


def _index_path(cfg) -> Path:
    return Path(cfg["data_dir"]) / "index.csv"


def _load_index(cfg) -> pd.DataFrame:
    """Load the audited index, optionally restricted to one acquisition group."""
    path = _index_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `follicle-ml audit` first (it runs the dedup check).")
    df = pd.read_csv(path)
    subset = cfg["data"].get("subset", "all")
    if subset != "all":
        df = df[df["acquisition"] == subset].reset_index(drop=True)
        print(f"Restricted to acquisition={subset}: {df['label'].value_counts().to_dict()}")
    return df


def _run_name(cfg, name: str) -> str:
    subset = cfg["data"].get("subset", "all")
    return name if subset == "all" else f"{name}__{subset}"


def cmd_download(cfg, args):
    from .download import download_main, download_secondary

    download_main(cfg["data_dir"], force=args.force)
    if args.secondary:
        download_secondary(cfg["data_dir"], force=args.force)


def cmd_audit(cfg, args):
    """Index the main dataset, find duplicates, record acquisition groups and assign folds."""
    from .data import index_main
    from .dedup import annotate_duplicates, write_report
    from .splits import make_folds

    df = index_main(Path(cfg["data_dir"]) / "main")
    df, pairs = annotate_duplicates(df, cfg["data"]["dup_threshold"])
    df["fold"] = make_folds(df, cfg["cv"]["n_splits"], cfg["seed"])
    out = Path(cfg["output_dir"]) / "audit"
    summary = write_report(df, pairs, out)
    acq = pd.crosstab(df["label"], df["acquisition"])
    acq.to_csv(out / "acquisition_by_class.csv")
    df.to_csv(_index_path(cfg), index=False)

    print(f"Images per class: {df['label'].value_counts().to_dict()}")
    print(f"Dedup: {json.dumps(summary)}")
    if len(pairs):
        print(pairs[["image_id_i", "image_id_j", "distance"]].to_string(index=False))
    print(f"Acquisition by class:\n{acq}")
    print(f"Wrote {_index_path(cfg)} and {out}/")


def _run_sklearn(name, X, df, make_model, cfg):
    from .baselines import cross_validate_sklearn
    from .metrics import format_summary, save_results, summarize_cv

    oof = cross_validate_sklearn(X, df, make_model, cfg["seed"])
    summary = summarize_cv(oof)
    name = _run_name(cfg, name)
    save_results(Path(cfg["output_dir"]) / name, oof, summary, name)
    print(format_summary(name, summary))


def cmd_baselines(cfg, args):
    from functools import partial

    from torch.utils.data import DataLoader

    from .baselines import knn_model, logreg_model, raw_pixel_features, shortcut_features
    from .models import extract_embeddings, feature_extractor
    from .torch_data import ImageCache, UltrasoundDataset
    from .transforms import eval_transform

    df = _load_index(cfg)
    k_values = cfg["knn"]["k_values"]
    _run_sklearn("shortcut_logreg", shortcut_features(df), df, logreg_model, cfg)
    raw = raw_pixel_features(df, cfg["knn"]["raw_pixel_size"], cfg["data"]["crop_to_fan"])
    _run_sklearn("knn_raw_pixels", raw, df, partial(knn_model, k_values), cfg)
    if args.skip_embeddings:
        return
    cache = ImageCache(cfg["data"]["image_size"], cfg["data"]["crop_to_fan"])
    loader = DataLoader(UltrasoundDataset(df, cache, eval_transform()), batch_size=32)
    name = cfg["model"]["name"]
    emb = extract_embeddings(feature_extractor(name, pretrained=True), loader)
    np.save(Path(cfg["output_dir"]) / _run_name(cfg, f"embeddings_{name}.npy"), emb)
    _run_sklearn(f"knn_{name}_embeddings", emb, df, partial(knn_model, k_values, "cosine"), cfg)
    _run_sklearn(f"logreg_{name}_embeddings", emb, df, logreg_model, cfg)


def cmd_train_cnn(cfg, args):
    from .metrics import format_summary, save_results, summarize_cv
    from .train import cross_validate_cnn

    df = _load_index(cfg)
    name = _run_name(cfg, f"cnn_{cfg['model']['name']}")
    out = Path(cfg["output_dir"]) / name
    oof = cross_validate_cnn(cfg, df, out_dir=out)
    summary = summarize_cv(oof)
    summary["config"] = cfg
    save_results(out, oof, summary, name)
    print(format_summary(name, summary))


def cmd_gradcam(cfg, args):
    """Explain each image with the fold model that did not train on it."""
    import torch

    from .gradcam import GradCAM, energy_inside_fan, save_overlay_grid
    from .preprocess import fan_mask
    from .models import build_model, last_conv_layer
    from .torch_data import prepare_image
    from .transforms import eval_transform

    run = Path(args.run or Path(cfg["output_dir"]) / _run_name(cfg, f"cnn_{cfg['model']['name']}"))
    oof = pd.read_csv(run / "oof_predictions.csv")
    size, crop = cfg["data"]["image_size"], cfg["data"]["crop_to_fan"]
    items, energies, areas = [], [], []
    rng = np.random.default_rng(cfg["seed"])
    for label in CLASS_NAMES:
        subset = oof[oof["label"] == label]
        picks = subset.iloc[rng.permutation(len(subset))[: args.per_class]]
        for _, row in picks.iterrows():
            model = build_model(cfg["model"]["name"], len(CLASS_NAMES), pretrained=False)
            model.load_state_dict(torch.load(run / f"model_fold{row['fold']}.pt", map_location="cpu"))
            img = prepare_image(row["path"], size, crop)
            cam, pred = GradCAM(model, last_conv_layer(model))(eval_transform()(img).unsqueeze(0))
            gray = np.asarray(img)
            energies.append(energy_inside_fan(cam, gray))
            areas.append(float(fan_mask(gray).mean()))
            items.append({"gray": gray, "cam": cam, "title": f"true {label} / pred {CLASS_NAMES[pred]}"})
    save_overlay_grid(items, run / "gradcam.png", cols=args.per_class)
    # Uniform attention would put (fan area / image area) of the energy inside the fan.
    print(f"Saved {run / 'gradcam.png'}; Grad-CAM energy inside fan {np.mean(energies):.2f} "
          f"vs. {np.mean(areas):.2f} expected from uniform attention")


def cmd_cross_dataset(cfg, args):
    from .data import index_folder_dataset
    from .dedup import annotate_duplicates, summarize
    from .evaluate_external import evaluate_binary_external

    btx = index_folder_dataset(Path(cfg["data_dir"]) / "btx24")
    btx, pairs = annotate_duplicates(btx, cfg["data"]["dup_threshold"])
    out = Path(cfg["output_dir"]) / "cross_dataset_btx24"
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "btx24_duplicate_pairs.csv", index=False)
    print(f"BTX24 dedup: {json.dumps(summarize(btx, pairs))}")
    btx = btx[btx["is_dup_keeper"]]
    result = evaluate_binary_external(cfg, _load_index(cfg), btx, out)
    print(json.dumps(result, indent=2))


def cmd_report(cfg, args):
    from .metrics import format_summary

    out = Path(cfg["output_dir"])
    lines = ["# Results", "", f"_{DISCLAIMER}_", ""]
    for path in sorted(out.glob("*/metrics.json")):
        with open(path) as f:
            summary = json.load(f)
        if "pooled" in summary:
            lines.append(f"- {format_summary(path.parent.name, summary)}")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="follicle-ml", description=DISCLAIMER)
    parser.add_argument("--config", default=None)
    parser.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. -s train.epochs=10 (repeatable)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download datasets into data/")
    p.add_argument("--secondary", action="store_true", help="also download the BTX24 PCOS set")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_download)

    sub.add_parser("audit", help="dedup check, acquisition audit and CV folds").set_defaults(fn=cmd_audit)

    p = sub.add_parser("baselines", help="shortcut, raw-pixel kNN and embedding kNN/logreg baselines")
    p.add_argument("--skip-embeddings", action="store_true", help="skip baselines that need pretrained weights")
    p.set_defaults(fn=cmd_baselines)

    sub.add_parser("train-cnn", help="fine-tune the CNN with 5-fold CV").set_defaults(fn=cmd_train_cnn)

    p = sub.add_parser("gradcam", help="Grad-CAM overlays from a CNN run")
    p.add_argument("--run", default=None)
    p.add_argument("--per-class", type=int, default=4)
    p.set_defaults(fn=cmd_gradcam)

    sub.add_parser("cross-dataset", help="binary PCO test on deduplicated BTX24").set_defaults(fn=cmd_cross_dataset)
    sub.add_parser("report", help="collect metrics into outputs/RESULTS.md").set_defaults(fn=cmd_report)

    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
