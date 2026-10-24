from functools import partial

import numpy as np
import pandas as pd
import torch

from follicle_ml import CLASS_NAMES
from follicle_ml.baselines import cross_validate_sklearn, knn_model, raw_pixel_features, shortcut_features
from follicle_ml.evaluate_external import binary_metrics, btx24_binary_labels
from follicle_ml.gradcam import GradCAM, energy_inside_fan
from follicle_ml.metrics import classification_metrics, summarize_cv
from follicle_ml.models import build_model, last_conv_layer
from follicle_ml.splits import inner_split
from follicle_ml.torch_data import ImageCache
from follicle_ml.train import class_weights, train_one
from follicle_ml.transforms import SpeckleNoise, eval_transform

TINY_CFG = {
    "model": {"name": "resnet18", "pretrained": False},
    "train": {"epochs": 1, "batch_size": 8, "lr": 1e-3, "weight_decay": 0.0, "patience": 1,
              "class_weights": True, "num_workers": 0},
    "augment": {"hflip": True, "rotation_deg": 10, "brightness": 0.2, "contrast": 0.2, "speckle_std": 0.1},
}


def test_metrics_perfect_and_cv_summary():
    m = classification_metrics([0, 1, 2, 2], [0, 1, 2, 2])
    assert m["macro_f1"] == 1.0 and m["recall"]["PCO"] == 1.0
    oof = pd.DataFrame({"fold": [0, 0, 1, 1, 1, 0], "label_idx": [0, 1, 2, 0, 1, 2], "pred": [0, 1, 2, 1, 1, 2],
                        "acquisition": ["a", "a", "a", "b", "a", "a"]})
    s = summarize_cv(oof)
    assert {"pooled", "per_fold_mean_std", "by_acquisition"} <= set(s)
    lo, hi = s["pooled"]["macro_f1_95ci"]
    assert 0 <= lo <= hi <= 1


def test_class_weights_upweight_minority():
    w = class_weights([0, 1, 1, 1, 2, 2], 3)
    assert w[0] > w[2] > w[1]


def test_speckle_keeps_range():
    x = torch.rand(1, 16, 16)
    y = SpeckleNoise(0.5)(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_sklearn_baselines_run(indexed):
    df, _ = indexed
    oof = cross_validate_sklearn(raw_pixel_features(df, 32), df, partial(knn_model, [1, 3]))
    assert len(oof) == len(df) and oof["pred"].isin(range(3)).all()
    assert shortcut_features(df).shape == (len(df), 4)


def test_cnn_trains_and_gradcam(indexed):
    df, _ = indexed
    fit, val = inner_split(df, 0.2, seed=0)
    model = train_one(TINY_CFG, fit, val, ImageCache(64), "cpu", log=lambda *_: None)
    cache = ImageCache(64)
    img = cache(df["path"].iloc[0])
    cam, cls = GradCAM(model, last_conv_layer(model))(eval_transform()(img).unsqueeze(0))
    assert cam.shape == (64, 64) and 0 <= cls < len(CLASS_NAMES)
    assert 0 <= energy_inside_fan(cam, np.asarray(img)) <= 1


def test_efficientnet_builds():
    out = build_model("efficientnet_b0", 3, pretrained=False)(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 3)


def test_btx24_label_mapping_and_binary_metrics():
    y = btx24_binary_labels(pd.Series(["infected", "notinfected", "Not Infected"]))
    assert y.tolist() == [1, 0, 0]
    m = binary_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.4, 0.2]))
    assert m["auroc"] == 1.0 and m["sensitivity"] == 0.5 and m["specificity"] == 1.0
