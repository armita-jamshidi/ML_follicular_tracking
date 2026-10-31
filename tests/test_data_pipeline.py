import numpy as np
from PIL import Image

from follicle_ml.config import apply_overrides, load_config
from follicle_ml.preprocess import crop_to_fan, letterbox
from follicle_ml.splits import inner_split


def test_config_overrides():
    cfg = load_config(overrides=["train.epochs=3", "model.name=efficientnet_b0"])
    assert cfg["train"]["epochs"] == 3 and cfg["model"]["name"] == "efficientnet_b0"
    assert apply_overrides({"a": {"b": 1}}, ["a.b=[1, 2]"])["a"]["b"] == [1, 2]


def test_crop_removes_text_outside_fan(main_root):
    img = Image.open(main_root / "PCO" / "PCO_1.jpg").convert("L")
    out = np.asarray(crop_to_fan(img))
    assert out.shape[0] < img.height and out.shape[1] <= img.width
    # The text sat in the top-left corner, outside the fan: it must be gone.
    assert np.asarray(img)[4:14, 4:60].max() > 200
    assert out[:10, :40].max() < 200


def test_letterbox_is_square_and_keeps_aspect():
    img = Image.new("L", (100, 50), 255)
    out = np.asarray(letterbox(img, 64))
    assert out.shape == (64, 64)
    assert out[0].max() == 0 and out[32].min() == 255


def test_index_records_acquisition(indexed):
    df, _ = indexed
    assert set(df["label"]) == {"Normal", "Dominant_Follicle", "PCO"}
    assert {"width", "height", "acquisition", "color_fraction"} <= set(df.columns)


def test_dedup_finds_planted_duplicate(indexed):
    df, pairs = indexed
    found = {frozenset((a, b)) for a, b in zip(pairs["image_id_i"], pairs["image_id_j"])}
    assert frozenset(("PCO/PCO_0.jpg", "PCO/PCO_dup.jpg")) in found
    assert df["is_dup_keeper"].sum() == len(df) - 1


def test_folds_are_stratified_and_keep_duplicates_together(indexed):
    df, _ = indexed
    assert df.groupby("dup_cluster")["fold"].nunique().max() == 1
    counts = df.groupby(["fold", "label"]).size().unstack()
    assert (counts.max() - counts.min()).max() <= 2


def test_inner_split_disjoint(indexed):
    df, _ = indexed
    train = df[df["fold"] != 0]
    fit, val = inner_split(train, 0.2, seed=0)
    assert set(fit["dup_cluster"]).isdisjoint(val["dup_cluster"])
    assert len(fit) + len(val) == len(train)
    assert set(val["label"]) == set(train["label"])
