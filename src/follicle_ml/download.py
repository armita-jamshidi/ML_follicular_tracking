"""Download datasets into ``data/``. Nothing downloaded here is ever committed."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path

# Pinned to a commit so the archive cannot change underneath an experiment.
MAIN_COMMIT = "20061a1d2cb7ed54ce7d8a71069a29b96222d960"
MAIN_URL = (
    "https://raw.githubusercontent.com/HananSaadat/ovarian_ultrasound_dataset/"
    f"{MAIN_COMMIT}/ovarian_ultrasound_dataset.rar"
)
MAIN_SHA256 = "438ee9602771aeae20870dff818fa1718fa1e3604588cff9a22acb9a1f0ae527"

SECONDARY_HF_ID = "BTX24/PCOS-ultrasound-dataset"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_rar(archive: Path, dest: Path) -> None:
    """Extract a RAR5 archive with whichever tool is installed."""
    dest.mkdir(parents=True, exist_ok=True)
    candidates = [
        ["unrar", "x", "-o+", str(archive), str(dest) + "/"],
        ["7z", "x", "-y", f"-o{dest}", str(archive)],
        ["bsdtar", "-xf", str(archive), "-C", str(dest)],
    ]
    for cmd in candidates:
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            return
    raise RuntimeError(
        "No RAR extractor found. Install one of: unrar, p7zip (7z), or libarchive-tools (bsdtar). "
        "Colab and Kaggle already ship unrar."
    )


def download_main(data_dir: str | Path = "data", force: bool = False) -> Path:
    """Download and extract the Borna/Saadat et al. (2025) ovarian ultrasound dataset.

    Returns the directory containing the class folders.
    """
    data_dir = Path(data_dir)
    out = data_dir / "main"
    if out.exists() and any(out.glob("*/*.jpg")) and not force:
        print(f"Main dataset already present at {out}")
        return out

    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "ovarian_ultrasound_dataset.rar"
    if not archive.exists() or force:
        print(f"Downloading {MAIN_URL}")
        urllib.request.urlretrieve(MAIN_URL, archive)
    digest = sha256(archive)
    if digest != MAIN_SHA256:
        raise RuntimeError(f"Checksum mismatch for {archive}: {digest} != {MAIN_SHA256}")

    _extract_rar(archive, out)
    for junk in out.rglob("desktop.ini"):
        junk.unlink()
    n = len(list(out.glob("*/*.jpg")))
    print(f"Extracted {n} images to {out}")
    return out


def download_secondary(data_dir: str | Path = "data", force: bool = False) -> Path:
    """Export the Hugging Face BTX24 PCOS set to ``data/btx24/<split>/<label>/*.png``.

    Use only as a deduplicated cross-dataset test set or for optional pre-training.
    Never merge it with the main dataset for training (see specs/cross_dataset.md).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("pip install -e '.[secondary]' to download the BTX24 dataset") from e

    out = Path(data_dir) / "btx24"
    if out.exists() and any(out.rglob("*.png")) and not force:
        print(f"BTX24 dataset already present at {out}")
        return out

    ds = load_dataset(SECONDARY_HF_ID)
    for split, part in ds.items():
        label_feature = part.features.get("label")
        names = getattr(label_feature, "names", None)
        for i, row in enumerate(part):
            label = row["label"]
            label = names[label] if names else str(label)
            target = out / split / label
            target.mkdir(parents=True, exist_ok=True)
            row["image"].save(target / f"{split}_{i:05d}.png")
    print(f"Exported BTX24 splits {list(ds.keys())} to {out}")
    return out
