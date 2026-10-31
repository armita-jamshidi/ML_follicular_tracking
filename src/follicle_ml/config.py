"""YAML config loading with dotted command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply overrides like ``train.epochs=10`` to a nested dict (returns a copy)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)
