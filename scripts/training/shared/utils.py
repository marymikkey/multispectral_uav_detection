#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common utility functions for training scripts.
"""

from pathlib import Path
from typing import Dict, Optional, Any

import torch
import yaml
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def check_cuda(device: int = 0) -> None:
    """Check CUDA availability and print GPU info."""
    assert torch.cuda.is_available(), "CUDA is not available"
    print("=" * 60)
    print("GPU Information:")
    print(f"  Device: {torch.cuda.get_device_name(device)}")
    print(f"  Device index: {device}")
    print(f"  Torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print("=" * 60)


def check_ultralytics_import(imported_path: Path, expected_init: Path) -> None:
    """Ensure that ultralytics was imported from the correct repo."""
    imported_resolved = imported_path.resolve()
    expected_resolved = expected_init.resolve()
    print(f"[CHECK] ultralytics imported from: {imported_resolved}")
    if imported_resolved != expected_resolved:
        raise RuntimeError(
            "Wrong ultralytics imported.\n"
            f"Expected: {expected_resolved}\n"
            f"Got:      {imported_resolved}"
        )
    print("[CHECK] OK: repo ultralytics is being used")


def resolve_yaml_paths(yaml_path: Path, split_name: str):
    """Resolve absolute paths for a given split from a YAML dataset config."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = data.get("path", "")
    root = Path(root) if root else None

    split_value = data.get(split_name, [])
    if isinstance(split_value, str):
        split_value = [split_value]

    resolved = []
    for p in split_value:
        pp = Path(p)
        if not pp.is_absolute() and root is not None:
            pp = root / pp
        resolved.append(pp)

    return resolved, data


def count_images_in_path(ds_path: Path) -> int:
    """Count image files in a directory (recursively)."""
    if not ds_path.exists():
        return 0

    if ds_path.name == "images":
        root = ds_path
    elif (ds_path / "images").exists():
        root = ds_path / "images"
    else:
        root = ds_path

    return sum(
        1 for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def count_split_images(yaml_path: Path, split_name: str) -> int:
    """Count total images in a dataset split defined by YAML."""
    paths, _ = resolve_yaml_paths(yaml_path, split_name)
    return sum(count_images_in_path(p) for p in paths)


def format_seconds(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_best_epoch_metrics(results_csv_path: Path) -> Optional[Dict[str, Any]]:
    """Extract best epoch metrics from results.csv."""
    if not results_csv_path.exists():
        return None

    df = pd.read_csv(results_csv_path)
    if df.empty:
        return None

    required_cols = [
        "epoch",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    ]
    for col in required_cols:
        if col not in df.columns:
            return None

    best_idx = df["metrics/mAP50-95(B)"].idxmax()
    return {
        "epoch": int(df.loc[best_idx, "epoch"]),
        "mAP50": float(df.loc[best_idx, "metrics/mAP50(B)"]),
        "mAP50_95": float(df.loc[best_idx, "metrics/mAP50-95(B)"]),
        "precision": float(df.loc[best_idx, "metrics/precision(B)"]),
        "recall": float(df.loc[best_idx, "metrics/recall(B)"]),
    }