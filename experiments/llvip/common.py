#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Settings shared by train_llvip.py and val_llvip.py.

All three modalities use the same training settings: same batch, same augmentation,
same schedule. The RGBT model reads the colour visible frame plus the thermal frame,
the RGB model the colour visible frame, the IR model the thermal frame.

Data: the split built by scripts/data_preparation/llvip/make_llvip_split.py
(train_split / val_split for training and model selection, the official test set only
for the final evaluation).
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.paths import YOLO_REPO_ROOT, DIPLOMA_ROOT, LLVIP_DATA_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "llvip"
SPLIT_DIR = REPO_ROOT / "experiments" / "llvip" / "splits"

PROJECT_DIR = DIPLOMA_ROOT / "train" / "runs" / "llvip_split"
TEST_DIR = DIPLOMA_ROOT / "train" / "tests" / "llvip_split_test"
GENERATED_CFG_DIR = DIPLOMA_ROOT / "train" / "configs" / "llvip_split"

DEFAULT_SEEDS = [0, 1, 2]

MODALITIES = {
    "rgb": dict(
        model_cfg="ultralytics/cfg/models/11/yolo11s.yaml",
        data_cfg="rgb_cfg.yaml",
        load=dict(use_simotm="RGB", channels=3),
    ),
    "ir": dict(
        model_cfg="ultralytics/cfg/models/11/yolo11s-gray.yaml",
        data_cfg="ir_cfg.yaml",
        load=dict(use_simotm="Gray", channels=1),
    ),
    "rgbt": dict(
        model_cfg="ultralytics/cfg/models/11-RGBT/yolo11s-RGBT-midfusion-P3.yaml",
        data_cfg="rgbt_cfg.yaml",
        load=dict(use_simotm="RGBT", channels=4, pairs_rgb_ir=["visible", "infrared"]),
    ),
}

# Same for every modality. Schedule and optimiser as in the Anti-UAV from-scratch stage.
# Augmentation: the Ultralytics defaults used for the Anti-UAV RGB and IR models. The fork
# applies hue/saturation to the visible channels and brightness to the thermal channel,
# so the same values work for 1, 3 and 4 channels.
TRAIN_ARGS = dict(
    cache=False,
    imgsz=640,
    epochs=100,
    batch=32,
    workers=8,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.1,
    cos_lr=True,
    patience=30,
    save=True,
    save_period=5,
    plots=True,
    val=True,
    verbose=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    bgr=0.0,
    degrees=20,
    translate=0.2,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.8,
    close_mosaic=12,
    mixup=0.0,
    deterministic=True,
    single_cls=True,
    exist_ok=False,
)


def setup_fork():
    """Put the YOLOv11-RGBT fork first on sys.path and check that it is the one imported."""
    if str(YOLO_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLO_REPO_ROOT))
    import ultralytics
    from scripts.training.shared.utils import check_ultralytics_import

    check_ultralytics_import(Path(ultralytics.__file__), YOLO_REPO_ROOT / "ultralytics" / "__init__.py")


def run_name(modality: str, seed: int) -> str:
    return f"yolo11s_{modality}_llvip_seed{seed}"


def write_data_yaml(modality: str) -> Path:
    """Copy configs/llvip/<modality>_cfg.yaml with `path` set from LLVIP_DATA_ROOT."""
    with open(CONFIG_DIR / MODALITIES[modality]["data_cfg"], "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = str(LLVIP_DATA_ROOT / "LLVIP_yolo")
    for split in ("train", "val", "test"):
        if not (Path(data["path"]) / data[split]).is_dir():
            raise FileNotFoundError(
                f"{Path(data['path']) / data[split]} not found. "
                "Run scripts.data_preparation.llvip.make_llvip_split first."
            )
    GENERATED_CFG_DIR.mkdir(parents=True, exist_ok=True)
    out = GENERATED_CFG_DIR / MODALITIES[modality]["data_cfg"]
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def git_commit(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_run_meta(save_dir: Path, info: dict) -> None:
    """Write run_meta.json next to Ultralytics' own args.yaml and results.csv."""
    info = dict(info)
    info["repo_commit"] = git_commit(REPO_ROOT)
    info["fork_commit"] = git_commit(YOLO_REPO_ROOT)
    info["split_lists"] = {
        s: str(SPLIT_DIR / f"{s}.txt") for s in ("train", "val", "test")
    }
    (save_dir / "run_meta.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
