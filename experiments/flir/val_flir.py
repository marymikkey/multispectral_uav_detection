#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate a model trained by train_flir.py on the held-out FLIR split.

    python -m experiments.flir.val_flir rgb|ir|rgbt

Writes mAP50, mAP50-95, precision and recall to <run>/val_metrics.txt.
"""

import argparse
import sys
from pathlib import Path

from scripts.paths import YOLO_REPO_ROOT, DIPLOMA_ROOT

REPO_ROOT = YOLO_REPO_ROOT
EXPECTED_ULTRA_INIT = REPO_ROOT / "ultralytics" / "__init__.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ultralytics
from ultralytics import YOLO

from scripts.training.shared.utils import check_cuda, check_ultralytics_import

PROJECT_DIR = DIPLOMA_ROOT / "experiments" / "flir"
DEVICE = 0
IMGSZ = 640
BATCH = 32

DATA = {
    "rgb": "configs/flir/rgb_cfg.yaml",
    "ir": "configs/flir/ir_cfg.yaml",
    "rgbt": "configs/flir/rgbt_cfg.yaml",
}
EXTRA = {
    "rgb": {},
    "ir": {},
    "rgbt": dict(channels=4, use_simotm="RGBT", pairs_rgb_ir=["visible", "infrared"]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modality", choices=DATA)
    modality = parser.parse_args().modality

    check_cuda(DEVICE)
    check_ultralytics_import(Path(ultralytics.__file__), EXPECTED_ULTRA_INIT)

    run_dir = PROJECT_DIR / f"yolo11n_{modality}"
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)

    metrics = YOLO(str(weights)).val(
        data=DATA[modality],
        device=DEVICE,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(run_dir),
        name="val",
        exist_ok=True,
        plots=True,
        **EXTRA[modality],
    )

    out = run_dir / "val_metrics.txt"
    out.write_text(
        f"weights: {weights}\n"
        f"mAP50: {metrics.box.map50:.4f}\n"
        f"mAP50-95: {metrics.box.map:.4f}\n"
        f"precision: {metrics.box.mp:.4f}\n"
        f"recall: {metrics.box.mr:.4f}\n",
        encoding="utf-8",
    )
    print(f"Metrics: {out}")


if __name__ == "__main__":
    main()
