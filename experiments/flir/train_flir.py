#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-domain check on FLIR ADAS (aligned RGB-T pairs, 3 classes): trains the RGB,
IR or RGBT mid-fusion YOLO11n model with the same settings.

    python -m experiments.flir.train_flir rgb|ir|rgbt

RGB and IR use the standard 3-channel yolo11 (IR frames are read as 3-channel
images); RGBT uses the P3 mid-fusion model with 4 input channels.
"""

import argparse
import os
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

# "11n" in the model name selects scale n (the file itself is the shared yolo11*.yaml).
MODALITIES = {
    "rgb": dict(
        data="configs/flir/rgb_cfg.yaml",
        model="ultralytics/cfg/models/11/yolo11n.yaml",
        epochs=200,
        extra={},
    ),
    "ir": dict(
        data="configs/flir/ir_cfg.yaml",
        model="ultralytics/cfg/models/11/yolo11n.yaml",
        epochs=300,
        extra={},
    ),
    "rgbt": dict(
        data="configs/flir/rgbt_cfg.yaml",
        model="ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-P3.yaml",
        epochs=300,
        extra=dict(channels=4, use_simotm="RGBT", pairs_rgb_ir=["visible", "infrared"]),
    ),
}

TRAIN_ARGS = dict(
    patience=20,
    batch=32,
    imgsz=640,
    workers=max(2, min(8, (os.cpu_count() or 8) // 2)),
    amp=True,
    pretrained=True,
    optimizer="auto",
    save=True,
    save_period=5,
    plots=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modality", choices=MODALITIES)
    modality = parser.parse_args().modality
    cfg = MODALITIES[modality]

    check_cuda(DEVICE)
    check_ultralytics_import(Path(ultralytics.__file__), EXPECTED_ULTRA_INIT)

    data_yaml = Path(cfg["data"])
    if not data_yaml.exists():
        raise FileNotFoundError(f"Config not found: {data_yaml}")

    model = YOLO(cfg["model"])
    model.train(
        task="detect",
        data=str(data_yaml),
        epochs=cfg["epochs"],
        device=DEVICE,
        project=str(PROJECT_DIR),
        name=f"yolo11n_{modality}",
        exist_ok=False,
        **TRAIN_ARGS,
        **cfg["extra"],
    )


if __name__ == "__main__":
    main()
