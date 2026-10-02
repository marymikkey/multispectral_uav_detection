#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv11 RGB pretrain on unpaired visible-spectrum UAV datasets.
"""

import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime

from scripts.paths import YOLO_REPO_ROOT, DIPLOMA_ROOT

# ---------- REPO SETUP ----------
REPO_ROOT = YOLO_REPO_ROOT
EXPECTED_ULTRA_INIT = REPO_ROOT / "ultralytics" / "__init__.py"

if "/ultralytics" in sys.path:
    sys.path.remove("/ultralytics")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ultralytics
from ultralytics import YOLO

import yaml
import torch
from scripts.training.shared.utils import (
    check_cuda,
    check_ultralytics_import,
    count_split_images,
    format_seconds,
    get_best_epoch_metrics,
)
from scripts.training.shared.summary import save_run_markdown

# =========================================================
# CONFIGURATION
# =========================================================

DATA_YAML = Path("configs/train/pretrain/rgb_cfg.yaml")
PROJECT_DIR = DIPLOMA_ROOT / "pretrain" / "runs"
RUN_NAME = "yolo11s_rgb_pretrain"

DEVICE = 0
MODEL_CFG = "ultralytics/cfg/models/11/yolo11s.yaml"

# ---------- Training hyperparameters ----------
TRAIN_ARGS = dict(
    cache=False,
    imgsz=640,
    epochs=100,
    batch=32,
    workers=4,
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
    project=str(PROJECT_DIR),
    name=RUN_NAME,
    exist_ok=False,
)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv11 RGB Pretrain")
    print("=" * 60)

    check_cuda(DEVICE)
    check_ultralytics_import(Path(ultralytics.__file__), EXPECTED_ULTRA_INIT)

    torch.cuda.empty_cache()
    gc.collect()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Config not found: {DATA_YAML}")

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    print(f"\n[INFO] Data config: {DATA_YAML}")
    print(f"  train: {data_config.get('train', 'NOT SET')}")
    print(f"  val:   {data_config.get('val', 'NOT SET')}")
    print(f"  nc:    {data_config.get('nc', 'NOT SET')}")
    print(f"  names: {data_config.get('names', 'NOT SET')}")

    train_images = count_split_images(DATA_YAML, "train")
    val_images = count_split_images(DATA_YAML, "val")

    if train_images == 0:
        raise RuntimeError("No training images found in dataset!")

    print(f"\n[DATA] train_images: {train_images}")
    print(f"[DATA] val_images:   {val_images}")

    print(f"\n[MODEL] Loading: {MODEL_CFG}")
    model = YOLO(MODEL_CFG)

    start_dt = datetime.now()
    start_ts = time.time()

    completed = False
    stop_reason = "finished"

    try:
        print("\n[INFO] Starting pretrain...")
        print(f"  Run name: {RUN_NAME}")
        print(f"  Project dir: {PROJECT_DIR}")
        print("=" * 60)

        model.train(
            data=str(DATA_YAML),
            single_cls=True,
            channels=3,
            device=DEVICE,
            **TRAIN_ARGS,
        )
        completed = True

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        raise

    except Exception as e:
        stop_reason = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        raise

    finally:
        end_dt = datetime.now()
        end_ts = time.time()
        duration = end_ts - start_ts

        # Безопасно получаем save_dir
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
            save_dir = Path(model.trainer.save_dir)
        else:
            save_dir = PROJECT_DIR / RUN_NAME
            print(f"[WARNING] Could not retrieve trainer.save_dir, using fallback: {save_dir}")
            save_dir.mkdir(parents=True, exist_ok=True)

        results_csv = save_dir / "results.csv"
        best_metrics = get_best_epoch_metrics(results_csv)

        if best_metrics:
            best_epoch = best_metrics["epoch"]
            best_mAP50 = best_metrics["mAP50"]
            best_mAP50_95 = best_metrics["mAP50_95"]
            best_precision = best_metrics["precision"]
            best_recall = best_metrics["recall"]
        else:
            best_epoch = "N/A"
            best_mAP50 = 0.0
            best_mAP50_95 = 0.0
            best_precision = 0.0
            best_recall = 0.0

        info = {
            "run_name": RUN_NAME,
            "resume_mode": False,
            "completed": completed,
            "stop_reason": stop_reason,
            "best_epoch": best_epoch,
            "best_mAP50": best_mAP50,
            "best_mAP50_95": best_mAP50_95,
            "best_precision": best_precision,
            "best_recall": best_recall,
            "model_cfg": MODEL_CFG,
            "data_yaml": str(DATA_YAML),
            "actual_save_dir": str(save_dir),
            "channels": 3,
            "single_cls": True,
            "device": DEVICE,
            "train_images": train_images,
            "val_images": val_images,
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": duration,
            "duration_hms": format_seconds(duration),
        }
        info.update(TRAIN_ARGS)

        save_run_markdown(save_dir, info)

        print("\n" + "=" * 60)
        print("PRETRAIN COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {save_dir}")
        print(f"Best mAP50-95: {best_mAP50_95:.4f} (epoch {best_epoch})")
        print(f"Run summary: {save_dir / 'run_summary.md'}")
