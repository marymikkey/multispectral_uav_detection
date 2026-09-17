#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune RGBT models on polygon experimental data (crop_x287_1248):
- rgb_neck_head_freeze_backbones: continues from the MCF transfer checkpoint
  with neck/head from the RGB pretrain (RGB/IR backbones frozen, same
  freeze layers as the transfer stage).
- ir_neck_head_freeze_backbones: continues from the MCF transfer checkpoint
  with neck/head from the IR pretrain (RGB/IR backbones frozen, same
  freeze layers as the transfer stage).
- from_scratch_baseline: continues from the RGBT from-scratch checkpoint,
  no freeze.

All three experiments run sequentially and are recorded in a shared
pipeline summary. Together with train_rgb_ir_poly_finetune.py's
rgb_from_scratch / ir_from_scratch, this covers all 5 model variants
compared in the polygon validation stage:
IR-only, RGB-only, RGBT from scratch, RGBT transfer w/ RGB neck/head,
RGBT transfer w/ IR neck/head. Polygon validation setup (1920x1080 -> center-crop+pad -> 1248x1248).
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

import yaml

# ---------- REPO SETUP ----------
REPO_ROOT = Path("/home/YOLOv11-RGBT")
EXPECTED_ULTRA_INIT = REPO_ROOT / "ultralytics" / "__init__.py"

if "/ultralytics" in sys.path:
    sys.path.remove("/ultralytics")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ultralytics
from ultralytics import YOLO

import torch
from scripts.training.shared.utils import (
    check_cuda,
    check_ultralytics_import,
    count_split_images,
    format_seconds,
    get_best_epoch_metrics,
)
from scripts.training.polygon_finetune.shared_polyfinetune import (
    copy_init_weight,
    save_run_markdown,
    save_pipeline_summary,
)

# =========================================================
# CONFIGURATION
# =========================================================

DATA_YAML = Path("/home/src/diploma/train/configs/poly_finetune/rgbt_poly_finetune_crop_x287_1248.yaml")

NEW_DATA_ROOT = Path(
    "/mnt/datasets/Maria_preprocess/mine_dpl/"
    "data_14may_selected_synced_warped_labeled_full_crop_x287_1248"
)

PROJECT_DIR = Path("/home/src/diploma/train/runs/antiuav_resync")
INIT_COPY_DIR = PROJECT_DIR / "poly_finetune_init"

DEVICE = 0

# RGB backbone: layers 2-6. IR backbone: layers 8-12. Same as the transfer stage.
FREEZE_BACKBONES = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

EXPERIMENTS = [
    {
        "tag": "rgb_neck_head_freeze_backbones",
        "run_name": "yolo11s_rgbt_midfusion_p3_rgb_neck_head_freeze_backbones_poly_finetune_crop_x287_1248",
        "source_checkpoint": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_rgb_ir_pretrained_rgb_neck_head_freeze_backbones"
        / "weights"
        / "best.pt",
        "init_copy_name": "yolo11s_rgbt_midfusion_p3_rgb_neck_head_freeze_backbones_best_copy_crop_x287_1248.pt",
        "freeze": FREEZE_BACKBONES,
    },
    {
        "tag": "ir_neck_head_freeze_backbones",
        "run_name": "yolo11s_rgbt_midfusion_p3_ir_neck_head_freeze_backbones_poly_finetune_crop_x287_1248",
        "source_checkpoint": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_rgb_ir_pretrained_ir_neck_head_freeze_backbones"
        / "weights"
        / "best.pt",
        "init_copy_name": "yolo11s_rgbt_midfusion_p3_ir_neck_head_freeze_backbones_best_copy_crop_x287_1248.pt",
        "freeze": FREEZE_BACKBONES,
    },
    {
        "tag": "from_scratch_baseline",
        "run_name": "yolo11s_rgbt_midfusion_p3_from_scratch_poly_finetune_crop_x287_1248",
        "source_checkpoint": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_from_scratch"
        / "weights"
        / "best.pt",
        "init_copy_name": "yolo11s_rgbt_midfusion_p3_from_scratch_best_copy_crop_x287_1248.pt",
        "freeze": None,
    },
]

# ---------- Training hyperparameters ----------
# Note: lr0 is lower than the other stages (1e-4 vs 1e-3) -- this is a
# fine-tune on a small polygon dataset, not a from-scratch/pretrain run.
TRAIN_ARGS = dict(
    cache=False,
    imgsz=1248,
    epochs=100,
    batch=32,
    workers=4,
    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.1,
    cos_lr=True,
    patience=30,
    save=True,
    save_period=5,
    plots=False,
    val=True,
    verbose=True,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.3,
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
)


# =========================================================
# FUNCTIONS
# =========================================================

def make_new_yaml():
    if not NEW_DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {NEW_DATA_ROOT}")

    data = {
        "path": str(NEW_DATA_ROOT),
        "train": [
            "2026-05-14_16-49-25_01490_01640/visible/images",
            "2026-05-14_16-49-25_04750_05210/visible/images",
            "2026-05-14_16-49-25_06618_06952/visible/images",
            "2026-05-14_17-02-50_01290_01717/visible/images",
            "2026-05-14_17-25-12_01924_02085/visible/images",
        ],
        "val": [
            "2026-05-14_16-49-25_02616_03299/visible/images",
        ],
        "test": [
            "2026-05-14_16-49-25_02616_03299/visible/images",
        ],
        "nc": 1,
        "names": ["drone"],
        "pairs_rgb_ir": ["visible", "infrared"],
        "ch": 4,
    }

    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] Saved data yaml: {DATA_YAML}")


def train_one_experiment(exp: dict, train_images: int, val_images: int):
    tag = exp["tag"]
    run_name = exp["run_name"]
    source_checkpoint = Path(exp["source_checkpoint"])
    init_pt = INIT_COPY_DIR / exp["init_copy_name"]
    freeze_layers = exp["freeze"]

    print("\n" + "#" * 90)
    print(f"[START EXPERIMENT] {tag}")
    print(f"[RUN_NAME] {run_name}")
    print(f"[SOURCE] {source_checkpoint}")
    print(f"[INIT] {init_pt}")
    print(f"[FREEZE] {freeze_layers}")
    print("#" * 90)

    copy_init_weight(source_checkpoint, init_pt, INIT_COPY_DIR)

    torch.cuda.empty_cache()
    gc.collect()

    model = YOLO(str(init_pt))

    start_dt = datetime.now()
    start_ts = time.time()

    completed = False
    stop_reason = "finished"

    try:
        train_kwargs = dict(
            data=str(DATA_YAML),
            use_simotm="RGBT",
            channels=4,
            pairs_rgb_ir=["visible", "infrared"],
            single_cls=True,
            device=DEVICE,
            project=str(PROJECT_DIR),
            name=run_name,
            exist_ok=False,
            **TRAIN_ARGS,
        )
        if freeze_layers is not None:
            train_kwargs["freeze"] = freeze_layers

        model.train(**train_kwargs)
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
        duration = time.time() - start_ts

        if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir"):
            save_dir = Path(model.trainer.save_dir)
        else:
            save_dir = PROJECT_DIR / run_name
            print(f"[WARNING] Could not retrieve trainer.save_dir, using fallback: {save_dir}")
            save_dir.mkdir(parents=True, exist_ok=True)

        best_metrics = get_best_epoch_metrics(save_dir / "results.csv") or {}

        info = {
            "tag": tag,
            "run_name": run_name,
            "completed": completed,
            "stop_reason": stop_reason,
            "source_checkpoint": str(source_checkpoint),
            "init_checkpoint": str(init_pt),
            "data_yaml": str(DATA_YAML),
            "dataset_root": str(NEW_DATA_ROOT),
            "device": DEVICE,
            "optimizer": "AdamW",
            "single_cls": True,
            "use_simotm": "RGBT",
            "channels": 4,
            "pairs_rgb_ir": ["visible", "infrared"],
            "freeze": freeze_layers,
            "train_images": train_images,
            "val_images": val_images,
            "best_epoch": best_metrics.get("epoch", "N/A"),
            "best_mAP50": best_metrics.get("mAP50", 0.0),
            "best_mAP50_95": best_metrics.get("mAP50_95", 0.0),
            "best_precision": best_metrics.get("precision", 0.0),
            "best_recall": best_metrics.get("recall", 0.0),
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_hms": format_seconds(duration),
            "save_dir": str(save_dir),
        }
        info.update(TRAIN_ARGS)

        save_run_markdown(save_dir, info)

        torch.cuda.empty_cache()
        gc.collect()

        print("\n" + "=" * 90)
        print(f"[DONE] {tag}")
        print(f"Results saved to: {save_dir}")
        print(f"Best mAP50-95: {info['best_mAP50_95']:.4f} epoch {info['best_epoch']}")
        print("=" * 90)

    return info


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 90)
    print("YOLOv11-RGBT sequential fine-tune on crop_x287_1248 dataset")
    print("=" * 90)

    check_ultralytics_import(Path(ultralytics.__file__), EXPECTED_ULTRA_INIT)
    check_cuda(DEVICE)
    make_new_yaml()

    train_images = count_split_images(DATA_YAML, "train")
    val_images = count_split_images(DATA_YAML, "val")

    print(f"\n[DATA] train_images: {train_images}")
    print(f"[DATA] val_images:   {val_images}")

    if train_images == 0:
        raise RuntimeError("No training images found!")
    if val_images == 0:
        raise RuntimeError("No validation images found!")

    print("\n[EXPERIMENTS]")
    for exp in EXPERIMENTS:
        print(f"  - {exp['tag']}")
        print(f"    source: {exp['source_checkpoint']}")
        print(f"    run:    {exp['run_name']}")
        print(f"    freeze: {exp['freeze']}")

        if not Path(exp["source_checkpoint"]).exists():
            raise FileNotFoundError(f"Missing source weights: {exp['source_checkpoint']}")

    infos = []
    for exp in EXPERIMENTS:
        info = train_one_experiment(exp=exp, train_images=train_images, val_images=val_images)
        infos.append(info)

    save_pipeline_summary(
        infos,
        out_path=PROJECT_DIR / "poly_finetune_crop_x287_1248_rgbt_models_summary.md",
        title="Poly fine-tune crop_x287_1248: RGB neck/head + IR neck/head + from scratch",
    )

    print("\n" + "=" * 90)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 90)
    for info in infos:
        print(
            f"{info['tag']}: "
            f"best mAP50-95={info['best_mAP50_95']:.4f}, "
            f"mAP50={info['best_mAP50']:.4f}, "
            f"P={info['best_precision']:.4f}, "
            f"R={info['best_recall']:.4f}, "
            f"epoch={info['best_epoch']}"
        )


if __name__ == "__main__":
    main()
