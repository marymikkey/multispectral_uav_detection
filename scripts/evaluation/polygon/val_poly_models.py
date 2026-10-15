#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate the five polygon-finetuned models on the held-out field interval with the
standard Ultralytics val() (the field-validation results table).

Models: RGB-only, IR-only, RGBT from scratch, RGBT transfer with IR neck/head,
RGBT transfer with RGB neck/head (backbones frozen in both transfer variants).
Metrics: Recall, Precision, AP@0.5, F1 (F1 = 2PR / (P + R)).

The dataset yaml for the validation interval is written to OUT_DIR, results are
saved there as results.csv and results.md.
"""

import csv
import sys
from pathlib import Path

from scripts.paths import YOLO_REPO_ROOT, DIPLOMA_ROOT, POLYGON_DATA_ROOT

# ---------- REPO SETUP ----------
REPO_ROOT = YOLO_REPO_ROOT
EXPECTED_ULTRA_INIT = REPO_ROOT / "ultralytics" / "__init__.py"

if "/ultralytics" in sys.path:
    sys.path.remove("/ultralytics")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ultralytics
from ultralytics import YOLO

from scripts.training.shared.utils import check_cuda, check_ultralytics_import

# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = POLYGON_DATA_ROOT / "data_14may_selected_synced_warped_labeled_full_crop_x287_1248"
VAL_INTERVAL = "2026-05-14_16-49-25_02616_03299"

PROJECT_DIR = DIPLOMA_ROOT / "train" / "runs" / "antiuav_resync"
OUT_DIR = DIPLOMA_ROOT / "train" / "tests" / "polygon_val"

DEVICE = 0
IMGSZ = 1248
BATCH = 16

RGBT = dict(use_simotm="RGBT", channels=4, pairs_rgb_ir=["visible", "infrared"])

# name, run folder, image modality, val() kwargs (same input modes as in fine-tuning)
MODELS = [
    ("rgbt_from_scratch", "yolo11s_rgbt_midfusion_p3_from_scratch", "visible", RGBT),
    ("rgbt_ir_neck_head_freeze", "yolo11s_rgbt_midfusion_p3_ir_neck_head_freeze_backbones", "visible", RGBT),
    ("rgbt_rgb_neck_head_freeze", "yolo11s_rgbt_midfusion_p3_rgb_neck_head_freeze_backbones", "visible", RGBT),
    ("ir_only", "yolo11s_ir_from_scratch", "infrared", dict(use_simotm="Gray", channels=1)),
    ("rgb_only", "yolo11s_rgb_from_scratch", "visible", dict(use_simotm="RGB", channels=3)),
]
RUN_SUFFIX = "_poly_finetune_crop_x287_1248"


# =========================================================
# HELPERS
# =========================================================

def write_data_yaml(modality: str, pairs: bool) -> Path:
    images = f"{VAL_INTERVAL}/{modality}/images"
    lines = [
        f"path: {DATA_ROOT}",
        f"train: {images}",
        f"val: {images}",
        "nc: 1",
        "names:",
        "  0: drone",
    ]
    if pairs:
        lines += ["ch: 4", "pairs_rgb_ir:", "  - visible", "  - infrared"]

    path = OUT_DIR / f"{'rgbt' if pairs else modality}_val.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_results(rows: list) -> None:
    with open(OUT_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "recall", "precision", "ap50", "f1"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Model | Recall | Precision | AP@0.5 | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['recall']:.3f} | {r['precision']:.3f} | {r['ap50']:.3f} | {r['f1']:.3f} |"
        )
    (OUT_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main():
    check_cuda(DEVICE)
    check_ultralytics_import(Path(ultralytics.__file__), EXPECTED_ULTRA_INIT)

    if not (DATA_ROOT / VAL_INTERVAL).exists():
        raise FileNotFoundError(DATA_ROOT / VAL_INTERVAL)
    for _, run, _, _ in MODELS:
        weights = PROJECT_DIR / f"{run}{RUN_SUFFIX}" / "weights" / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(weights)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, run, modality, kwargs in MODELS:
        weights = PROJECT_DIR / f"{run}{RUN_SUFFIX}" / "weights" / "best.pt"
        data_yaml = write_data_yaml(modality, pairs=kwargs["channels"] == 4)

        print("=" * 80)
        print(f"MODEL: {name}  ({weights})")
        print("=" * 80)

        results = YOLO(str(weights)).val(
            data=str(data_yaml),
            split="val",
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            plots=False,
            verbose=False,
            project=str(OUT_DIR / "val_runs"),
            name=name,
            exist_ok=True,
            **kwargs,
        )

        precision = float(results.box.mp)
        recall = float(results.box.mr)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append({
            "model": name,
            "recall": recall,
            "precision": precision,
            "ap50": float(results.box.map50),
            "f1": f1,
        })
        print(f"[{name}] R={recall:.3f} P={precision:.3f} AP50={rows[-1]['ap50']:.3f} F1={f1:.3f}")

    save_results(rows)
    print(f"\nDONE. Results: {OUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
