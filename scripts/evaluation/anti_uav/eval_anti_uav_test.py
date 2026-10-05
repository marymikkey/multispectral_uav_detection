#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the five compared models on the Anti-UAV test set and its subsets
(full test, day, night, hard conditions) with the standard Ultralytics val().

Models: RGB-only, IR-only, RGBT from scratch, RGBT transfer with RGB neck/head,
RGBT transfer with IR neck/head (backbones frozen in both transfer variants).
Metrics: Recall, Precision, AP@0.5, F1 (F1 = 2PR / (P + R)).

Dataset configs:
  full   - configs/train/from_scratch/anti_uav/{rgb,ir,rgbt}_cfg.yaml  (test split)
  day    - configs/test/anti_uav/day/
  night  - configs/test/anti_uav/night/
  meteo  - configs/test/anti_uav/meteo/   (hard conditions)

Results are written to OUT_DIR as results.csv and results.md.
"""

import csv
import sys
from pathlib import Path

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

from scripts.training.shared.utils import check_cuda, check_ultralytics_import

# =========================================================
# CONFIG
# =========================================================

CONFIGS = Path(__file__).resolve().parents[3] / "configs"

PROJECT_DIR = DIPLOMA_ROOT / "train" / "runs" / "antiuav_resync"
OUT_DIR = DIPLOMA_ROOT / "train" / "tests" / "anti_uav_test"

DEVICE = 0
IMGSZ = 640
BATCH = 32

# Per-modality kwargs for val() (same as in the training scripts).
MODALITIES = {
    "rgb": dict(use_simotm="RGB", channels=3),
    "ir": dict(use_simotm="Gray", channels=1),
    "rgbt": dict(use_simotm="RGBT", channels=4, pairs_rgb_ir=["visible", "infrared"]),
}

MODELS = [
    {
        "name": "rgb_only",
        "modality": "rgb",
        "weights": PROJECT_DIR / "yolo11s_rgb_from_scratch_2" / "weights" / "best.pt",
    },
    {
        "name": "ir_only",
        "modality": "ir",
        "weights": PROJECT_DIR / "yolo11s_ir_from_scratch" / "weights" / "best.pt",
    },
    {
        "name": "rgbt_from_scratch",
        "modality": "rgbt",
        "weights": PROJECT_DIR / "yolo11s_rgbt_midfusion_p3_from_scratch" / "weights" / "best.pt",
    },
    {
        "name": "rgbt_rgb_neck_head_freeze",
        "modality": "rgbt",
        "weights": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_rgb_ir_pretrained_rgb_neck_head_freeze_backbones"
        / "weights" / "best.pt",
    },
    {
        "name": "rgbt_ir_neck_head_freeze",
        "modality": "rgbt",
        "weights": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_rgb_ir_pretrained_ir_neck_head_freeze_backbones"
        / "weights" / "best.pt",
    },
]

# subset name -> dataset yaml for a given modality
SUBSETS = {
    "full": lambda m: CONFIGS / "train" / "from_scratch" / "anti_uav" / f"{m}_cfg.yaml",
    "day": lambda m: CONFIGS / "test" / "anti_uav" / "day" / f"{m}_day_cfg.yaml",
    "night": lambda m: CONFIGS / "test" / "anti_uav" / "night" / f"{m}_night_cfg.yaml",
    "meteo": lambda m: CONFIGS / "test" / "anti_uav" / "meteo" / f"{m}_cfg.yaml",
}


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, model_cfg: dict, subset: str, data_yaml: Path) -> dict:
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        plots=False,
        verbose=False,
        project=str(OUT_DIR / "val_runs"),
        name=f"{model_cfg['name']}_{subset}",
        exist_ok=True,
        **MODALITIES[model_cfg["modality"]],
    )

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    ap50 = float(results.box.map50)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "model": model_cfg["name"],
        "subset": subset,
        "recall": recall,
        "precision": precision,
        "ap50": ap50,
        "f1": f1,
    }


def save_results(rows: list) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "subset", "recall", "precision", "ap50", "f1"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Model | Subset | Recall | Precision | AP@0.5 | F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['subset']} | {r['recall']:.3f} | "
            f"{r['precision']:.3f} | {r['ap50']:.3f} | {r['f1']:.3f} |"
        )
    (OUT_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    check_cuda(DEVICE)
    check_ultralytics_import(Path(ultralytics.__file__), EXPECTED_ULTRA_INIT)

    for model_cfg in MODELS:
        if not Path(model_cfg["weights"]).exists():
            raise FileNotFoundError(f"Missing weights for {model_cfg['name']}: {model_cfg['weights']}")
        for subset, make_cfg in SUBSETS.items():
            cfg = make_cfg(model_cfg["modality"])
            if not cfg.exists():
                raise FileNotFoundError(f"Missing dataset config: {cfg}")

    rows = []
    for model_cfg in MODELS:
        print("=" * 80)
        print(f"MODEL: {model_cfg['name']}  ({model_cfg['weights']})")
        print("=" * 80)

        model = YOLO(str(model_cfg["weights"]))

        for subset, make_cfg in SUBSETS.items():
            row = evaluate(model, model_cfg, subset, make_cfg(model_cfg["modality"]))
            rows.append(row)
            print(
                f"[{model_cfg['name']} / {subset}] "
                f"R={row['recall']:.3f} P={row['precision']:.3f} "
                f"AP50={row['ap50']:.3f} F1={row['f1']:.3f}"
            )

        del model
        save_results(rows)

    print(f"\nDONE. Results: {OUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
