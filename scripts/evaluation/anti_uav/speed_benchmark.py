#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-pair inference speed test for the RGB, IR and RGBT (mid-fusion) models.

Each model is loaded once, warmed up, then predict() is timed on one image
(RGB / IR) or one RGB+IR pair (RGBT) N_RUNS times. The reported speed is
FPS from the minimum single-run time. Default device is CPU (the thesis numbers
were measured on an Intel NUC); set DEVICE = 0 for GPU.

Results are written to OUT_DIR / speed_<device>.md.
"""

import gc
import shutil
import sys
import time
from pathlib import Path

from scripts.paths import YOLO_REPO_ROOT, DIPLOMA_ROOT, ANTIUAV_DATA_ROOT

# ---------- REPO SETUP ----------
REPO_ROOT = YOLO_REPO_ROOT
EXPECTED_ULTRA_INIT = REPO_ROOT / "ultralytics" / "__init__.py"

if "/ultralytics" in sys.path:
    sys.path.remove("/ultralytics")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import ultralytics
from ultralytics import YOLO

from scripts.training.shared.utils import check_ultralytics_import

# =========================================================
# CONFIG
# =========================================================

PROJECT_DIR = DIPLOMA_ROOT / "train" / "runs" / "antiuav_resync"
OUT_DIR = DIPLOMA_ROOT / "train" / "tests" / "speed_single_pair"

SCENE_DIR = ANTIUAV_DATA_ROOT / "ANTI-UAV_synced_lite_by_scene" / "test" / "20190925_111757_1_6"
FRAME_NAME = "20190925_111757_1_6_ir000004_rgb000013.jpg"
RGB_IMAGE = SCENE_DIR / "visible" / "images" / FRAME_NAME
IR_IMAGE = SCENE_DIR / "infrared" / "images" / FRAME_NAME

DEVICE = "cpu"
IMGSZ = 640
N_RUNS = 10
CONF = 0.25
IOU = 0.7

MODALITIES = {
    "rgb": {
        "weights": PROJECT_DIR / "yolo11s_rgb_from_scratch" / "weights" / "best.pt",
        "kwargs": dict(channels=3),
    },
    "ir": {
        "weights": PROJECT_DIR / "yolo11s_ir_from_scratch" / "weights" / "best.pt",
        "kwargs": dict(channels=1, use_simotm="Gray"),
    },
    "rgbt": {
        "weights": PROJECT_DIR / "yolo11s_rgbt_midfusion_p3_from_scratch" / "weights" / "best.pt",
        "kwargs": dict(channels=4, use_simotm="RGBT"),
    },
}


# =========================================================
# HELPERS
# =========================================================

def prepare_rgbt_pair(rgb_path: Path, ir_path: Path, tmp_root: Path) -> Path:
    """The RGBT model reads the IR image from the sibling infrared/ folder."""
    vis_dir = tmp_root / "visible" / "images"
    ir_dir = tmp_root / "infrared" / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(rgb_path, vis_dir / "pair.jpg")
    shutil.copy2(ir_path, ir_dir / "pair.jpg")
    return vis_dir / "pair.jpg"


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_predict(model: YOLO, source: Path, extra_kwargs: dict) -> dict:
    kwargs = dict(
        device=DEVICE, imgsz=IMGSZ, conf=CONF, iou=IOU,
        verbose=False, save=False, plots=False, **extra_kwargs,
    )

    model.predict(source=str(source), **kwargs)  # warmup, not timed
    sync()

    times = []
    for i in range(N_RUNS):
        sync()
        start = time.perf_counter()
        model.predict(source=str(source), **kwargs)
        sync()
        times.append(time.perf_counter() - start)
        print(f"  run {i + 1}/{N_RUNS}: {times[-1]:.6f} s")

    best = min(times)
    return {
        "min_sec": best,
        "mean_sec": sum(times) / len(times),
        "fps": 1.0 / best if best > 0 else 0.0,
        "times": times,
    }


def save_report(results: dict) -> Path:
    lines = [
        "# Single-pair inference speed test",
        "",
        f"- RGB image: `{RGB_IMAGE}`",
        f"- IR image: `{IR_IMAGE}`",
        f"- Timed runs per model: {N_RUNS} (warmup not included)",
        "- Metric: FPS from the minimum single-run predict time",
        f"- Device: {DEVICE}, image size: {IMGSZ}, conf: {CONF}, IoU: {IOU}",
        "",
        "| Model | Min predict time, s | FPS | Mean time, s |",
        "|---|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['min_sec']:.6f} | {r['fps']:.3f} | {r['mean_sec']:.6f} |")
    lines.append("")

    out = OUT_DIR / f"speed_{DEVICE}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# =========================================================
# MAIN
# =========================================================

def main():
    check_ultralytics_import(Path(ultralytics.__file__), EXPECTED_ULTRA_INIT)

    for p in [RGB_IMAGE, IR_IMAGE] + [m["weights"] for m in MODALITIES.values()]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_source = prepare_rgbt_pair(RGB_IMAGE, IR_IMAGE, OUT_DIR / "tmp_rgbt_pair")
    sources = {"rgb": RGB_IMAGE, "ir": IR_IMAGE, "rgbt": pair_source}

    results = {}
    for name, cfg in MODALITIES.items():
        print("=" * 80)
        print(f"[{name}] {cfg['weights']}")
        print("=" * 80)

        cleanup_memory()
        model = YOLO(str(cfg["weights"]))
        results[name] = timed_predict(model, sources[name], cfg["kwargs"])
        print(f"[{name}] min={results[name]['min_sec']:.6f}s  FPS={results[name]['fps']:.3f}")
        del model
        cleanup_memory()

    print(f"\nReport: {save_report(results)}")


if __name__ == "__main__":
    main()
