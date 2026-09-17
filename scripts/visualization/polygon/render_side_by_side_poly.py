#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render side-by-side (RGB | IR) annotated videos of polygon-finetuned model
predictions on a held-out polygon test interval, for multiple models and
confidence thresholds.

MODELS-driven pipeline; drawing and video-encoding helpers live in
shared_poly_viz.py.
"""

import sys
import tempfile
from pathlib import Path

# ---------- REPO SETUP ----------
REPO_ROOT = Path("/home/YOLOv11-RGBT")
EXPECTED_ULTRA_INIT = REPO_ROOT / "ultralytics" / "__init__.py"

if "/ultralytics" in sys.path:
    sys.path.remove("/ultralytics")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ultralytics
from ultralytics import YOLO

from scripts.training.shared.utils import check_ultralytics_import
from scripts.evaluation.polygon.shared_poly_eval import prepare_tmp_test_interval
from scripts.visualization.polygon.shared_poly_viz import (
    ensure_dir,
    make_video_from_images,
    save_side_by_side_frames,
)

# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = Path(
    "/mnt/datasets/Maria_preprocess/mine_dpl/"
    "data_14may_selected_synced_warped_labeled_full_crop_x287_1248"
)

OUT_ROOT = Path(
    "/home/src/diploma/anti_uav/polygon_prep/"
    "result_videos_side_by_side_crop_x287_1248"
)

TEST_INTERVAL = "2026-05-14_16-49-25_02616_03299"

FPS = 25
IMGSZ = 1248
CONF_VALUES = [0.25, 0.15, 0.10]

PROJECT_DIR = Path("/home/src/diploma/train/runs/antiuav_resync")

# All 5 model variants compared in the polygon validation stage. RGB/IR-only models run on their single modality; the three
# RGBT variants run on the paired visible dir with use_simotm="RGBT".
#
# Each model renders ONE video per confidence value (3 videos/model, same
# CONF_VALUES for everyone). "max_det1" is a simple per-model on/off switch:
# True -> max_det=1 passed to predict(), False -> Ultralytics default
# (no cap). Defaults to True everywhere; flip individual models if needed.
MODELS = [
    {
        "model_name": "rgb_only",
        "base_tag": "yolo11s_rgb_from_scratch_2_poly_finetune",
        "weight_path": PROJECT_DIR
        / "yolo11s_rgb_from_scratch_2_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "RGB",
        "channels": 3,
        "source_key": "visible",
        "max_det1": True,
    },
    {
        "model_name": "ir_only",
        "base_tag": "yolo11s_ir_from_scratch_poly_finetune",
        "weight_path": PROJECT_DIR
        / "yolo11s_ir_from_scratch_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "Gray",
        "channels": 1,
        "source_key": "infrared",
        "max_det1": True,
    },
    {
        "model_name": "rgbt_from_scratch",
        "base_tag": "yolo11s_rgbt_midfusion_p3_from_scratch_poly_finetune",
        "weight_path": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_from_scratch_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "RGBT",
        "channels": 4,
        "source_key": "visible",
        "max_det1": True,
    },
    {
        "model_name": "rgb_neck_head_freeze",
        "base_tag": "yolo11s_rgbt_midfusion_p3_rgb_neck_head_freeze_backbones_poly_finetune",
        "weight_path": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_rgb_neck_head_freeze_backbones_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "RGBT",
        "channels": 4,
        "source_key": "visible",
        "max_det1": True,
    },
    {
        "model_name": "ir_neck_head_freeze",
        "base_tag": "yolo11s_rgbt_midfusion_p3_ir_neck_head_freeze_backbones_poly_finetune",
        "weight_path": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_ir_neck_head_freeze_backbones_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "RGBT",
        "channels": 4,
        "source_key": "visible",
        "max_det1": True,
    },
]


# =========================================================
# RENDERING
# =========================================================

def render_variant(model, tag: str, max_det, use_simotm: str, channels: int, source_key: str):
    print("\n" + "#" * 100)
    print(f"[VARIANT]  {tag}")
    print(f"[MODALITY] use_simotm={use_simotm} channels={channels} source={source_key}")
    print("[IOU]      default Ultralytics")
    print(f"[MAX_DET]  {max_det}")
    print("#" * 100)

    for conf in CONF_VALUES:
        conf_name = f"conf_{conf:.2f}"

        print("\n" + "=" * 90)
        print(f"VARIANT = {tag}")
        print(f"CONF    = {conf}")
        print("=" * 90)

        with tempfile.TemporaryDirectory(prefix=f"rgbt_{tag}_test_") as tmp:
            tmp_root = Path(tmp)

            visible_dir, pairs = prepare_tmp_test_interval(DATA_ROOT, TEST_INTERVAL, tmp_root)

            if not pairs:
                print("[WARN] No paired frames found")
                continue

            # Both dirs are always symlinked by prepare_tmp_test_interval;
            # derive infrared_dir from the pairs rather than assuming its
            # internal folder name. Note: predictions always run on the
            # model's own modality dir, but boxes are drawn on BOTH panels
            # of the side-by-side frame regardless (see save_side_by_side_frames).
            infrared_dir = pairs[0][1].parent
            source_dir = visible_dir if source_key == "visible" else infrared_dir

            print(f"[INFO] Paired frames: {len(pairs)}")
            print(f"[INFO] source dir: {source_dir}")

            predict_kwargs = dict(
                source=str(source_dir),
                imgsz=IMGSZ,
                show=False,
                save=False,
                stream=True,
                use_simotm=use_simotm,
                channels=channels,
                conf=conf,
                verbose=True,
            )

            if max_det is not None:
                predict_kwargs["max_det"] = max_det

            results = list(model.predict(**predict_kwargs))

            frames_dir = tmp_root / "side_by_side_frames"

            saved = save_side_by_side_frames(
                results=results,
                pairs=pairs,
                out_frames_dir=frames_dir,
            )

            if saved == 0:
                print(f"[WARN] No frames saved for {tag}, conf={conf}")
                continue

            out_dir = OUT_ROOT / tag / conf_name
            ensure_dir(out_dir)

            out_video = out_dir / f"{tag}_{TEST_INTERVAL}_{conf_name}_side_by_side.mp4"

            make_video_from_images(
                image_dir=frames_dir,
                out_video=out_video,
                fps=FPS,
            )

            print(f"[OK] Saved video: {out_video}")


# =========================================================
# MAIN
# =========================================================

def main():
    check_ultralytics_import(Path(ultralytics.__file__), EXPECTED_ULTRA_INIT)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Missing dataset root: {DATA_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] {DATA_ROOT}")
    print(f"[TEST] {TEST_INTERVAL}")
    print(f"[IMGSZ] {IMGSZ}")
    print(f"[OUT] {OUT_ROOT}")

    for model_cfg in MODELS:
        model_name = model_cfg["model_name"]
        weight_path = Path(model_cfg["weight_path"])

        if not weight_path.exists():
            raise FileNotFoundError(f"Missing weights for {model_name}: {weight_path}")

        print("\n" + "=" * 120)
        print(f"LOADING MODEL: {model_name}")
        print(f"WEIGHTS: {weight_path}")
        print("=" * 120)

        model = YOLO(str(weight_path))

        max_det = 1 if model_cfg["max_det1"] else None
        tag = f"{model_cfg['base_tag']}_maxdet1" if model_cfg["max_det1"] else f"{model_cfg['base_tag']}_default"

        render_variant(
            model,
            tag=tag,
            max_det=max_det,
            use_simotm=model_cfg["use_simotm"],
            channels=model_cfg["channels"],
            source_key=model_cfg["source_key"],
        )

        del model

    print("\nDONE")
    print(f"All videos saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
