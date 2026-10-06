#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate all 5 polygon-finetuned model variants (IR-only, RGB-only, RGBT from scratch, RGBT transfer w/ RGB neck/head,
RGBT transfer w/ IR neck/head) on a held-out polygon test interval:
computes TP/FP/FN, Precision/Recall/F1 and mean best IoU per
(model, prediction-variant, confidence) combination and ranks them.

RGB/IR-only models run on their single modality (own image dir, own
use_simotm/channels); the three RGBT variants all run on the paired
visible dir with use_simotm="RGBT", channels=4.

This is inference + custom matching, not ultralytics val() -- single class /
single GT per frame, so recall at IoU>=0.5 here is used as a mAP50-like
proxy for comparing inference configs (see map50_like in evaluate_results).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

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

from scripts.training.shared.utils import check_ultralytics_import
from scripts.evaluation.polygon.shared_poly_eval import (
    prepare_tmp_test_interval as _prepare_tmp_test_interval,
    get_pred_boxes_and_confs,
    match_frame_boxes,
    read_yolo_boxes,
)

# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = POLYGON_DATA_ROOT / "data_14may_selected_synced_warped_labeled_full_crop_x287_1248"

TEST_INTERVAL = "2026-05-14_16-49-25_02616_03299"

IMGSZ = 1248
CONF_VALUES = [0.25, 0.15, 0.10]
EVAL_IOU_THR = 0.5

PROJECT_DIR = DIPLOMA_ROOT / "train" / "runs" / "antiuav_resync"

# All 5 model variants compared in the polygon validation stage: IR-only, RGB-only, RGBT from scratch, and the two RGBT
# MCF-transfer variants (RGB neck/head vs IR neck/head, both backbones
# frozen). RGB/IR-only models run on a single modality (use_simotm/channels
# differ accordingly, and each reads from its own image dir); the RGBT
# variants all run on the paired visible dir with use_simotm="RGBT".
MODELS = [
    {
        "model_name": "rgb_only",
        "weight_path": PROJECT_DIR
        / "yolo11s_rgb_from_scratch_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "RGB",
        "channels": 3,
        "source_key": "visible",
    },
    {
        "model_name": "ir_only",
        "weight_path": PROJECT_DIR
        / "yolo11s_ir_from_scratch_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "Gray",
        "channels": 1,
        "source_key": "infrared",
    },
    {
        "model_name": "rgbt_from_scratch",
        "weight_path": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_from_scratch_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "RGBT",
        "channels": 4,
        "source_key": "visible",
    },
    {
        "model_name": "rgb_neck_head_freeze",
        "weight_path": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_rgb_neck_head_freeze_backbones_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "RGBT",
        "channels": 4,
        "source_key": "visible",
    },
    {
        "model_name": "ir_neck_head_freeze",
        "weight_path": PROJECT_DIR
        / "yolo11s_rgbt_midfusion_p3_ir_neck_head_freeze_backbones_poly_finetune_crop_x287_1248"
        / "weights" / "best.pt",
        "use_simotm": "RGBT",
        "channels": 4,
        "source_key": "visible",
    },
]

PRED_VARIANTS = [
    {"pred_name": "default_maxdet", "max_det": None},
    {"pred_name": "maxdet1", "max_det": 1},
]


# =========================================================
# HELPERS
# =========================================================

def prepare_tmp_test_interval(tmp_root: Path):
    """Thin wrapper binding the shared helper to this script's DATA_ROOT/TEST_INTERVAL."""
    return _prepare_tmp_test_interval(DATA_ROOT, TEST_INTERVAL, tmp_root)


def evaluate_results(results, pairs, model_name: str, pred_name: str, conf: float):
    tp = 0
    fp = 0
    fn = 0

    total_gt = 0
    total_pred = 0

    detected_frames = 0
    missed_frames = 0
    multi_pred_frames = 0

    best_ious = []

    for result, (vis_path, _) in zip(results, pairs):
        label_path = (
            DATA_ROOT / TEST_INTERVAL / "visible" / "labels" / f"{vis_path.stem}.txt"
        )

        gt_boxes = read_yolo_boxes(label_path, img_w=IMGSZ, img_h=IMGSZ)
        pred_boxes, _pred_confs = get_pred_boxes_and_confs(result)

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        if len(pred_boxes) > 0:
            detected_frames += 1
        else:
            missed_frames += 1

        if len(pred_boxes) > 1:
            multi_pred_frames += 1

        tp_frame, fp_frame, fn_frame, best_iou, _best_pi = match_frame_boxes(
            pred_boxes, gt_boxes, EVAL_IOU_THR
        )

        if gt_boxes:
            best_ious.append(float(best_iou))

        tp += tp_frame
        fp += fp_frame
        fn += fn_frame

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    mean_best_iou = float(np.mean(best_ious)) if best_ious else 0.0

    # Single class / single GT per frame in this scenario:
    # recall at IoU>=0.5 is used as a mAP50-like proxy for comparing
    # inference configs.
    map50_like = recall

    return {
        "model": model_name,
        "pred": pred_name,
        "variant": f"{model_name}_{pred_name}",
        "conf": conf,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50_like": map50_like,
        "mean_best_iou": mean_best_iou,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "detected_frames": detected_frames,
        "missed_frames": missed_frames,
        "multi_pred_frames": multi_pred_frames,
    }


def run_model_variant(model, model_cfg: dict, pred_variant: dict):
    model_name = model_cfg["model_name"]
    use_simotm = model_cfg["use_simotm"]
    channels = model_cfg["channels"]
    source_key = model_cfg["source_key"]

    pred_name = pred_variant["pred_name"]
    max_det = pred_variant["max_det"]

    rows = []

    print("\n" + "#" * 100)
    print(f"[MODEL]    {model_name}")
    print(f"[MODALITY] use_simotm={use_simotm} channels={channels} source={source_key}")
    print(f"[PRED]     {pred_name}")
    print("[PRED_IOU] default Ultralytics")
    print(f"[MAX_DET]  {max_det}")
    print("#" * 100)

    for conf in CONF_VALUES:
        print("\n" + "=" * 90)
        print(f"model={model_name}, pred={pred_name}, conf={conf}")
        print("=" * 90)

        with tempfile.TemporaryDirectory(prefix=f"eval_{model_name}_{pred_name}_") as tmp:
            tmp_root = Path(tmp)

            visible_dir, pairs = prepare_tmp_test_interval(tmp_root)

            if not pairs:
                print("[WARN] No paired frames found, skipping")
                continue

            # Both dirs are always symlinked by prepare_tmp_test_interval;
            # derive infrared_dir from the pairs rather than assuming its
            # internal folder name.
            infrared_dir = pairs[0][1].parent
            source_dir = visible_dir if source_key == "visible" else infrared_dir

            predict_kwargs = dict(
                source=str(source_dir),
                imgsz=IMGSZ,
                show=False,
                save=False,
                stream=True,
                use_simotm=use_simotm,
                channels=channels,
                conf=conf,
                verbose=False,
            )

            if max_det is not None:
                predict_kwargs["max_det"] = max_det

            results = list(model.predict(**predict_kwargs))

            metrics = evaluate_results(
                results=results,
                pairs=pairs,
                model_name=model_name,
                pred_name=pred_name,
                conf=conf,
            )

            rows.append(metrics)

            print(
                f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} | "
                f"P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"mAP50_like={metrics['map50_like']:.4f} "
                f"mean_best_iou={metrics['mean_best_iou']:.4f} "
                f"missed={metrics['missed_frames']} "
                f"multi={metrics['multi_pred_frames']}"
            )

    return rows


def print_table(rows):
    print("\n" + "=" * 140)
    print("ALL RESULTS")
    print("=" * 140)

    sorted_rows = sorted(
        rows,
        key=lambda x: (
            x["f1"],
            x["recall"],
            x["precision"],
            x["mean_best_iou"],
        ),
        reverse=True,
    )

    header = (
        f"{'rank':>4}  "
        f"{'model':<22} "
        f"{'pred':<15} "
        f"{'conf':>6} "
        f"{'P':>8} "
        f"{'R/mAP50':>8} "
        f"{'F1':>8} "
        f"{'meanIoU':>8} "
        f"{'TP':>5} "
        f"{'FP':>5} "
        f"{'FN':>5} "
        f"{'miss':>5} "
        f"{'multi':>5}"
    )

    print(header)
    print("-" * len(header))

    for i, r in enumerate(sorted_rows, 1):
        print(
            f"{i:>4}  "
            f"{r['model']:<22} "
            f"{r['pred']:<15} "
            f"{r['conf']:>6.2f} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{r['mean_best_iou']:>8.4f} "
            f"{r['tp']:>5} "
            f"{r['fp']:>5} "
            f"{r['fn']:>5} "
            f"{r['missed_frames']:>5} "
            f"{r['multi_pred_frames']:>5}"
        )

    print("\n" + "=" * 140)
    print("TOP-3")
    print("=" * 140)

    for i, r in enumerate(sorted_rows[:3], 1):
        print(
            f"{i}. model={r['model']}, pred={r['pred']}, conf={r['conf']:.2f} | "
            f"F1={r['f1']:.4f}, "
            f"P={r['precision']:.4f}, "
            f"R/mAP50_like={r['recall']:.4f}, "
            f"meanIoU={r['mean_best_iou']:.4f}, "
            f"TP={r['tp']}, FP={r['fp']}, FN={r['fn']}, "
            f"missed={r['missed_frames']}, multi={r['multi_pred_frames']}"
        )

    best = sorted_rows[0]

    print("\n" + "=" * 140)
    print("BEST CONFIG")
    print("=" * 140)
    print(
        f"model={best['model']}, pred={best['pred']}, conf={best['conf']:.2f} | "
        f"F1={best['f1']:.4f}, "
        f"P={best['precision']:.4f}, "
        f"R/mAP50_like={best['recall']:.4f}, "
        f"meanIoU={best['mean_best_iou']:.4f}, "
        f"TP={best['tp']}, FP={best['fp']}, FN={best['fn']}"
    )


# =========================================================
# MAIN
# =========================================================

def main():
    check_ultralytics_import(Path(ultralytics.__file__), EXPECTED_ULTRA_INIT)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Missing dataset root: {DATA_ROOT}")

    print(f"[DATA]  {DATA_ROOT}")
    print(f"[TEST]  {TEST_INTERVAL}")
    print(f"[IMGSZ] {IMGSZ}")
    print("[SAVE]  nothing will be saved")
    print("[EVAL]  ranking by F1 -> Recall/mAP50_like -> Precision -> meanIoU")

    all_rows = []

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

        for pred_variant in PRED_VARIANTS:
            rows = run_model_variant(
                model=model,
                model_cfg=model_cfg,
                pred_variant=pred_variant,
            )
            all_rows.extend(rows)

        del model

    print_table(all_rows)


if __name__ == "__main__":
    main()
