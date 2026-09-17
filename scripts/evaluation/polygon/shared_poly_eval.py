#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared inference-evaluation primitives for the polygon stage: reading YOLO
label boxes, IoU, building a symlinked RGB/IR test-interval directory pair,
and the per-frame TP/FP/FN IoU-matching logic used by
eval_rgbt_poly_models.py (all 5 model variants x maxdet, console-only).

Kept generic (no hardcoded DATA_ROOT/TEST_INTERVAL/IMGSZ): callers pass their
own config in, since each script's config values already differ slightly in
practice (e.g. which model/interval is under test).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def read_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> List[List[float]]:
    """Read a YOLO-format label file and return absolute-pixel xyxy boxes."""
    if not label_path.exists():
        return []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    boxes = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, cx, cy, bw, bh = map(float, parts)

        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        boxes.append([x1, y1, x2, y2])

    return boxes


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def prepare_tmp_test_interval(
    data_root: Path,
    test_interval: str,
    tmp_root: Path,
) -> Tuple[Path, List[Tuple[Path, Path]]]:
    """Symlink a paired RGB/IR test interval into tmp_root/{visible,infrared}.

    Returns (visible_dir, pairs) where pairs is a list of
    (visible_image_path, infrared_image_path) symlinks, in sorted order.
    """
    visible_dir = tmp_root / "visible"
    infrared_dir = tmp_root / "infrared"

    visible_dir.mkdir(parents=True, exist_ok=True)
    infrared_dir.mkdir(parents=True, exist_ok=True)

    src_visible = data_root / test_interval / "visible" / "images"
    src_infrared = data_root / test_interval / "infrared" / "images"

    if not src_visible.exists():
        raise FileNotFoundError(f"Missing visible images dir: {src_visible}")

    if not src_infrared.exists():
        raise FileNotFoundError(f"Missing infrared images dir: {src_infrared}")

    pairs = []

    for vis_img in sorted(src_visible.glob("*.jpg")):
        ir_img = src_infrared / vis_img.name

        if not ir_img.exists():
            print(f"[WARN] Missing IR pair for: {vis_img.name}")
            continue

        dst_vis = visible_dir / vis_img.name
        dst_ir = infrared_dir / ir_img.name

        dst_vis.symlink_to(vis_img.resolve())
        dst_ir.symlink_to(ir_img.resolve())

        pairs.append((dst_vis, dst_ir))

    return visible_dir, pairs


def match_frame_boxes(pred_boxes: List[List[float]], gt_boxes: List[List[float]], iou_thr: float):
    """Greedy best-IoU-first matching of predicted vs. GT boxes in one frame.

    Returns (tp, fp, fn, best_iou, best_pi) where best_iou is the highest
    IoU seen among any pred/gt pair in the frame (0.0 if there were no GT
    boxes, or no candidate pairs), and best_pi is the index into pred_boxes
    of the prediction involved in that best pair (None if there were no
    candidate pairs).
    """
    if not gt_boxes:
        return 0, len(pred_boxes), 0, 0.0, None

    matched_gt = set()
    matched_pred = set()

    iou_pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou = box_iou(pb, gb)
            iou_pairs.append((iou, pi, gi))

    iou_pairs.sort(reverse=True, key=lambda x: x[0])

    for iou, pi, gi in iou_pairs:
        if iou < iou_thr:
            continue
        if pi in matched_pred or gi in matched_gt:
            continue

        matched_pred.add(pi)
        matched_gt.add(gi)

    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)

    if iou_pairs:
        best_iou, best_pi, _best_gi = iou_pairs[0]
    else:
        best_iou, best_pi = 0.0, None

    return tp, fp, fn, float(best_iou), best_pi


def get_pred_boxes_and_confs(result) -> Tuple[List[List[float]], List[float]]:
    """Extract (boxes_xyxy, confs) from an Ultralytics predict() result."""
    if result.boxes is None or len(result.boxes) == 0:
        return [], []

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().tolist()
    confs = result.boxes.conf.detach().cpu().numpy().tolist()
    return boxes_xyxy, confs
