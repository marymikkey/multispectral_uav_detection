#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop and pad the merged polygon dataset from 1920x1080 to the final
1248x1248 used everywhere downstream (training/eval/visualization).

Crop window: x=[287, 1535), y=[0, 1080) (full height kept), then pad
168px on the bottom to reach 1248x1248. The x=287 offset was chosen by
visualize_crop_selection.py (accumulates every GT box across the whole
capture on one frame, and picks the window that keeps the most drone
positions while cutting the fewest boxes). RGB and IR are cropped/padded
identically; YOLO boxes are shifted+clipped to the new frame and
renormalized (boxes with no area left after clipping are dropped).

Input:  data_14may_selected_synced_warped_labeled_full (see
        merge_cvat_labels_ir_rgb.py)
Output: data_14may_selected_synced_warped_labeled_full_crop_x287_1248
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
from tqdm import tqdm

from scripts.paths import POLYGON_DATA_ROOT


ROOT = POLYGON_DATA_ROOT

SRC_DATA = ROOT / "data_14may_selected_synced_warped_labeled_full"
OUT_DATA = ROOT / "data_14may_selected_synced_warped_labeled_full_crop_x287_1248"

CROP_X = 287
CROP_Y = 0
CROP_W = 1248
CROP_H_REAL = 1080

FINAL_W = 1248
FINAL_H = 1248
PAD_BOTTOM = FINAL_H - CROP_H_REAL

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_yolo_boxes(label_path: Path):
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

        # skip malformed label lines instead of crashing the whole run
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except Exception:
            continue

        boxes.append((cls, cx, cy, w, h))

    return boxes


def format_yolo_boxes(boxes):
    lines = []
    for cls, cx, cy, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def crop_and_pad_image(img):
    h, w = img.shape[:2]

    x1 = CROP_X
    y1 = CROP_Y
    x2 = CROP_X + CROP_W
    y2 = CROP_Y + CROP_H_REAL

    if x2 > w or y2 > h:
        raise RuntimeError(
            f"Crop outside image: crop=({x1},{y1})-({x2},{y2}), image={w}x{h}"
        )

    crop = img[y1:y2, x1:x2].copy()

    crop = cv2.copyMakeBorder(
        crop,
        top=0,
        bottom=PAD_BOTTOM,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    return crop


def transform_boxes_to_crop_pad(boxes, old_w, old_h):
    new_boxes = []

    crop_x1 = CROP_X
    crop_y1 = CROP_Y
    crop_x2 = CROP_X + CROP_W
    crop_y2 = CROP_Y + CROP_H_REAL

    for cls, cx, cy, bw, bh in boxes:
        x1 = (cx - bw / 2) * old_w
        y1 = (cy - bh / 2) * old_h
        x2 = (cx + bw / 2) * old_w
        y2 = (cy + bh / 2) * old_h

        nx1 = max(x1, crop_x1) - crop_x1
        ny1 = max(y1, crop_y1) - crop_y1
        nx2 = min(x2, crop_x2) - crop_x1
        ny2 = min(y2, crop_y2) - crop_y1

        nx1 = max(0.0, min(float(FINAL_W), nx1))
        ny1 = max(0.0, min(float(FINAL_H), ny1))
        nx2 = max(0.0, min(float(FINAL_W), nx2))
        ny2 = max(0.0, min(float(FINAL_H), ny2))

        if nx2 <= nx1 or ny2 <= ny1:
            continue

        new_cx = ((nx1 + nx2) / 2) / FINAL_W
        new_cy = ((ny1 + ny2) / 2) / FINAL_H
        new_w = (nx2 - nx1) / FINAL_W
        new_h = (ny2 - ny1) / FINAL_H

        new_boxes.append((cls, new_cx, new_cy, new_w, new_h))

    return new_boxes


def process_modality(interval_dir: Path, modality: str, stats):
    src_img_dir = interval_dir / modality / "images"
    src_lbl_dir = interval_dir / modality / "labels"

    out_img_dir = OUT_DATA / interval_dir.name / modality / "images"
    out_lbl_dir = OUT_DATA / interval_dir.name / modality / "labels"

    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    image_files = sorted(
        p for p in src_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    for img_path in tqdm(image_files, desc=f"{interval_dir.name}/{modality}"):
        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            stats["image_read_fail"] += 1
            continue

        old_h, old_w = img.shape[:2]

        new_img = crop_and_pad_image(img)

        if new_img.shape[1] != FINAL_W or new_img.shape[0] != FINAL_H:
            raise RuntimeError(
                f"Bad image shape for {img_path}: "
                f"{new_img.shape[1]}x{new_img.shape[0]}, expected {FINAL_W}x{FINAL_H}"
            )

        boxes = read_yolo_boxes(label_path)
        new_boxes = transform_boxes_to_crop_pad(boxes, old_w, old_h)

        cv2.imwrite(str(out_img_dir / img_path.name), new_img)
        (out_lbl_dir / f"{img_path.stem}.txt").write_text(
            format_yolo_boxes(new_boxes),
            encoding="utf-8",
        )

        stats["images_saved"] += 1
        stats["boxes_before"] += len(boxes)
        stats["boxes_after"] += len(new_boxes)

        if boxes and not new_boxes:
            stats["images_lost_all_boxes"] += 1
        if len(new_boxes) < len(boxes):
            stats["boxes_removed_or_clipped_out"] += len(boxes) - len(new_boxes)


def copy_json_if_exists(src: Path, dst: Path):
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def main():
    if not SRC_DATA.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_DATA}")

    # One-off preprocessing run -- fail loudly instead of silently mixing a
    # new run's output into stale files from a previous one (same
    # reasoning as merge_cvat_labels_ir_rgb.py).
    if OUT_DATA.exists():
        raise RuntimeError(
            f"Output already exists: {OUT_DATA}\n"
            "Remove it manually first if you intend to regenerate it "
            "(e.g. after re-running the merge step, or changing the crop "
            "window above), otherwise this run would mix new files with "
            "stale ones from a previous run."
        )

    ensure_dir(OUT_DATA)
    stats = defaultdict(int)

    interval_dirs = sorted(
        p for p in SRC_DATA.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )

    print(f"Source: {SRC_DATA}")
    print(f"Output: {OUT_DATA}")
    print(f"Crop: x={CROP_X}, y={CROP_Y}, w={CROP_W}, h={CROP_H_REAL}")
    print(f"Pad bottom: {PAD_BOTTOM}")
    print(f"Final size: {FINAL_W}x{FINAL_H}")

    for interval_dir in interval_dirs:
        for modality in ["visible", "infrared"]:
            if not (interval_dir / modality / "images").exists():
                print(f"[WARN] missing images dir: {interval_dir / modality / 'images'}")
                continue
            if not (interval_dir / modality / "labels").exists():
                print(f"[WARN] missing labels dir: {interval_dir / modality / 'labels'}")
                continue

            process_modality(interval_dir, modality, stats)

        copy_json_if_exists(
            interval_dir / "pairs.json",
            OUT_DATA / interval_dir.name / "pairs.json",
        )

    copy_json_if_exists(
        SRC_DATA / "merge_summary.json",
        OUT_DATA / "merge_summary_original.json",
    )

    summary = {
        "source_dataset": str(SRC_DATA),
        "output_dataset": str(OUT_DATA),
        "operation": "crop x=[287,1535), y=[0,1080), then pad bottom to 1248x1248",
        "crop": {
            "x": CROP_X,
            "y": CROP_Y,
            "width": CROP_W,
            "height_real": CROP_H_REAL,
            "pad_bottom": PAD_BOTTOM,
            "final_width": FINAL_W,
            "final_height": FINAL_H,
            "source_resolution": "1920x1080",
            "output_resolution": "1248x1248",
            "note": "RGB and IR are transformed identically; YOLO labels are shifted by crop and normalized to 1248x1248.",
        },
        "stats": dict(stats),
    }

    summary_path = OUT_DATA / "crop_x287_1248_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDONE")
    print(f"Saved dataset: {OUT_DATA}")
    print(f"Summary: {summary_path}")
    print("\nStats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
