#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decision-support visualization for choosing the polygon crop window (not
part of the automated data pipeline -- run once by hand to pick the
crop offset used by crop_pad_to_1248.py).

Draws every GT box across the whole capture (all 6 intervals, both
modalities) onto a single frame, colored by interval, plus the IR field-
of-view rectangle and the candidate crop/pad window. This is how the
x=287 crop offset in crop_pad_to_1248.py was chosen: the window that
keeps the most accumulated drone positions in frame while dropping the
fewest boxes. Also renders a preview of the final cropped+padded
1248x1248 frame with the bottom padding highlighted.

Saves one debug image to OUT_IMG -- no dataset files are modified.
"""

from pathlib import Path
import cv2
import numpy as np

from scripts.paths import POLYGON_DATA_ROOT, DIPLOMA_ROOT

DATA_ROOT = POLYGON_DATA_ROOT / "data_14may_selected_synced_warped_labeled_full"

BASE_IMG = (
    DATA_ROOT
    / "2026-05-14_16-49-25_02616_03299/visible/images/"
    "2026-05-14_16-49-25_right003066_left003066.jpg"
)

OUT_DIR = DIPLOMA_ROOT / "anti_uav" / "polygon_prep" / "debugs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_IMG = OUT_DIR / "all_dataset_boxes_by_interval_with_ir_fov_and_crop_1248.jpg"

# IR FOV on original 1920x1080 canvas
IR_X1 = 87
IR_Y1 = 31
IR_X2 = 1521
IR_Y2 = 1080

# New crop/pad plan
CROP_X1 = 287
CROP_Y1 = 0
CROP_W = 1248
CROP_H_REAL = 1080
FINAL_H = 1248
PAD_BOTTOM = FINAL_H - CROP_H_REAL

CROP_X2 = CROP_X1 + CROP_W
CROP_Y2_REAL = CROP_Y1 + CROP_H_REAL

IR_COLOR = (0, 255, 255)          # yellow
CROP_COLOR = (255, 0, 255)        # purple / magenta in BGR
PAD_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)

BOX_THICKNESS = 1
IR_THICKNESS = 4
CROP_THICKNESS = 5

PAD_ALPHA = 0.28

INTERVAL_COLORS = {
    "2026-05-14_16-49-25_01490_01640": (0, 0, 255),       # red
    "2026-05-14_16-49-25_02616_03299": (0, 255, 0),       # green
    "2026-05-14_16-49-25_04750_05210": (255, 0, 0),       # blue
    "2026-05-14_16-49-25_06618_06952": (0, 255, 255),     # yellow
    "2026-05-14_17-02-50_01290_01717": (255, 0, 255),     # magenta
    "2026-05-14_17-25-12_01924_02085": (255, 255, 0),     # cyan
}


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

        cls, cx, cy, bw, bh = parts
        boxes.append((int(float(cls)), float(cx), float(cy), float(bw), float(bh)))

    return boxes


def yolo_to_xyxy_px(box, img_w, img_h):
    cls, cx, cy, bw, bh = box

    x1 = int(round((cx - bw / 2) * img_w))
    y1 = int(round((cy - bh / 2) * img_h))
    x2 = int(round((cx + bw / 2) * img_w))
    y2 = int(round((cy + bh / 2) * img_h))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    return x1, y1, x2, y2


def draw_text_with_bg(img, text, org, font_scale=0.7, color=TEXT_COLOR, bg=TEXT_BG, thickness=2):
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    cv2.rectangle(
        img,
        (x - 4, y - th - baseline - 4),
        (x + tw + 4, y + baseline + 4),
        bg,
        -1,
    )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main():
    img = cv2.imread(str(BASE_IMG), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read base image: {BASE_IMG}")

    h, w = img.shape[:2]
    vis = img.copy()

    all_boxes = []
    per_interval = {k: 0 for k in INTERVAL_COLORS}
    per_modality = {"visible": 0, "infrared": 0}

    for label_path in sorted(DATA_ROOT.rglob("*.txt")):
        path_str = str(label_path)

        if "/labels/" not in path_str:
            continue

        interval = None
        for interval_name in INTERVAL_COLORS:
            if f"/{interval_name}/" in path_str:
                interval = interval_name
                break

        if interval is None:
            continue

        modality = "visible" if "/visible/labels/" in path_str else (
            "infrared" if "/infrared/labels/" in path_str else "unknown"
        )

        boxes = read_yolo_boxes(label_path)

        for box in boxes:
            x1, y1, x2, y2 = yolo_to_xyxy_px(box, w, h)
            all_boxes.append((x1, y1, x2, y2, interval, modality))

            per_interval[interval] += 1
            if modality in per_modality:
                per_modality[modality] += 1

    # all boxes
    for x1, y1, x2, y2, interval, modality in all_boxes:
        color = INTERVAL_COLORS[interval]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # IR FOV border
    cv2.rectangle(
        vis,
        (IR_X1, IR_Y1),
        (IR_X2 - 1, IR_Y2 - 1),
        IR_COLOR,
        IR_THICKNESS,
    )

    draw_text_with_bg(
        vis,
        "IR",
        (IR_X1 + 10, max(25, IR_Y1 - 8)),
        font_scale=1.0,
        color=IR_COLOR,
        bg=TEXT_BG,
        thickness=2,
    )

    # Crop vertical lines on original image
    cv2.line(vis, (CROP_X1, 0), (CROP_X1, h - 1), CROP_COLOR, CROP_THICKNESS)
    cv2.line(vis, (CROP_X2 - 1, 0), (CROP_X2 - 1, h - 1), CROP_COLOR, CROP_THICKNESS)

    # Crop real frame border on original 1920x1080
    cv2.rectangle(
        vis,
        (CROP_X1, CROP_Y1),
        (CROP_X2 - 1, CROP_Y2_REAL - 1),
        CROP_COLOR,
        CROP_THICKNESS,
    )

    draw_text_with_bg(
        vis,
        "CROP 1248x1080 -> PAD TO 1248x1248",
        (CROP_X1 + 10, 70),
        font_scale=0.75,
        color=CROP_COLOR,
        bg=TEXT_BG,
        thickness=2,
    )

    # Build separate canvas with bottom pad visualized
    padded_canvas = np.zeros((FINAL_H, CROP_W, 3), dtype=np.uint8)
    crop_part = vis[CROP_Y1:CROP_Y2_REAL, CROP_X1:CROP_X2].copy()
    padded_canvas[:CROP_H_REAL, :, :] = crop_part

    # Purple transparent padding area
    overlay = padded_canvas.copy()
    overlay[CROP_H_REAL:FINAL_H, :, :] = PAD_COLOR
    padded_canvas = cv2.addWeighted(overlay, PAD_ALPHA, padded_canvas, 1.0 - PAD_ALPHA, 0)

    # Padding border and labels
    cv2.rectangle(
        padded_canvas,
        (0, CROP_H_REAL),
        (CROP_W - 1, FINAL_H - 1),
        CROP_COLOR,
        4,
    )

    cv2.line(
        padded_canvas,
        (0, CROP_H_REAL),
        (CROP_W - 1, CROP_H_REAL),
        CROP_COLOR,
        4,
    )

    draw_text_with_bg(
        padded_canvas,
        f"BOTTOM PAD = {PAD_BOTTOM}px",
        (25, CROP_H_REAL + 55),
        font_scale=0.9,
        color=CROP_COLOR,
        bg=TEXT_BG,
        thickness=2,
    )

    draw_text_with_bg(
        padded_canvas,
        "FINAL 1248x1248",
        (25, 45),
        font_scale=0.9,
        color=CROP_COLOR,
        bg=TEXT_BG,
        thickness=2,
    )

    # Main info on original visualization
    text_lines = [
        f"all boxes: {len(all_boxes)}",
        f"visible boxes: {per_modality['visible']}",
        f"infrared boxes: {per_modality['infrared']}",
        f"IR FOV: x=[{IR_X1},{IR_X2}), y=[{IR_Y1},{IR_Y2})",
        f"crop: x=[{CROP_X1},{CROP_X2}), y=[0,1080)",
        f"pad bottom: {PAD_BOTTOM}px",
        f"final: 1248x1248",
        f"image: {w}x{h}",
    ]

    y = 35
    for line in text_lines:
        draw_text_with_bg(vis, line, (25, y), font_scale=0.7)
        y += 32

    # Legend
    legend_x = 25
    legend_y = 315

    draw_text_with_bg(vis, "Interval colors:", (legend_x, legend_y), font_scale=0.7)
    legend_y += 32

    for interval, color in INTERVAL_COLORS.items():
        cv2.rectangle(
            vis,
            (legend_x, legend_y - 18),
            (legend_x + 28, legend_y + 4),
            color,
            -1,
        )

        short_name = interval.replace("2026-05-14_", "")
        label = f"{short_name}: {per_interval[interval]} boxes"

        draw_text_with_bg(
            vis,
            label,
            (legend_x + 40, legend_y),
            font_scale=0.55,
            color=TEXT_COLOR,
            bg=TEXT_BG,
            thickness=1,
        )

        legend_y += 28

    # Combine original full-view + final padded crop preview
    gap = np.full((max(vis.shape[0], padded_canvas.shape[0]), 14, 3), 40, dtype=np.uint8)

    if vis.shape[0] < padded_canvas.shape[0]:
        pad_h = padded_canvas.shape[0] - vis.shape[0]
        vis = cv2.copyMakeBorder(vis, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    combined = np.hstack([vis, gap, padded_canvas])

    cv2.imwrite(str(OUT_IMG), combined)

    print("DONE")
    print(f"base image: {BASE_IMG}")
    print(f"dataset:    {DATA_ROOT}")
    print(f"saved:      {OUT_IMG}")
    print(f"image size: {w}x{h}")
    print(f"crop:       x=[{CROP_X1},{CROP_X2}), y=[0,1080)")
    print(f"final:      1248x1248")
    print(f"pad bottom: {PAD_BOTTOM}px")
    print(f"all boxes:  {len(all_boxes)}")
    print(f"visible:    {per_modality['visible']}")
    print(f"infrared:   {per_modality['infrared']}")
    print(f"IR FOV:     x=[{IR_X1},{IR_X2}), y=[{IR_Y1},{IR_Y2})")


if __name__ == "__main__":
    main()
