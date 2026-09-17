#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared side-by-side (RGB | IR) video-rendering helpers for the polygon
inference-visualization stage. Used by render_side_by_side_poly.py.

`prepare_tmp_test_interval` for symlinking the paired test-interval images
is not redefined here -- it's identical to the one already used by
scripts/evaluation/polygon/shared_poly_eval.py, so this module imports it
from there instead of holding a third copy.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluation.polygon.shared_poly_eval import prepare_tmp_test_interval  # noqa: F401

BOX_COLOR_BGR = (0, 0, 255)
BOX_THICKNESS = 2
SEP_WIDTH = 8

TEXT_COLOR_BGR = (255, 255, 255)
TEXT_BG_BGR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_video_from_images(image_dir: Path, out_video: Path, fps: int):
    images = sorted(image_dir.glob("*.jpg"))

    if not images:
        print(f"[WARN] No frames in {image_dir}")
        return

    list_file = image_dir / "_frames.txt"

    with open(list_file, "w", encoding="utf-8") as f:
        for img in images:
            f.write(f"file '{img.resolve()}'\n")
            f.write(f"duration {1 / fps}\n")
        f.write(f"file '{images[-1].resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-vsync", "vfr",
        "-pix_fmt", "yuv420p",
        str(out_video),
    ]

    subprocess.run(cmd, check=True)
    list_file.unlink(missing_ok=True)


def draw_label(img, text: str, x1: int, y1: int):
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    y_text = max(0, y1 - th - baseline - 4)

    cv2.rectangle(
        img,
        (x1, y_text),
        (x1 + tw + 6, y_text + th + baseline + 6),
        TEXT_BG_BGR,
        -1,
    )

    cv2.putText(
        img,
        text,
        (x1 + 3, y_text + th + 2),
        FONT,
        FONT_SCALE,
        TEXT_COLOR_BGR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_boxes(img, boxes_xyxy, confs):
    h, w = img.shape[:2]
    out = img.copy()

    for box, conf in zip(boxes_xyxy, confs):
        x1, y1, x2, y2 = box

        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR_BGR, BOX_THICKNESS)
        draw_label(out, f"drone {conf:.2f}", x1, y1)

    return out


def add_panel_title(img, title: str):
    out = img.copy()

    cv2.rectangle(out, (0, 0), (300, 38), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (10, 27),
        FONT,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def make_side_by_side_frame(visible_path: Path, infrared_path: Path, boxes_xyxy, confs):
    rgb = cv2.imread(str(visible_path), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(infrared_path), cv2.IMREAD_COLOR)

    if rgb is None:
        raise RuntimeError(f"Failed to read RGB image: {visible_path}")
    if ir is None:
        raise RuntimeError(f"Failed to read IR image: {infrared_path}")

    h, w = rgb.shape[:2]

    if ir.shape[:2] != (h, w):
        ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)

    rgb_draw = draw_boxes(rgb, boxes_xyxy, confs)
    ir_draw = draw_boxes(ir, boxes_xyxy, confs)

    rgb_draw = add_panel_title(rgb_draw, "RGB / visible")
    ir_draw = add_panel_title(ir_draw, "IR / infrared")

    sep = np.full((h, SEP_WIDTH, 3), BOX_COLOR_BGR, dtype=np.uint8)

    return np.hstack([rgb_draw, sep, ir_draw])


def save_side_by_side_frames(results, pairs, out_frames_dir: Path):
    ensure_dir(out_frames_dir)

    saved = 0
    n = min(len(results), len(pairs))

    if len(results) != len(pairs):
        print(f"[WARN] results/pairs mismatch: results={len(results)}, pairs={len(pairs)}")

    for i in range(n):
        result = results[i]
        visible_path, infrared_path = pairs[i]

        if result.boxes is None or len(result.boxes) == 0:
            boxes_xyxy = []
            confs = []
        else:
            boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().tolist()
            confs = result.boxes.conf.detach().cpu().numpy().tolist()

        frame = make_side_by_side_frame(
            visible_path=visible_path,
            infrared_path=infrared_path,
            boxes_xyxy=boxes_xyxy,
            confs=confs,
        )

        out_path = out_frames_dir / f"{i:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    return saved
