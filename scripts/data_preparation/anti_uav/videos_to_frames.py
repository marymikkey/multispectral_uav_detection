#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert raw Anti-UAV videos into per-frame images with YOLO labels.

Input:
    <split>/<scene>/{visible,infrared}.mp4 and {visible,infrared}.json
    (json keys: "exist", "gt_rect" with one [x, y, w, h] box per frame)
Output:
    ANTI-UAV_each_frame/{visible,infrared}/{images,labels}/<split>/<scene>_f<NNNNNN>.{jpg,txt}

Frames without an object get an empty label file. This is the input of
sync_anti_uav_rgbt.py.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import cv2
from tqdm import tqdm

from scripts.paths import ANTIUAV_DATA_ROOT

SRC_ROOT = ANTIUAV_DATA_ROOT
OUT_ROOT = SRC_ROOT / "ANTI-UAV_each_frame"

SPLITS = ["train", "val", "test"]
MODALITIES = ["visible", "infrared"]
CLASS_ID = 0  # one class: drone


def yolo_line(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Optional[str]:
    """[x, y, w, h] in pixels -> normalised YOLO line, or None for an invalid box."""
    if w <= 0 or h <= 0:
        return None

    cx = min(max(x + w / 2.0, 0.0), img_w - 1.0) / img_w
    cy = min(max(y + h / 2.0, 0.0), img_h - 1.0) / img_h
    w_n = w / img_w
    h_n = h / img_h

    if not (0.0 < w_n <= 1.0 and 0.0 < h_n <= 1.0):
        return None
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}"


def frame_labels(exist: int, rect: Union[List[float], List[List[float]]], img_w: int, img_h: int) -> List[str]:
    """Labels of one frame; rect is a single box or a list of boxes."""
    if exist != 1 or not isinstance(rect, list) or not rect:
        return []

    boxes = [rect] if all(isinstance(v, (int, float)) for v in rect) else rect
    lines = []
    for box in boxes:
        if isinstance(box, list) and len(box) == 4:
            line = yolo_line(*map(float, box), img_w, img_h)
            if line:
                lines.append(line)
    return lines


def convert_scene(scene_dir: Path, split: str, modality: str) -> None:
    video_path = scene_dir / f"{modality}.mp4"
    json_path = scene_dir / f"{modality}.json"
    if not video_path.exists() or not json_path.exists():
        return

    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    if "exist" not in ann or "gt_rect" not in ann:
        raise ValueError(f"{json_path}: keys 'exist' and 'gt_rect' are required")
    exist, gt_rect = ann["exist"], ann["gt_rect"]

    img_dir = OUT_ROOT / modality / "images" / split
    lbl_dir = OUT_ROOT / modality / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n_frames = min(len(exist), len(gt_rect))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video > 0:
        n_frames = min(n_frames, n_video)

    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        img_h, img_w = frame.shape[:2]
        stem = f"{scene_dir.name}_f{idx:06d}"

        cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
        lines = frame_labels(exist[idx], gt_rect[idx], img_w, img_h)
        (lbl_dir / f"{stem}.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    cap.release()


def main() -> None:
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise RuntimeError(
            f"Output already exists: {OUT_ROOT}\n"
            "Remove it first if you intend to regenerate it."
        )

    print(f"[INFO] source: {SRC_ROOT}")
    print(f"[INFO] output: {OUT_ROOT}")

    for split in SPLITS:
        split_dir = SRC_ROOT / split
        if not split_dir.exists():
            continue

        scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        for scene_dir in tqdm(scene_dirs, desc=split, unit="scene"):
            for modality in MODALITIES:
                convert_scene(scene_dir, split, modality)

        n_images = len(list((OUT_ROOT / "visible" / "images" / split).glob("*.jpg")))
        print(f"[{split}] {len(scene_dirs)} scenes, {n_images} visible frames")


if __name__ == "__main__":
    main()
