#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge separately CVAT-exported IR and RGB polygon labels into a single
unified per-frame ground truth (same label written to both visible/labels
and infrared/labels), for the 6 polygon capture intervals.

IR and RGB were annotated independently in CVAT under a shared, sequential
per-frame index ("cvat_id", contiguous across all 6 intervals). Per-frame
merge rule (see choose_final_boxes): explicit delete list / delete ranges
win first, then explicit RGB-only ranges, then explicit union ids, then an
automatic default (both present -> union bbox; else whichever spectrum has
a box; else drop the frame).

Input:  data_14may_selected_synced_warped (already synced + warped, not
        yet cropped -- see crop_pad_to_1248.py for the next stage).
Output: data_14may_selected_synced_warped_labeled_full
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict


ROOT = Path("/mnt/datasets/Maria_preprocess/mine_dpl")

SRC_DATA = ROOT / "data_14may_selected_synced_warped"
OUT_DATA = ROOT / "data_14may_selected_synced_warped_labeled_full"

IR_LABEL_ROOT = ROOT / "poly_ir_cvat/labels/Train/ir_export"
RGB_FULL_ROOT = ROOT / "poly_rgb_full"

INTERVALS = [
    ("2026-05-14_16-49-25_01490_01640", 0, 149),
    ("2026-05-14_16-49-25_02616_03299", 150, 831),
    ("2026-05-14_16-49-25_04750_05210", 832, 1290),
    ("2026-05-14_16-49-25_06618_06952", 1291, 1623),
    ("2026-05-14_17-02-50_01290_01717", 1624, 2049),
    ("2026-05-14_17-25-12_01924_02085", 2050, 2208),
]

DELETE_IDS = {8}

DELETE_RANGES = [
    (150, 162),
    (822, 837),
    (1251, 1296),
    (1615, 1623),
    (1815, 1835),
    (1842, 1869),
    (1922, 1942),
    (1959, 1976),
    (1992, 1994),
    (2047, 2049),
    (2199, 2208),
]

RGB_ONLY_RANGES = [
    (260, 263),
    (1551, 1564),
]

EXPLICIT_UNION_IDS = {
    259,
    *range(264, 275),
    1550,
    *range(1565, 1570),
    1614,
    1836,
    1870,
}

CLASS_ID = 0


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def in_ranges(idx: int, ranges):
    return any(a <= idx <= b for a, b in ranges)


def find_rgb_label_root():
    candidates = [
        p for p in RGB_FULL_ROOT.rglob("*")
        if p.is_dir() and any(p.glob("*.txt"))
    ]

    if not candidates:
        raise FileNotFoundError(f"No label txt folder found inside: {RGB_FULL_ROOT}")

    # Обычно это .../labels/Train/<export_name>/<interval>/*.txt,
    # поэтому корень нужен на уровень выше interval folders.
    for p in candidates:
        if any(child.is_dir() for child in p.iterdir()):
            continue

    interval_names = {x[0] for x in INTERVALS}

    for p in RGB_FULL_ROOT.rglob("*"):
        if p.is_dir():
            child_names = {c.name for c in p.iterdir() if c.is_dir()}
            if interval_names & child_names:
                return p

    # fallback: если txt лежат плоско
    return candidates[0]


def read_yolo_boxes(path: Path):
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    boxes = []

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            cls = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except Exception:
            continue

        boxes.append((cls, cx, cy, w, h))

    return boxes


def yolo_to_xyxy(box):
    _, cx, cy, w, h = box
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def xyxy_to_yolo(x1, y1, x2, y2):
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return CLASS_ID, cx, cy, w, h


def union_boxes(boxes):
    if not boxes:
        return []

    xyxys = [yolo_to_xyxy(b) for b in boxes]

    x1 = min(b[0] for b in xyxys)
    y1 = min(b[1] for b in xyxys)
    x2 = max(b[2] for b in xyxys)
    y2 = max(b[3] for b in xyxys)

    return [xyxy_to_yolo(x1, y1, x2, y2)]


def format_yolo_boxes(boxes):
    lines = []

    for _, cx, cy, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines) + ("\n" if lines else "")


def build_cvat_mapping(rgb_label_root: Path):
    mapping = {}

    for interval_name, cvat_start, cvat_end in INTERVALS:
        visible_dir = SRC_DATA / interval_name / "left"
        infrared_dir = SRC_DATA / interval_name / "right"

        files = sorted(visible_dir.glob("*.jpg"))

        expected_count = cvat_end - cvat_start + 1
        if len(files) != expected_count:
            print(f"[WARN] {interval_name}: expected {expected_count}, found {len(files)}")

        for local_i, visible_img in enumerate(files):
            cvat_id = cvat_start + local_i
            stem = visible_img.stem

            mapping[cvat_id] = {
                "interval": interval_name,
                "stem": stem,
                "visible_img": visible_img,
                "infrared_img": infrared_dir / f"{stem}.jpg",
                "ir_label": IR_LABEL_ROOT / interval_name / f"{stem}.txt",
                "rgb_label": rgb_label_root / interval_name / f"{stem}.txt",
            }

    return mapping


def choose_final_boxes(cvat_id: int, ir_boxes, rgb_boxes):
    if cvat_id in DELETE_IDS or in_ranges(cvat_id, DELETE_RANGES):
        return [], "delete"

    if in_ranges(cvat_id, RGB_ONLY_RANGES):
        return rgb_boxes, "rgb_only"

    if cvat_id in EXPLICIT_UNION_IDS:
        return union_boxes(ir_boxes + rgb_boxes), "explicit_union"

    if ir_boxes and rgb_boxes:
        return union_boxes(ir_boxes + rgb_boxes), "auto_union_ir_rgb"

    if rgb_boxes:
        return rgb_boxes, "rgb_only_auto"

    if ir_boxes:
        return ir_boxes, "ir_only"

    return [], "delete_empty"


def main():
    # This is a one-off preprocessing run, not something re-run on every
    # pipeline execution -- fail loudly instead of silently mixing a new
    # run's output into stale files from a previous one (e.g. a frame that
    # used to be "kept" but should now be deleted after a re-annotation
    # would otherwise be left behind untouched).
    if OUT_DATA.exists():
        raise RuntimeError(
            f"Output already exists: {OUT_DATA}\n"
            "Remove it manually first if you intend to regenerate it "
            "(e.g. after re-exporting CVAT labels or changing the merge "
            "rules above), otherwise this run would mix new files with "
            "stale ones from a previous run."
        )

    rgb_label_root = find_rgb_label_root()

    print(f"Source synced data: {SRC_DATA}")
    print(f"IR labels:          {IR_LABEL_ROOT}")
    print(f"RGB full labels:    {rgb_label_root}")
    print(f"Output:             {OUT_DATA}")

    mapping = build_cvat_mapping(rgb_label_root)

    stats = defaultdict(int)
    per_interval = defaultdict(lambda: defaultdict(int))
    actions_log = []

    for cvat_id in sorted(mapping.keys()):
        item = mapping[cvat_id]
        interval = item["interval"]
        stem = item["stem"]

        visible_img = item["visible_img"]
        infrared_img = item["infrared_img"]

        if not visible_img.exists():
            stats["visible_missing"] += 1
            continue

        if not infrared_img.exists():
            stats["infrared_missing"] += 1
            continue

        ir_boxes = read_yolo_boxes(item["ir_label"])
        rgb_boxes = read_yolo_boxes(item["rgb_label"])

        final_boxes, action = choose_final_boxes(cvat_id, ir_boxes, rgb_boxes)

        if action.startswith("delete") or not final_boxes:
            stats["deleted"] += 1
            per_interval[interval]["deleted"] += 1
            actions_log.append({
                "cvat_id": cvat_id,
                "interval": interval,
                "stem": stem,
                "action": action,
                "ir_boxes": len(ir_boxes),
                "rgb_boxes": len(rgb_boxes),
                "final_boxes": 0,
            })
            continue

        out_visible_img_dir = OUT_DATA / interval / "visible" / "images"
        out_visible_lbl_dir = OUT_DATA / interval / "visible" / "labels"
        out_infrared_img_dir = OUT_DATA / interval / "infrared" / "images"
        out_infrared_lbl_dir = OUT_DATA / interval / "infrared" / "labels"

        ensure_dir(out_visible_img_dir)
        ensure_dir(out_visible_lbl_dir)
        ensure_dir(out_infrared_img_dir)
        ensure_dir(out_infrared_lbl_dir)

        shutil.copy2(visible_img, out_visible_img_dir / f"{stem}.jpg")
        shutil.copy2(infrared_img, out_infrared_img_dir / f"{stem}.jpg")

        label_text = format_yolo_boxes(final_boxes)

        (out_visible_lbl_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")
        (out_infrared_lbl_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")

        stats["kept"] += 1
        stats[f"action_{action}"] += 1

        per_interval[interval]["kept"] += 1
        per_interval[interval][f"action_{action}"] += 1

        actions_log.append({
            "cvat_id": cvat_id,
            "interval": interval,
            "stem": stem,
            "action": action,
            "ir_boxes": len(ir_boxes),
            "rgb_boxes": len(rgb_boxes),
            "final_boxes": len(final_boxes),
        })

    summary = {
        "source_synced_data": str(SRC_DATA),
        "output": str(OUT_DATA),
        "ir_label_root": str(IR_LABEL_ROOT),
        "rgb_label_root": str(rgb_label_root),
        "total_cvat_items": len(mapping),
        "stats": dict(stats),
        "per_interval": {k: dict(v) for k, v in per_interval.items()},
        "rules": {
            "delete_ids": sorted(DELETE_IDS),
            "delete_ranges": DELETE_RANGES,
            "rgb_only_ranges": RGB_ONLY_RANGES,
            "explicit_union_ids": sorted(EXPLICIT_UNION_IDS),
            "default_logic": "if IR and RGB exist -> union; else RGB; else IR; else delete",
        },
        "actions_log": actions_log,
    }

    ensure_dir(OUT_DATA)
    (OUT_DATA / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDONE")
    print(f"Output saved to: {OUT_DATA}")
    print("\nStats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    print(f"\nSummary: {OUT_DATA / 'merge_summary.json'}")


if __name__ == "__main__":
    main()
