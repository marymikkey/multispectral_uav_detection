#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert the official LLVIP release into the layout read by the YOLOv11-RGBT fork.

Input (unpacked LLVIP.zip):
    LLVIP/{visible,infrared}/{train,test}/<id>.jpg
    LLVIP/Annotations/<id>.xml        one VOC file per pair (the pairs are aligned)
Output:
    LLVIP_yolo/{visible,infrared}/{train,test}/<id>.jpg   symlink to the original image
    LLVIP_yolo/{visible,infrared}/{train,test}/<id>.txt   YOLO labels, class 0 = person

The same label file is written for both modalities.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.paths import LLVIP_DATA_ROOT

SRC_ROOT = LLVIP_DATA_ROOT / "LLVIP"
OUT_ROOT = LLVIP_DATA_ROOT / "LLVIP_yolo"

SPLITS = ["train", "test"]
MODALITIES = ["visible", "infrared"]
CLASS_NAME = "person"


def voc_to_yolo(xml_path: Path) -> list:
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)

    lines = []
    for obj in root.findall("object"):
        if obj.find("name").text.strip() != CLASS_NAME:
            continue
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, x1), min(w, x2)
        y1, y2 = max(0.0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        )
    return lines


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Unpacked LLVIP not found: {SRC_ROOT}")
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise RuntimeError(f"Output already exists: {OUT_ROOT}\nRemove it first to regenerate.")

    for split in SPLITS:
        images = sorted((SRC_ROOT / "visible" / split).glob("*.jpg"))
        n_boxes = n_empty = 0

        for vis in images:
            ir = SRC_ROOT / "infrared" / split / vis.name
            xml = SRC_ROOT / "Annotations" / f"{vis.stem}.xml"
            if not ir.exists() or not xml.exists():
                raise FileNotFoundError(f"Missing pair or annotation for {vis.name}")

            lines = voc_to_yolo(xml)
            n_boxes += len(lines)
            n_empty += not lines

            for modality, src in (("visible", vis), ("infrared", ir)):
                out_dir = OUT_ROOT / modality / split
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / vis.name).symlink_to(src.resolve())
                (out_dir / f"{vis.stem}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")

        print(f"[{split}] {len(images)} pairs, {n_boxes} boxes, {n_empty} frames without people")

    print(f"Done: {OUT_ROOT}")


if __name__ == "__main__":
    main()
