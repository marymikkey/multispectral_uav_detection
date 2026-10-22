#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the LLVIP models trained by train_llvip.py on the LLVIP test split with
Ultralytics val(). Metrics: Recall, Precision, AP@0.5, AP@0.5:0.95, F1.

    python -m experiments.llvip.val_llvip            # all trained models
    python -m experiments.llvip.val_llvip rgbt       # one model
"""

import sys
from pathlib import Path

from scripts.paths import YOLO_REPO_ROOT, DIPLOMA_ROOT

if str(YOLO_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_REPO_ROOT))

from ultralytics import YOLO

PROJECT_DIR = DIPLOMA_ROOT / "train" / "runs" / "llvip"
OUT_DIR = DIPLOMA_ROOT / "train" / "tests" / "llvip_test"

MODALITIES = {
    "rgb": ("configs/llvip/rgb_cfg.yaml", dict(use_simotm="RGB", channels=3)),
    "ir": ("configs/llvip/ir_cfg.yaml", dict(use_simotm="Gray", channels=1)),
    "rgbt": ("configs/llvip/rgbt_cfg.yaml",
             dict(use_simotm="RGBT", channels=4, pairs_rgb_ir=["visible", "infrared"])),
}


def main():
    names = sys.argv[1:] or list(MODALITIES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["| Model | Recall | Precision | AP@0.5 | AP@0.5:0.95 | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        weights = PROJECT_DIR / f"yolo11s_{name}_llvip" / "weights" / "best.pt"
        if not weights.exists():
            print(f"[SKIP] {weights} not found")
            continue
        data, kwargs = MODALITIES[name]
        r = YOLO(str(weights)).val(data=data, split="test", imgsz=640, batch=32, device=0,
                                   plots=False, verbose=False, project=str(OUT_DIR / "val_runs"),
                                   name=name, exist_ok=True, **kwargs)
        p, rc = float(r.box.mp), float(r.box.mr)
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        row = f"| {name} | {rc:.3f} | {p:.3f} | {float(r.box.map50):.3f} | {float(r.box.map):.3f} | {f1:.3f} |"
        print(row)
        lines.append(row)

    (OUT_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Results: {OUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
