#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train the RGB-only, IR-only or RGBT mid-fusion model on LLVIP with the same model and
hyperparameters as the Anti-UAV from-scratch stage (taken from
scripts/training/from_scratch/train_*_from_scratch.py). The RGB model reads the
visible frame in colour.

    python -m experiments.llvip.train_llvip rgb|ir|rgbt
"""

import argparse
from importlib import import_module
from pathlib import Path

from scripts.paths import DIPLOMA_ROOT

PROJECT_DIR = DIPLOMA_ROOT / "train" / "runs" / "llvip"

MODALITIES = {
    "rgb": ("train_rgb_from_scratch", "configs/llvip/rgb_cfg.yaml",
            dict(use_simotm="RGB", channels=3)),
    "ir": ("train_ir_from_scratch", "configs/llvip/ir_cfg.yaml",
           dict(use_simotm="Gray", channels=1)),
    "rgbt": ("train_rgbt_from_scratch", "configs/llvip/rgbt_cfg.yaml",
             dict(use_simotm="RGBT", channels=4, pairs_rgb_ir=["visible", "infrared"])),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modality", choices=MODALITIES)
    modality = parser.parse_args().modality
    script, data, kwargs = MODALITIES[modality]

    # Importing the Anti-UAV script sets up the fork import and gives its settings.
    ref = import_module(f"scripts.training.from_scratch.{script}")
    import ultralytics
    from ultralytics import YOLO
    from scripts.training.shared.utils import check_cuda, check_ultralytics_import

    check_cuda(ref.DEVICE)
    check_ultralytics_import(Path(ultralytics.__file__), ref.EXPECTED_ULTRA_INIT)
    if not Path(data).exists():
        raise FileNotFoundError(data)

    args = dict(ref.TRAIN_ARGS)
    args.update(project=str(PROJECT_DIR), name=f"yolo11s_{modality}_llvip")
    print(f"[RUN] {PROJECT_DIR / args['name']}  model={ref.MODEL_CFG}  data={data}")

    YOLO(ref.MODEL_CFG).train(data=data, single_cls=True, device=ref.DEVICE, **kwargs, **args)


if __name__ == "__main__":
    main()
