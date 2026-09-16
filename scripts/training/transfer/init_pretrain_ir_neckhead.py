#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an initialized RGBT (mid-fusion, P3) checkpoint from separately
pretrained RGB and IR YOLOv11s models.

Backbones: RGB branch <- RGB pretrain, IR branch <- IR pretrain.
Neck/head (common part after fusion): from the IR pretrain model.

Counterpart: init_pretrain_rgb_neckhead.py (neck/head from the RGB pretrain model).
Verify with: init_pretrain_verify.py
"""

import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/YOLOv11-RGBT")

if "/ultralytics" in sys.path:
    sys.path.remove("/ultralytics")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

from scripts.training.transfer.weight_transfer import (
    RGB_BRANCH_MAP,
    IR_BRANCH_MAP,
    COMMON_MAP,
    make_nc3_cfg,
    get_model_from_ckpt,
    transfer_by_prefix,
)

# =========================================================
# CONFIGURATION
# =========================================================

RGB_PRETRAIN = Path("/home/src/diploma/pretrain/runs/yolo11s_rgb_pretrain/weights/best.pt")
IR_PRETRAIN = Path("/home/src/diploma/pretrain/runs/yolo11s_gray_ir_pretrain_100ep/weights/best.pt")

RGBT_CFG_SRC = Path("/home/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11s-RGBT-midfusion-P3.yaml")

OUT_DIR = Path("/home/src/diploma/train/runs/antiuav_resync/rgbt_init_from_rgb_ir_pretrain")
RGBT_CFG_NC3 = OUT_DIR / "yolo11s-RGBT-midfusion-P3-nc3.yaml"
OUT_PT = OUT_DIR / "yolo11s_rgbt_midfusion_p3_rgb_ir_backbones_ir_neck_head_init.pt"


# =========================================================
# MAIN
# =========================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_nc3_cfg(RGBT_CFG_SRC, RGBT_CFG_NC3)

    print("=" * 80)
    print("[1] Loading pretrained RGB and IR models")
    print("=" * 80)

    rgb_model = get_model_from_ckpt(RGB_PRETRAIN)
    ir_model = get_model_from_ckpt(IR_PRETRAIN)

    rgb_sd = rgb_model.state_dict()
    ir_sd = ir_model.state_dict()

    print(f"RGB pretrain: {RGB_PRETRAIN}")
    print(f"IR  pretrain: {IR_PRETRAIN}")

    print("=" * 80)
    print("[2] Creating RGBT MidFusion-P3 model with nc=3")
    print("=" * 80)

    rgbt_yolo = YOLO(str(RGBT_CFG_NC3))
    rgbt_model = rgbt_yolo.model
    rgbt_sd = rgbt_model.state_dict()

    print(f"RGBT cfg: {RGBT_CFG_NC3}")

    print("=" * 80)
    print("[3] Transferring RGB branch")
    print("=" * 80)
    loaded_rgb_branch, _ = transfer_by_prefix(
        rgb_sd, rgbt_sd, RGB_BRANCH_MAP, "RGB_BRANCH"
    )

    print("=" * 80)
    print("[4] Transferring IR branch")
    print("=" * 80)
    loaded_ir_branch, _ = transfer_by_prefix(
        ir_sd, rgbt_sd, IR_BRANCH_MAP, "IR_BRANCH"
    )

    print("=" * 80)
    print("[5] Transferring common neck/head from IR pretrain")
    print("=" * 80)
    loaded_common, _ = transfer_by_prefix(
        ir_sd, rgbt_sd, COMMON_MAP, "IR_COMMON_NECK_HEAD", require_nonzero=True
    )

    print("=" * 80)
    print("[6] Loading transferred state into RGBT model")
    print("=" * 80)

    rgbt_model.load_state_dict(rgbt_sd, strict=True)

    ckpt = {
        "model": rgbt_model,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {
            "model": str(RGBT_CFG_NC3),
            "rgb_pretrain": str(RGB_PRETRAIN),
            "ir_pretrain": str(IR_PRETRAIN),
            "init_type": (
                "RGB branch from RGB YOLO11s pretrain; "
                "IR branch from IR YOLO11s-gray pretrain; "
                "common neck/head from IR YOLO11s-gray pretrain; "
                "fusion layers random initialized"
            ),
        },
        "date": None,
        "version": "custom-rgbt-init-rgb-ir-backbones-ir-neck-head",
    }

    torch.save(ckpt, OUT_PT)

    print("=" * 80)
    print("[DONE]")
    print("=" * 80)
    print("Saved initialized RGBT checkpoint to:")
    print(f"{OUT_PT}")
    print()
    print(f"RGB branch loaded tensors:       {loaded_rgb_branch}")
    print(f"IR branch loaded tensors:        {loaded_ir_branch}")
    print(f"IR common neck/head tensors:     {loaded_common}")
    print()
    print("Random initialized parts:")
    print("- fusion concat/compression layers")
    print("- layers whose tensor shapes did not match")
    print()
    print("Use this checkpoint for final RGBT training.")


if __name__ == "__main__":
    main()
