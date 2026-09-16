#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity-check that init_pretrain_rgb_neckhead.py correctly transferred weights
from the RGB/IR pretrain checkpoints into the initialized RGBT checkpoint
(tensor-by-tensor comparison).

Checks the RGB-neck/head variant (yolo11s_rgbt_midfusion_p3_rgb_ir_backbones_
rgb_neck_head_init.pt). To verify the IR-neck/head variant instead, point
RGBT_INIT at that checkpoint (produced by init_pretrain_ir_neckhead.py) --
note the common-part check below would then need to compare against ir_sd
using COMMON_MAP instead of rgb_sd.
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/YOLOv11-RGBT")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.transfer.weight_transfer import (
    RGB_BRANCH_MAP,
    IR_BRANCH_MAP,
    COMMON_MAP,
    get_model_from_ckpt,
)

RGB_PRETRAIN = Path("/home/src/diploma/pretrain/runs/yolo11s_rgb_pretrain/weights/best.pt")
IR_PRETRAIN = Path("/home/src/diploma/pretrain/runs/yolo11s_gray_ir_pretrain_100ep/weights/best.pt")

RGBT_INIT = Path(
    "/home/src/diploma/train/runs/antiuav_resync/"
    "rgbt_init_from_rgb_ir_pretrain/"
    "yolo11s_rgbt_midfusion_p3_rgb_ir_backbones_rgb_neck_head_init.pt"
)


def check_mapping(src_sd, dst_sd, prefix_map, name):
    total = 0
    ok = 0
    bad = []
    missing = []
    shape_bad = []

    print("=" * 90)
    print(f"CHECK {name}")
    print("=" * 90)

    for src_prefix, dst_prefix in prefix_map.items():
        for src_k, src_v in src_sd.items():
            if not src_k.startswith(src_prefix):
                continue

            dst_k = dst_prefix + src_k[len(src_prefix):]
            total += 1

            if dst_k not in dst_sd:
                missing.append((src_k, dst_k))
                continue

            if tuple(src_v.shape) != tuple(dst_sd[dst_k].shape):
                shape_bad.append((src_k, dst_k, tuple(src_v.shape), tuple(dst_sd[dst_k].shape)))
                continue

            if torch.equal(src_v, dst_sd[dst_k]):
                ok += 1
            else:
                max_diff = (src_v - dst_sd[dst_k]).abs().max().item() if src_v.numel() > 0 else 0.0
                bad.append((src_k, dst_k, max_diff))

    print(f"Total expected tensors: {total}")
    print(f"Exact equal tensors:    {ok}")
    print(f"Missing dst tensors:    {len(missing)}")
    print(f"Shape mismatches:       {len(shape_bad)}")
    print(f"Value mismatches:       {len(bad)}")

    if missing:
        print("\nFirst missing:")
        for x in missing[:10]:
            print("  ", x)

    if shape_bad:
        print("\nFirst shape mismatches:")
        for x in shape_bad[:10]:
            print("  ", x)

    if bad:
        print("\nFirst value mismatches:")
        for x in bad[:10]:
            print("  ", x)

    passed = (total > 0 and ok == total and not missing and not shape_bad and not bad)
    print(f"\nRESULT {name}: {'PASS' if passed else 'FAIL'}")
    print()

    return passed, total, ok


def main():
    print("=" * 90)
    print("VERIFY FULL RGBT INIT")
    print("=" * 90)

    for p in [RGB_PRETRAIN, IR_PRETRAIN, RGBT_INIT]:
        if not p.exists():
            raise FileNotFoundError(p)

    rgb_model = get_model_from_ckpt(RGB_PRETRAIN)
    ir_model = get_model_from_ckpt(IR_PRETRAIN)
    rgbt_model = get_model_from_ckpt(RGBT_INIT)

    rgb_sd = rgb_model.state_dict()
    ir_sd = ir_model.state_dict()
    rgbt_sd = rgbt_model.state_dict()

    pass_rgb, n_rgb, ok_rgb = check_mapping(rgb_sd, rgbt_sd, RGB_BRANCH_MAP, "RGB BRANCH")
    pass_ir, n_ir, ok_ir = check_mapping(ir_sd, rgbt_sd, IR_BRANCH_MAP, "IR BRANCH")
    pass_common, n_common, ok_common = check_mapping(rgb_sd, rgbt_sd, COMMON_MAP, "RGB COMMON NECK/HEAD")

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"RGB branch:        {ok_rgb}/{n_rgb} exact tensors")
    print(f"IR branch:         {ok_ir}/{n_ir} exact tensors")
    print(f"RGB common part:   {ok_common}/{n_common} exact tensors")
    print()

    if pass_rgb and pass_ir and pass_common:
        print("ALL CHECKS PASSED")
        print("RGBT checkpoint is correctly initialized.")
    else:
        print("CHECKS FAILED")
        raise RuntimeError("Some transferred tensors do not match.")


if __name__ == "__main__":
    main()
