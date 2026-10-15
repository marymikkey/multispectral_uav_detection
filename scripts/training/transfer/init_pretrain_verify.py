#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity-check that init_pretrain_rgb_neckhead.py and init_pretrain_ir_neckhead.py
correctly transferred weights from the RGB/IR pretrain checkpoints into the
initialized RGBT checkpoints (tensor-by-tensor comparison).

For both variants the RGB and IR branches are compared with the matching pretrain
model; the shared neck/head is compared with the RGB pretrain (RGB-neck/head
variant) or with the IR pretrain (IR-neck/head variant).
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
from pathlib import Path

import torch

from scripts.paths import YOLO_REPO_ROOT, DIPLOMA_ROOT

REPO_ROOT = YOLO_REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.transfer.weight_transfer import (
    RGB_BRANCH_MAP,
    IR_BRANCH_MAP,
    COMMON_MAP,
    get_model_from_ckpt,
)

RGB_PRETRAIN = DIPLOMA_ROOT / "pretrain" / "runs" / "yolo11s_rgb_pretrain" / "weights" / "best.pt"
IR_PRETRAIN = DIPLOMA_ROOT / "pretrain" / "runs" / "yolo11s_gray_ir_pretrain_100ep" / "weights" / "best.pt"

INIT_DIR = DIPLOMA_ROOT / "train" / "runs" / "antiuav_resync" / "rgbt_init_from_rgb_ir_pretrain"

# variant name, initialized checkpoint, pretrain the shared neck/head came from
VARIANTS = [
    ("RGB NECK/HEAD", INIT_DIR / "yolo11s_rgbt_midfusion_p3_rgb_ir_backbones_rgb_neck_head_init.pt", "rgb"),
    ("IR NECK/HEAD", INIT_DIR / "yolo11s_rgbt_midfusion_p3_rgb_ir_backbones_ir_neck_head_init.pt", "ir"),
]


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

    for p in [RGB_PRETRAIN, IR_PRETRAIN] + [ckpt for _, ckpt, _ in VARIANTS]:
        if not p.exists():
            raise FileNotFoundError(p)

    pretrain_sd = {
        "rgb": get_model_from_ckpt(RGB_PRETRAIN).state_dict(),
        "ir": get_model_from_ckpt(IR_PRETRAIN).state_dict(),
    }

    summary = []
    for name, ckpt, common_src in VARIANTS:
        rgbt_sd = get_model_from_ckpt(ckpt).state_dict()
        checks = [
            ("RGB branch", check_mapping(pretrain_sd["rgb"], rgbt_sd, RGB_BRANCH_MAP, f"{name}: RGB BRANCH")),
            ("IR branch", check_mapping(pretrain_sd["ir"], rgbt_sd, IR_BRANCH_MAP, f"{name}: IR BRANCH")),
            (f"{common_src.upper()} common part",
             check_mapping(pretrain_sd[common_src], rgbt_sd, COMMON_MAP, f"{name}: COMMON NECK/HEAD")),
        ]
        summary.append((name, checks))

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    all_passed = True
    for name, checks in summary:
        print(name)
        for label, (passed, total, ok) in checks:
            print(f"  {label + ':':20s} {ok}/{total} exact tensors")
            all_passed = all_passed and passed
    print()

    if all_passed:
        print("ALL CHECKS PASSED")
        print("Both RGBT checkpoints are correctly initialized.")
    else:
        print("CHECKS FAILED")
        raise RuntimeError("Some transferred tensors do not match.")


if __name__ == "__main__":
    main()
