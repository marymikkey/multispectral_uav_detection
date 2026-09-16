#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for building an initialized RGBT (mid-fusion, P3) checkpoint
from separately pretrained RGB and IR YOLOv11s models.

Used by:
- init_pretrain_rgb_neckhead.py: neck/head initialized from the RGB pretrain model
- init_pretrain_ir_neckhead.py:  neck/head initialized from the IR pretrain model
- init_pretrain_verify.py:       sanity-checks the resulting checkpoint
"""

import re
from pathlib import Path

import torch

# ---------------------------------------------------------
# Layer-index mapping between a single-branch YOLOv11s model
# and the two-branch YOLOv11s-RGBT-midfusion-P3 architecture.
#
# RGB branch: RGB pretrained early backbone -> visible stream
# RGB model:  0,1,2,3,4
# RGBT model: 2,3,4,5,6
# ---------------------------------------------------------
RGB_BRANCH_MAP = {
    "model.0.": "model.2.",
    "model.1.": "model.3.",
    "model.2.": "model.4.",
    "model.3.": "model.5.",
    "model.4.": "model.6.",
}

# ---------------------------------------------------------
# IR branch: IR pretrained early backbone -> infrared stream
# IR model:   0,1,2,3,4
# RGBT model: 8,9,10,11,12
# ---------------------------------------------------------
IR_BRANCH_MAP = {
    "model.0.": "model.8.",
    "model.1.": "model.9.",
    "model.2.": "model.10.",
    "model.3.": "model.11.",
    "model.4.": "model.12.",
}

# ---------------------------------------------------------
# Common part after fusion (neck/head), shared layer indices.
#
# RGBT model has fusion around layers 13-14.
# Layer 14 is new fusion compression Conv, so it remains random.
#
# Source model:  5,6,7,8,9,10 -> RGBT model: 15,16,17,18,19,20
# Source head:   13,16,17,19,20,22,23 -> RGBT: 23,26,27,29,30,32,33
#
# Same index mapping regardless of whether the common part is taken
# from the RGB pretrain model or the IR pretrain model -- only the
# source state dict differs, the destination indices are the same.
# ---------------------------------------------------------
COMMON_MAP = {
    "model.5.": "model.15.",
    "model.6.": "model.16.",
    "model.7.": "model.17.",
    "model.8.": "model.18.",
    "model.9.": "model.19.",
    "model.10.": "model.20.",

    "model.13.": "model.23.",
    "model.16.": "model.26.",
    "model.17.": "model.27.",
    "model.19.": "model.29.",
    "model.20.": "model.30.",
    "model.22.": "model.32.",
    "model.23.": "model.33.",
}


def make_nc3_cfg(rgbt_cfg_src: Path, out_path: Path) -> None:
    """Copy an RGBT model cfg, forcing nc=3 (matches the RGB/IR pretrain nc)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = rgbt_cfg_src.read_text(encoding="utf-8")
    text = re.sub(r"^nc:\s*\d+", "nc: 3", text, count=1, flags=re.MULTILINE)

    out_path.write_text(text, encoding="utf-8")
    print(f"[CFG] Saved nc=3 RGBT cfg to: {out_path}")


def get_model_from_ckpt(path: Path):
    """Load a YOLO checkpoint and return the underlying nn.Module (fp32)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if ckpt.get("ema") is not None:
            return ckpt["ema"].float()
        if ckpt.get("model") is not None:
            return ckpt["model"].float()

    return ckpt.float()


def transfer_by_prefix(src_sd, dst_sd, prefix_map, branch_name, require_nonzero=True):
    """Copy tensors from src_sd into dst_sd according to a prefix mapping."""
    loaded = 0
    skipped = []

    for src_prefix, dst_prefix in prefix_map.items():
        for src_k, src_v in src_sd.items():
            if not src_k.startswith(src_prefix):
                continue

            dst_k = dst_prefix + src_k[len(src_prefix):]

            if dst_k not in dst_sd:
                skipped.append((src_k, dst_k, "missing_dst"))
                continue

            if tuple(dst_sd[dst_k].shape) != tuple(src_v.shape):
                skipped.append(
                    (src_k, dst_k, f"shape {tuple(src_v.shape)} -> {tuple(dst_sd[dst_k].shape)}")
                )
                continue

            dst_sd[dst_k] = src_v.detach().clone()
            loaded += 1

    print(f"[{branch_name}] loaded tensors: {loaded}")
    print(f"[{branch_name}] skipped tensors: {len(skipped)}")

    if skipped:
        print(f"[{branch_name}] first skipped examples:")
        for item in skipped[:10]:
            print("  ", item)

    if require_nonzero and loaded == 0:
        raise RuntimeError(f"{branch_name}: loaded 0 tensors. Check prefixes / model scale.")

    return loaded, skipped
