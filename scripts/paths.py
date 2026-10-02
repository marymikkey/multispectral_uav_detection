#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Central place for the handful of absolute root paths this repo's scripts
depend on: the external YOLOv11-RGBT fork location, where training runs /
checkpoints / eval+visualization outputs get written, and where the
datasets live.

Every root below defaults to the exact path used on the original training
server -- so nothing changes here if you don't touch anything. To run this
repo against your own layout (different server, different drive), either
set the matching environment variable before running a script, or just
edit the default string below -- one place instead of hunting through
every script that used to hardcode it locally.

Individual scripts build their own subpaths by joining onto these roots
(e.g. `DIPLOMA_ROOT / "train" / "runs" / "antiuav_resync"`) -- this module
intentionally only holds the roots, not every derived path, so each
script's specific folder layout stays visible and easy to read in that
script rather than buried in here.
"""

import os
from pathlib import Path

# Where the custom YOLOv11-RGBT fork is cloned (see README.md "External
# dependencies" for the exact commit and clone/setup instructions).
YOLO_REPO_ROOT = Path(os.environ.get("YOLO_REPO_ROOT", "/home/YOLOv11-RGBT"))

# Base directory for everything this project WRITES: training runs,
# checkpoints, run_summary.md files, eval/visualization outputs,
# dynamically generated configs (subfolders train/, pretrain/, anti_uav/).
DIPLOMA_ROOT = Path(os.environ.get("DIPLOMA_ROOT", "/home/src/diploma"))

# Base directory for the polygon (field) dataset.
POLYGON_DATA_ROOT = Path(
    os.environ.get("POLYGON_DATA_ROOT", "/mnt/datasets/Maria_preprocess/mine_dpl")
)

# Base directory for the Anti-UAV dataset (raw / synced). Used by
# scripts/data_preparation/anti_uav/ and scripts/visualization/plot_scene_shift.py.
ANTIUAV_DATA_ROOT = Path(
    os.environ.get("ANTIUAV_DATA_ROOT", "/mnt/datasets/IR_DATA/ANTI-UAV")
)
