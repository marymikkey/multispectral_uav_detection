#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regroup the synchronised Anti-UAV pairs by scene.

Input  (output of sync_anti_uav_rgbt.py):
    ANTI-UAV_synced_lite/<split>/{visible,infrared}/{images,labels}/<scene>_ir<NNNNNN>_rgb<NNNNNN>.*
Output:
    ANTI-UAV_synced_lite_by_scene/<split>/<scene>/{visible,infrared}/{images,labels}/<same file names>

The per-scene layout is what the day / night / meteo test configs in
configs/test/anti_uav/ point at, so subsets can be selected by listing scenes.
Files are symlinked by default (the test set is large); set LINK_MODE = "copy"
for real copies.
"""

import shutil
from collections import defaultdict
from pathlib import Path

from scripts.paths import ANTIUAV_DATA_ROOT

SRC_ROOT = ANTIUAV_DATA_ROOT / "ANTI-UAV_synced_lite"
OUT_ROOT = ANTIUAV_DATA_ROOT / "ANTI-UAV_synced_lite_by_scene"

SPLITS = ["test"]
LINK_MODE = "symlink"  # "symlink" or "copy"

MODALITIES = ["visible", "infrared"]
KINDS = ["images", "labels"]


def scene_from_stem(stem: str) -> str:
    return stem.rsplit("_ir", 1)[0]


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if LINK_MODE == "symlink":
        dst.symlink_to(src.resolve())
    elif LINK_MODE == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown LINK_MODE: {LINK_MODE}")


def main():
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        raise RuntimeError(
            f"Output already exists: {OUT_ROOT}\n"
            "Remove it first if you intend to regenerate it."
        )

    for split in SPLITS:
        counts = defaultdict(int)

        for modality in MODALITIES:
            for kind in KINDS:
                src_dir = SRC_ROOT / split / modality / kind
                if not src_dir.exists():
                    raise FileNotFoundError(f"Missing source dir: {src_dir}")

                for src in sorted(src_dir.iterdir()):
                    if not src.is_file():
                        continue
                    scene = scene_from_stem(src.stem)
                    place(src, OUT_ROOT / split / scene / modality / kind / src.name)
                    counts[scene] += 1

        print(f"[{split}] {len(counts)} scenes, {sum(counts.values())} files -> {OUT_ROOT / split}")


if __name__ == "__main__":
    main()
