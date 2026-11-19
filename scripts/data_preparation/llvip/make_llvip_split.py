#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split the official LLVIP train set into train / val by scene and build the folders
used by the LLVIP experiments. The official test set is left unchanged.

LLVIP file names encode the scene in their first two digits (010001.jpg -> scene 01).
The official train set has 19 scenes (01-18, 25), the test set 7 (19-24, 26); the two
do not share scenes. Validation scenes are chosen once, at random with a fixed seed:
the train scenes are shuffled and taken in that order until the chosen scenes hold
at least VAL_FRACTION of the train frames. No scene is split between train and val.

The resulting frame lists are written to experiments/llvip/splits/ and committed, so
every later run uses exactly the same split. If the lists already exist, they are
read and the folders are built from them; pass --regenerate to choose the scenes again.

Input (written by prepare_llvip.py):
    $LLVIP_DATA_ROOT/LLVIP_yolo/{visible,infrared}/{train,test}/<id>.jpg + <id>.txt
Output:
    experiments/llvip/splits/{train,val,test}.txt      one frame id per line
    experiments/llvip/splits/split_summary.md          scenes, frame and box counts
    $LLVIP_DATA_ROOT/LLVIP_yolo/{visible,infrared}/{train_split,val_split}/
        <id>.jpg  symlink to the original image, <id>.txt  copy of the YOLO label

    python -m scripts.data_preparation.llvip.make_llvip_split
    python -m scripts.data_preparation.llvip.make_llvip_split --regenerate
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from scripts.paths import LLVIP_DATA_ROOT

YOLO_ROOT = LLVIP_DATA_ROOT / "LLVIP_yolo"
SPLIT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "llvip" / "splits"

VAL_FRACTION = 0.12
SPLIT_SEED = 0
MODALITIES = ["visible", "infrared"]

# Output folder name for each split. The official test folder is used as is.
OUT_FOLDERS = {"train": "train_split", "val": "val_split", "test": "test"}


def scene_of(frame_id: str) -> str:
    return frame_id[:2]


def list_ids(folder: Path) -> list:
    ids = sorted(p.stem for p in folder.glob("*.jpg"))
    if not ids:
        raise FileNotFoundError(f"No images in {folder}. Run prepare_llvip.py first.")
    return ids


def choose_val_scenes(train_ids: list) -> list:
    counts = Counter(scene_of(i) for i in train_ids)
    scenes = sorted(counts)
    random.Random(SPLIT_SEED).shuffle(scenes)

    chosen, n = [], 0
    for scene in scenes:
        if n >= VAL_FRACTION * len(train_ids):
            break
        chosen.append(scene)
        n += counts[scene]
    return sorted(chosen)


def read_list(path: Path) -> list:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_list(path: Path, ids: list) -> None:
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")


def count_boxes(folder: Path, ids: list) -> int:
    return sum(len(read_list(folder / f"{i}.txt")) for i in ids)


def build_folder(split: str, ids: list) -> None:
    """Create <modality>/<split folder>/ with image symlinks and label copies."""
    for modality in MODALITIES:
        src_dir = YOLO_ROOT / modality / "train"
        out_dir = YOLO_ROOT / modality / OUT_FOLDERS[split]
        if out_dir.exists():
            if sorted(p.stem for p in out_dir.glob("*.jpg")) == sorted(ids):
                print(f"[BUILD] {out_dir} already matches the list, kept")
                continue
            raise RuntimeError(
                f"{out_dir} exists but does not match the list. Remove it (and its .cache file) to rebuild."
            )
        out_dir.mkdir(parents=True)
        for i in ids:
            (out_dir / f"{i}.jpg").symlink_to((src_dir / f"{i}.jpg").resolve())
            shutil.copyfile(src_dir / f"{i}.txt", out_dir / f"{i}.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true",
                        help="choose the validation scenes again and overwrite the committed lists")
    args = parser.parse_args()

    official_train = list_ids(YOLO_ROOT / "visible" / "train")
    official_test = list_ids(YOLO_ROOT / "visible" / "test")
    for modality in MODALITIES[1:]:
        for split, ids in (("train", official_train), ("test", official_test)):
            if list_ids(YOLO_ROOT / modality / split) != ids:
                raise RuntimeError(f"{modality}/{split} does not match visible/{split}")

    lists = {s: SPLIT_DIR / f"{s}.txt" for s in OUT_FOLDERS}
    if all(p.exists() for p in lists.values()) and not args.regenerate:
        print(f"[SPLIT] Using committed lists from {SPLIT_DIR}")
        split_ids = {s: read_list(p) for s, p in lists.items()}
    else:
        val_scenes = set(choose_val_scenes(official_train))
        split_ids = {
            "train": [i for i in official_train if scene_of(i) not in val_scenes],
            "val": [i for i in official_train if scene_of(i) in val_scenes],
            "test": official_test,
        }
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        for s, p in lists.items():
            write_list(p, split_ids[s])
        print(f"[SPLIT] New lists written to {SPLIT_DIR}")

    # Checks: train and val partition the official train set, test is the official test
    # set, and no scene appears in more than one split.
    if sorted(split_ids["train"] + split_ids["val"]) != official_train:
        raise RuntimeError("train + val do not match the official train set")
    if split_ids["test"] != official_test:
        raise RuntimeError("test list does not match the official test set")
    scenes = {s: sorted({scene_of(i) for i in ids}) for s, ids in split_ids.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(scenes[a]) & set(scenes[b])
        if shared:
            raise RuntimeError(f"Scenes {sorted(shared)} are in both {a} and {b}")

    label_dir = {"train": "train", "val": "train", "test": "test"}
    rows = ["| Split | Folder | Scenes | Frames | Boxes |", "|---|---|---|---:|---:|"]
    for s, ids in split_ids.items():
        boxes = count_boxes(YOLO_ROOT / "visible" / label_dir[s], ids)
        rows.append(f"| {s} | `{OUT_FOLDERS[s]}` | {', '.join(scenes[s])} | {len(ids)} | {boxes} |")
        print(rows[-1])

    summary = [
        "# LLVIP split",
        "",
        f"Validation scenes are taken from the official train set (seed {SPLIT_SEED}, at least "
        f"{VAL_FRACTION:.0%} of its frames, whole scenes only). The official test set is unchanged "
        "and is used only for the final evaluation. Built by "
        "`scripts/data_preparation/llvip/make_llvip_split.py`.",
        "",
        *rows,
        "",
    ]
    (SPLIT_DIR / "split_summary.md").write_text("\n".join(summary), encoding="utf-8")

    for s in ("train", "val"):
        build_folder(s, split_ids[s])
        print(f"[BUILD] {OUT_FOLDERS[s]}: {len(split_ids[s])} pairs ready")
    print(f"Done: {YOLO_ROOT}")


if __name__ == "__main__":
    main()
