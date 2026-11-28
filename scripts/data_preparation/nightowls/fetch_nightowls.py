#!/usr/bin/env python3
"""
Download a subset of NightOwls frames without downloading the whole archive.

The official archives are large (training 135 GB, validation 50 GB). A zip file keeps a
directory of its members at the end, so with HTTP range requests we read only that
directory and then only the selected frames.

Selection: frames with at least one box of --select-classes; within each recording,
every --step-th of them (the frames are video, neighbours are nearly identical).
Labels are written in YOLO format with one class, 0 = person, from the boxes of
--label-classes. By default riders count as person, as in LLVIP. Boxes marked ignore
are skipped; the number of selected frames that contain them is printed.

    python fetch_nightowls.py --split training   --step 10 --out /mnt/datasets/Maria_preprocess/nightowls
    python fetch_nightowls.py --split validation --step 10 --out /mnt/datasets/Maria_preprocess/nightowls
    python fetch_nightowls.py --split training --step 10 --out ... --dry-run   # counts only
    python fetch_nightowls.py --split training --step 10 --out ... --labels-only   # rewrite labels

Expects nightowls_<split>.json in --out (download it first, ~32 MB / 10 MB).
Interrupted runs can be restarted: existing images are skipped.
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict

BASE_URL = "https://thor.robots.ox.ac.uk/nightowls/python"


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file via HTTP Range requests."""

    def __init__(self, url, retries=5):
        self.url, self.retries, self.pos = url, retries, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f"server answered {r.status}, range requests not supported")
                    data = r.read()
                break
            except RuntimeError:
                raise
            except Exception as e:  # network hiccup: retry
                if attempt == self.retries - 1:
                    raise
                print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
                time.sleep(2 ** attempt)
        self.requests += 1
        n = len(data)
        b[:n] = data
        self.pos += n
        return n


def select_images(ann, step, select_classes, label_classes):
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    sel_ids = {i for i, n in cats.items() if n in select_classes}
    lbl_ids = {i for i, n in cats.items() if n in label_classes}
    sel_frames, boxes, ignored = set(), defaultdict(list), set()
    for a in ann["annotations"]:
        if a.get("ignore", False) or cats.get(a["category_id"]) == "ignore":
            ignored.add(a["image_id"])
            continue
        if a["category_id"] in sel_ids:
            sel_frames.add(a["image_id"])
        if a["category_id"] in lbl_ids:
            boxes[a["image_id"]].append(a["bbox"])

    # Group frames by recording when the field exists, keep temporal order.
    rec_key = next((k for k in ("recordings_id", "recording_id", "video_id", "sequence_id")
                    if ann["images"] and k in ann["images"][0]), None)
    order_key = next((k for k in ("timestamp", "frame_id", "frame_number", "id")
                      if ann["images"] and k in ann["images"][0]), "id")
    groups = defaultdict(list)
    for img in ann["images"]:
        if img["id"] in sel_frames:
            groups[img.get(rec_key) if rec_key else 0].append(img)
    selected = []
    for g in groups.values():
        g.sort(key=lambda im: im[order_key])
        selected.extend(g[::step])
    return cats, boxes, selected, rec_key, ignored


def yolo_lines(bboxes, w, h):
    out = []
    for x, y, bw, bh in bboxes:
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        out.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["training", "validation"], required=True)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default=None, help="archive URL (default: official)")
    p.add_argument("--select-classes", default="pedestrian",
                   help="comma-separated categories a frame must contain to be selected")
    p.add_argument("--label-classes", default="pedestrian,bicycledriver,motorbikedriver",
                   help="comma-separated categories written as person")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--labels-only", action="store_true",
                   help="rewrite label files for the selected frames, do not download")
    args = p.parse_args()

    ann = json.load(open(os.path.join(args.out, f"nightowls_{args.split}.json")))
    cats, boxes, selected, rec_key, ignored = select_images(
        ann, args.step, set(args.select_classes.split(",")), set(args.label_classes.split(",")))
    n_boxes = sum(len(boxes[im["id"]]) for im in selected)
    n_ign = sum(im["id"] in ignored for im in selected)
    print(f"[{args.split}] categories: {cats}")
    print(f"[{args.split}] images: {len(ann['images'])}, grouped by: {rec_key or 'none'}, "
          f"selected (step {args.step}): {len(selected)}, person boxes: {n_boxes}, "
          f"selected frames with ignore boxes: {n_ign}")
    if args.dry_run:
        return

    img_dir = os.path.join(args.out, "images", args.split)
    lbl_dir = os.path.join(args.out, "labels", args.split)
    os.makedirs(lbl_dir, exist_ok=True)
    if args.labels_only:
        for im in selected:
            name = os.path.splitext(os.path.basename(im["file_name"]))[0]
            lines = yolo_lines(boxes[im["id"]], im["width"], im["height"])
            with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"[{args.split}] labels written: {len(selected)}")
        return

    url = args.url or f"{BASE_URL}/nightowls_{args.split}.zip"
    remote = HttpRangeFile(url)
    print(f"[{args.split}] archive: {remote.size / 1e9:.1f} GB, reading its directory...")
    zf = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))
    members = {os.path.basename(i.filename): i for i in zf.infolist() if not i.is_dir()}
    print(f"[{args.split}] {len(members)} files in archive ({remote.requests} requests so far)")

    os.makedirs(img_dir, exist_ok=True)

    done = missing = 0
    t0 = time.time()
    for k, im in enumerate(selected, 1):
        name = os.path.basename(im["file_name"])
        dst = os.path.join(img_dir, name)
        lines = yolo_lines(boxes[im["id"]], im["width"], im["height"])
        with open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if os.path.exists(dst):
            done += 1
            continue
        info = members.get(name)
        if info is None:
            missing += 1
            continue
        tmp = dst + ".part"
        with zf.open(info) as src, open(tmp, "wb") as f:
            f.write(src.read())
        os.replace(tmp, dst)
        done += 1
        if k % 200 == 0 or k == len(selected):
            rate = k / max(time.time() - t0, 1e-6)
            print(f"  {k}/{len(selected)}  {rate:.1f} img/s  eta {(len(selected) - k) / rate / 60:.0f} min")
    print(f"[{args.split}] done: {done}, not found in archive: {missing}, output: {args.out}")


if __name__ == "__main__":
    main()
