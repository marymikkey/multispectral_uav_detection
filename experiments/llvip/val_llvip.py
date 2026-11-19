#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final evaluation of the LLVIP models trained by train_llvip.py on the official LLVIP
test set (Ultralytics val(), split="test"). Metrics: Recall, Precision, AP@0.5,
AP@0.5:0.95, F1. Every run found for the requested modalities and seeds is evaluated;
the summary gives mean and standard deviation over seeds.

    python -m experiments.llvip.val_llvip                     # all modalities, seeds 0 1 2
    python -m experiments.llvip.val_llvip rgbt --seeds 0

Results: $DIPLOMA_ROOT/train/tests/llvip_split_test/{per_run.csv, results.md}
"""

import argparse
import csv
import statistics

from experiments.llvip.common import (
    DEFAULT_SEEDS,
    MODALITIES,
    PROJECT_DIR,
    TEST_DIR,
    TRAIN_ARGS,
    run_name,
    setup_fork,
    write_data_yaml,
)

METRICS = ["recall", "precision", "ap50", "ap50_95", "f1"]
HEADERS = ["Recall", "Precision", "AP@0.5", "AP@0.5:0.95", "F1"]


def evaluate(weights, data_yaml, load_args, name, device):
    from ultralytics import YOLO

    r = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=TRAIN_ARGS["imgsz"],
                               batch=TRAIN_ARGS["batch"], device=device, single_cls=True,
                               plots=False, verbose=False, project=str(TEST_DIR / "val_runs"),
                               name=name, exist_ok=True, **load_args)
    p, rc = float(r.box.mp), float(r.box.mr)
    return {
        "recall": rc,
        "precision": p,
        "ap50": float(r.box.map50),
        "ap50_95": float(r.box.map),
        "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modalities", nargs="*", default=list(MODALITIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    setup_fork()
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for modality in args.modalities:
        data_yaml = write_data_yaml(modality)
        for seed in args.seeds:
            name = run_name(modality, seed)
            weights = PROJECT_DIR / name / "weights" / "best.pt"
            if not weights.exists():
                print(f"[SKIP] {weights} not found")
                continue
            m = evaluate(weights, data_yaml, MODALITIES[modality]["load"], name, args.device)
            rows.append({"modality": modality, "seed": seed, **m})
            print(f"[{name}] " + "  ".join(f"{h} {m[k]:.3f}" for k, h in zip(METRICS, HEADERS)))

    if not rows:
        raise RuntimeError(f"No trained runs found in {PROJECT_DIR}")

    with open(TEST_DIR / "per_run.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["modality", "seed", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "LLVIP official test set, mean ± std over seeds (n = number of seeds).",
        "",
        "| Model | n | " + " | ".join(HEADERS) + " |",
        "|---|---:|" + "---:|" * len(HEADERS),
    ]
    for modality in args.modalities:
        runs = [r for r in rows if r["modality"] == modality]
        if not runs:
            continue
        cells = []
        for k in METRICS:
            values = [r[k] for r in runs]
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{statistics.mean(values):.3f} ± {std:.3f}")
        lines.append(f"| {modality} | {len(runs)} | " + " | ".join(cells) + " |")

    lines += ["", "Per run:", "", "| Run | " + " | ".join(HEADERS) + " |", "|---|" + "---:|" * len(HEADERS)]
    for r in rows:
        lines.append(f"| {run_name(r['modality'], r['seed'])} | "
                     + " | ".join(f"{r[k]:.3f}" for k in METRICS) + " |")

    (TEST_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nResults: {TEST_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
