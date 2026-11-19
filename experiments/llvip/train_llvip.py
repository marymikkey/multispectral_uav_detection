#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train the RGB-only, IR-only or RGBT mid-fusion P3 model on LLVIP from scratch.

Training uses train_split, model selection (best.pt) uses val_split; the official test
set is not touched here (see val_llvip.py). All modalities share the settings in
common.py. Each seed is a separate run:

    python -m experiments.llvip.train_llvip rgbt                 # seeds 0 1 2
    python -m experiments.llvip.train_llvip rgbt --seeds 0
    python -m experiments.llvip.train_llvip rgb --batch 64       # override a setting
    python -m experiments.llvip.train_llvip rgbt --resume        # after an interruption

With --resume, finished runs (run_meta.json present) are skipped and an interrupted run
continues from its weights/last.pt with the settings stored in the checkpoint.

Runs: $DIPLOMA_ROOT/train/runs/llvip_split/yolo11s_<modality>_llvip_seed<N>/
Each run folder keeps Ultralytics' args.yaml and results.csv plus run_meta.json
(repo and fork commits, data config, split lists, duration).
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from experiments.llvip.common import (
    DEFAULT_SEEDS,
    MODALITIES,
    PROJECT_DIR,
    TRAIN_ARGS,
    run_name,
    save_run_meta,
    setup_fork,
    write_data_yaml,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modality", choices=MODALITIES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--batch", type=int, default=TRAIN_ARGS["batch"])
    parser.add_argument("--epochs", type=int, default=TRAIN_ARGS["epochs"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true",
                        help="skip finished runs, continue interrupted ones from weights/last.pt")
    args = parser.parse_args()

    setup_fork()
    from ultralytics import YOLO
    from scripts.training.shared.utils import check_cuda

    check_cuda(int(str(args.device).split(",")[0]))

    spec = MODALITIES[args.modality]
    data_yaml = write_data_yaml(args.modality)

    for seed in args.seeds:
        name = run_name(args.modality, seed)
        run_dir = PROJECT_DIR / name
        last = run_dir / "weights" / "last.pt"
        train_args = dict(TRAIN_ARGS, batch=args.batch, epochs=args.epochs)
        resumed = False

        if run_dir.exists():
            if not args.resume:
                raise RuntimeError(f"{run_dir} already exists; use --resume to continue it, or remove it")
            if (run_dir / "run_meta.json").exists():
                print(f"[SKIP] {run_dir} is finished")
                continue
            if not last.exists():
                raise RuntimeError(f"{run_dir} has no weights/last.pt to resume from; remove it and rerun")
            resumed = True

        start = time.time()
        start_dt = datetime.now()
        if resumed:
            print(f"[RESUME] {run_dir} from {last}")
            model = YOLO(str(last))
            model.train(resume=True, device=args.device)
        else:
            print(f"[RUN] {run_dir}  model={spec['model_cfg']}  data={data_yaml}  seed={seed}")
            model = YOLO(spec["model_cfg"])
            model.train(data=str(data_yaml), seed=seed, device=args.device,
                        project=str(PROJECT_DIR), name=name, **spec["load"], **train_args)

        save_dir = Path(model.trainer.save_dir)
        save_run_meta(save_dir, {
            "modality": args.modality,
            "seed": seed,
            "model_cfg": spec["model_cfg"],
            "data_yaml": str(data_yaml),
            "load_args": spec["load"],
            "train_args": train_args,
            "start_time": start_dt.isoformat(timespec="seconds"),
            "duration_sec": round(time.time() - start, 1),
            # A resumed run continues from last.pt: duration covers only the last part, and
            # random number generator state is not restored, so it is not bit-identical.
            "resumed": resumed,
        })
        print(f"[DONE] {save_dir}")


if __name__ == "__main__":
    main()
