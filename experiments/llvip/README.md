# LLVIP: second domain

The RGB-only, IR-only and RGBT mid-fusion P3 models trained on
[LLVIP](https://bupt-ai-cz.github.io/LLVIP/) (Jia et al., ICCV 2021): 15,488 aligned
visible-thermal pairs, 1280x1024, mostly night street scenes, one class (person). The
pairs are already aligned, so the alignment and synchronisation steps used for
Anti-UAV-RGBT are not needed.

## Protocol

- **Split.** LLVIP has an official train set (12,025 pairs, scenes 01-18 and 25) and test
  set (3,463 pairs, scenes 19-24 and 26), but no validation set. Whole scenes are moved
  from the official train set into a validation set (seed 0, at least 12% of the train
  frames). Validation is used for early stopping and for choosing `best.pt`; the official
  test set is used only for the final evaluation. The frame lists are committed in
  [`splits/`](splits/), so every run uses the same split; scene and box counts are in
  [`splits/split_summary.md`](splits/split_summary.md).
- **Training.** YOLO11s from scratch, imgsz 640, 100 epochs with early stopping after
  30, AdamW, lr 1e-3 with cosine decay. The same settings for all three modalities
  ([`common.py`](common.py)): batch 32, same augmentation. The RGB model reads the
  visible frame in colour (`use_simotm="RGB"`), the IR model the thermal frame
  (`"Gray"`), the RGBT model both (`"RGBT"`).
- **Seeds.** Each model is trained with seeds 0, 1 and 2; results are reported as mean ±
  standard deviation over seeds.
- **Logged with each run:** Ultralytics `args.yaml` and `results.csv`, and `run_meta.json`
  with the repository and fork commits, data config, split lists and training time.

## Data

Download LLVIP from the official page (access form) and unpack it into
`$LLVIP_DATA_ROOT` (default `/mnt/datasets/Maria_preprocess/llvip`), which gives
`LLVIP/{visible,infrared}/{train,test}/*.jpg` and `LLVIP/Annotations/*.xml`. Then:

```bash
python -m scripts.data_preparation.llvip.prepare_llvip     # VOC -> YOLO, LLVIP_yolo/{visible,infrared}/{train,test}/
python -m scripts.data_preparation.llvip.make_llvip_split  # LLVIP_yolo/{visible,infrared}/{train_split,val_split}/
```

`prepare_llvip` writes the images as symlinks with YOLO labels next to them; the same
labels are used for both modalities, since the pairs are aligned. `make_llvip_split`
reads the committed lists in `splits/` and builds the train and validation folders from
them (`--regenerate` chooses the validation scenes again).

## Run

From the repository root:

```bash
python -m experiments.llvip.train_llvip rgb      # seeds 0 1 2; --seeds 0 for one run
python -m experiments.llvip.train_llvip ir
python -m experiments.llvip.train_llvip rgbt
python -m experiments.llvip.val_llvip            # official test set, all runs, mean ± std
```

If training stops (for example, the container loses the GPU), rerun the same command
with `--resume`: finished runs are skipped and an interrupted run continues from its
`weights/last.pt`.

```bash
for m in rgb ir rgbt; do python -m experiments.llvip.train_llvip $m --resume || break; done
```

Runs are written to `$DIPLOMA_ROOT/train/runs/llvip_split/yolo11s_<modality>_llvip_seed<N>/`,
test results to `$DIPLOMA_ROOT/train/tests/llvip_split_test/` (`results.md`, `per_run.csv`).

## Results

Official LLVIP test set (3,463 pairs, 8,302 people), `best.pt` of each run chosen on the
validation split. Mean ± sample standard deviation over seeds 0, 1, 2:

| Model | Recall | Precision | AP@0.5 | AP@0.5:0.95 | F1 |
|---|---:|---:|---:|---:|---:|
| RGB-only | 0.767 ± 0.022 | 0.857 ± 0.007 | 0.845 ± 0.014 | 0.463 ± 0.014 | 0.809 ± 0.013 |
| IR-only | 0.849 ± 0.012 | 0.922 ± 0.008 | 0.930 ± 0.010 | 0.581 ± 0.015 | 0.884 ± 0.010 |
| RGBT mid-fusion P3 | **0.882 ± 0.004** | **0.950 ± 0.005** | **0.954 ± 0.002** | **0.618 ± 0.001** | **0.915 ± 0.004** |

Per run:

| Run | Recall | Precision | AP@0.5 | AP@0.5:0.95 | F1 |
|---|---:|---:|---:|---:|---:|
| RGB, seed 0 | 0.788 | 0.863 | 0.861 | 0.479 | 0.823 |
| RGB, seed 1 | 0.744 | 0.860 | 0.836 | 0.456 | 0.798 |
| RGB, seed 2 | 0.770 | 0.849 | 0.837 | 0.454 | 0.807 |
| IR, seed 0 | 0.859 | 0.931 | 0.936 | 0.584 | 0.893 |
| IR, seed 1 | 0.836 | 0.916 | 0.918 | 0.565 | 0.874 |
| IR, seed 2 | 0.854 | 0.920 | 0.935 | 0.594 | 0.886 |
| RGBT, seed 0 | 0.878 | 0.945 | 0.952 | 0.616 | 0.910 |
| RGBT, seed 1 | 0.885 | 0.951 | 0.956 | 0.618 | 0.917 |
| RGBT, seed 2 | 0.884 | 0.954 | 0.955 | 0.619 | 0.918 |

- RGBT is ahead of IR-only on every metric: +2.4 pp AP@0.5, +3.7 pp AP@0.5:0.95,
  +3.1 pp F1. The weakest RGBT run is better than the strongest IR run on every metric,
  so with three seeds the gap is larger than the seed-to-seed spread.
- RGBT also varies least between seeds (standard deviation 0.001-0.005 against
  0.007-0.022 for the single-spectrum models).
- With three seeds the standard deviations are rough estimates.

Training stopped early (no improvement on validation for 30 epochs) at epoch 84 and 86
for RGB seeds 0 and 2, 75 for IR seed 2, 89 and 96 for RGBT seeds 0 and 1; the other runs
reached 100 epochs. Approximate time per epoch on one A40: 75-78 s (RGB), 71 s (IR),
102-103 s (RGBT).
