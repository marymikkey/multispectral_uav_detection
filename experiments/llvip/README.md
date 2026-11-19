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

Runs are written to `$DIPLOMA_ROOT/train/runs/llvip_split/yolo11s_<modality>_llvip_seed<N>/`,
test results to `$DIPLOMA_ROOT/train/tests/llvip_split_test/` (`results.md`, `per_run.csv`).

## Results

Pending.
