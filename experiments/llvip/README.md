# LLVIP: second domain

The RGB-only, IR-only and RGBT mid-fusion P3 models trained on
[LLVIP](https://bupt-ai-cz.github.io/LLVIP/) (Jia et al., ICCV 2021): 15,488 aligned
visible-thermal pairs, 1280x1024, mostly night street scenes, one class (person).
Model and hyperparameters are the same as in the Anti-UAV from-scratch stage (YOLO11s,
imgsz 640, 100 epochs with early stopping after 30, AdamW, lr 1e-3, batch 32 for RGB and
IR, 64 for RGBT). The RGB model reads the visible frame in colour (`use_simotm="RGB"`).

Test split, 3,463 pairs, 8,302 people:

| Model | Recall | Precision | AP@0.5 | AP@0.5:0.95 | F1 |
|---|---|---|---|---|---|
| RGB-only | 0.797 | 0.888 | 0.873 | 0.488 | 0.840 |
| IR-only | 0.890 | 0.941 | 0.955 | 0.617 | 0.914 |
| RGBT mid-fusion P3 | 0.896 | 0.951 | 0.958 | 0.625 | 0.923 |

The ranking is the same as on Anti-UAV-RGBT: RGBT is better than both single-spectrum
models on every metric. In these night scenes the thermal frame carries most of the
information, so the gain of RGBT over IR-only is small, and the RGB-only model falls well
behind. LLVIP has no separate validation split; as in other work on this dataset, the
test split is also used for validation.

## Data

Download LLVIP from the official page (access form) and unpack it into
`$LLVIP_DATA_ROOT` (default `/mnt/datasets/Maria_preprocess/llvip`), which gives
`LLVIP/{visible,infrared}/{train,test}/*.jpg` and `LLVIP/Annotations/*.xml`. Then convert
it into the layout read by the fork:

```bash
python -m scripts.data_preparation.llvip.prepare_llvip
```

This writes `LLVIP_yolo/{visible,infrared}/{train,test}/` with the images as symlinks and
YOLO labels next to them; the same labels are used for both modalities, since the pairs
are aligned.

## Run

From the repository root:

```bash
python -m experiments.llvip.train_llvip rgbt     # rgb | ir | rgbt
python -m experiments.llvip.val_llvip            # evaluates all trained models
```

Runs are written to `$DIPLOMA_ROOT/train/runs/llvip/yolo11s_<modality>_llvip/`, results to
`$DIPLOMA_ROOT/train/tests/llvip_test/results.md`.
