# FLIR ADAS cross-domain check

The same three configurations as in the main pipeline (RGB, IR, RGBT mid-fusion P3),
trained on FLIR ADAS aligned RGB-T pairs (3 classes: person, car, bicycle; 512x640).
Models are YOLO11**n**, so the numbers are not comparable with the YOLO11s results on
Anti-UAV.

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| RGB | 0.585 | 0.278 | 0.722 | 0.518 |
| IR | 0.705 | 0.372 | 0.738 | 0.635 |
| RGBT mid-fusion P3 | 0.743 | 0.390 | 0.778 | 0.668 |

Settings: batch 32, imgsz 640, patience 20, 200 epochs (RGB) / 300 epochs (IR, RGBT),
optimizer `auto`, pretrained.

## Data

Expected layout (path in `configs/flir/*.yaml`):

```
FLIR3C/
  visible/{train,test}/   *.jpg + *.txt (YOLO labels)
  infrared/{train,test}/  *.jpg + *.txt
```

Pairs share file names. The test folder is used as the validation split. Of 5142 stems,
5137 have both images (5 RGB images are missing), and 7 IR images are stored with 3
channels instead of 1.

## Run

From the repository root:

```bash
python -m experiments.flir.train_flir rgbt     # rgb | ir | rgbt
python -m experiments.flir.val_flir rgbt
```

Runs are written to `$DIPLOMA_ROOT/experiments/flir/yolo11n_<modality>/`.
