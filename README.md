# multispectral_detection

Multispectral (RGB + IR) drone detection with YOLOv11. Code for data preparation,
unimodal pretraining, weight transfer into a mid-fusion architecture, training, testing,
and validation on real-world field data.

The detector is a modification of [YOLOv11-RGBT](https://github.com/wandahangFY/YOLOv11-RGBT):
two parallel backbones (RGB, 3 channels; IR, 1 channel) with mid-fusion at level P3 (a
Zero Conv2d block, so the pretrained single-spectrum weights are not disturbed at the start
of training), followed by a shared neck and three detection heads. Input: two video
streams (visible + IR). Output: bounding boxes of drones in the visible-image coordinate
system.

## Results in brief

Test set: Anti-UAV-RGBT (synchronised pairs), 640x640, one class (drone).

| Model | Recall | Precision | AP@0.5 | F1 |
|---|---|---|---|---|
| YOLO11s RGB-only | 0.817 | 0.930 | 0.875 | 0.870 |
| YOLO11s IR-only | 0.848 | 0.963 | 0.907 | 0.902 |
| YOLO11s-RGBT from scratch | 0.893 | 0.986 | 0.951 | 0.937 |
| RGBT + frozen RGB/IR backbones + RGB neck/head | 0.938 | 0.987 | 0.977 | 0.962 |
| RGBT + frozen RGB/IR backbones + IR neck/head | 0.952 | 0.980 | 0.984 | 0.966 |

Mid-fusion helps most at night and in hard conditions (blur, glare, low contrast). On
the field data the from-scratch RGBT model is the best (AP@0.5
0.947); the frozen-backbone variants drop to 0.860-0.880, most likely because the
field sensor differs from the pretraining data.

## Pipeline

The work is organised as a chain of blocks. Each one maps to a folder under
`scripts/`.

### 1. Data preparation - `scripts/data_preparation/`

Anti-UAV-RGBT has RGB (1920x1080) and IR (640x512) videos that are not aligned in space
and whose time shift is not constant, so frames cannot be paired by frame number.
`anti_uav/sync_anti_uav_rgbt.py` builds paired training data in three steps:

- **Spatial alignment.** IR is warped into the RGB coordinate system with one affine
  transform (scale 2.8515, rotation -0.33 deg, offset 10 / -245 px; tuned by hand on
  static background). The same transform is applied to the box annotations.
- **Temporal synchronisation.** (a) For each scene a frame-difference signal is computed
  per stream (with the object masked), and a local RGB-IR shift is estimated in a sliding
  window (100 frames, step 50, shift up to 25 frames, normalised correlation, threshold
  0.18), then interpolated into a per-frame shift curve. (b) For every IR frame the RGB
  frame is chosen in a +-3 window around the estimated shift by maximum IoU of the
  object boxes.
- **Annotation merge.** The final box is the union of the RGB and IR boxes (YOLO
  format). Pairs whose union is more than 50% larger than the bigger box are dropped as
  badly synchronised.

The same pipeline is applied to the field data in `polygon/`: `merge_cvat_labels_ir_rgb.py`
merges separately annotated IR and RGB labels (CVAT) into one label per frame,
`crop_pad_to_1248.py` crops RGB/IR from 1920x1080 and pads to 1248x1248, and
`visualize_crop_selection.py` is a helper used to choose the crop window.

### 2. Unimodal pretraining - `scripts/training/pretrain/`

Two independent YOLOv11s models, one on RGB (3 ch) and one on IR (1 ch), trained on
unpaired data: Anti-UAV frames not used in the synchronised set plus the open datasets
listed in [DATA_SOURCES.md](DATA_SOURCES.md). Their backbone weights later initialise the
two branches of the multispectral network. Data configs: `configs/train/pretrain/`.

### 3. Baselines from scratch - `scripts/training/from_scratch/`

RGB-only, IR-only and RGBT mid-fusion (4 channels) models trained on the synchronised
Anti-UAV-RGBT data. They serve as reference points for the transfer experiments.
Data configs: `configs/train/from_scratch/`.

### 4. Weight transfer - `scripts/training/transfer/`

Based on the MCF (Multispectral Controllable Fine-Tuning) idea from YOLOv11-RGBT.
`init_pretrain_*_neckhead.py` builds an initialised RGBT checkpoint: RGB branch from the
RGB pretrain, IR branch from the IR pretrain, and the shared neck/head from either the IR
model or the RGB model (two variants). `init_pretrain_verify.py` checks tensor by tensor
that the transfer is exact. `train_rgbt_transfer_*_neckhead.py` then trains with both
backbones frozen, updating only the fusion blocks, neck and head. `weight_transfer.py`
holds the layer-name mappings.

### 5. Testing on Anti-UAV - `scripts/evaluation/anti_uav/`, `configs/test/anti_uav/`

- `eval_anti_uav_test.py` - runs Ultralytics `val()` for all five models on the full test
  set and on the day, night and hard-conditions (`meteo`) subsets (configs in
  `configs/test/anti_uav/`). Metrics: Recall, Precision, AP@0.5, F1. Results are saved as
  CSV and a markdown table.
- `speed_benchmark.py` - single-pair inference speed (RGB, IR, RGBT), FPS from the
  minimum of 10 timed `predict` calls; CPU by default.

### 6. Field validation - `scripts/training/polygon_finetune/`, `scripts/evaluation/polygon/`, `scripts/visualization/`

Real captures from a dual-channel optical system (iRay IRS-PT 4S4, visible 1920x1080,
IR 1280x1024). After the preparation steps from block 1 the models are fine-tuned on
1350 pairs and validated on 659 pairs from separate video sequences. Five models are
compared: RGB-only, IR-only, RGBT from scratch, RGBT transfer with RGB neck/head, RGBT
transfer with IR neck/head.

- `training/polygon_finetune/` - fine-tuning of all five models.
- `evaluation/polygon/eval_rgbt_poly_models.py` - Recall / Precision / F1 / IoU computed
  by explicit TP / FP / FN matching (frames contain 0-1 objects, so a standard mAP is a
  poor fit; recall at IoU 0.5 is used as the AP proxy).
- `visualization/polygon/render_side_by_side_poly.py` - side-by-side RGB | IR videos of
  predictions at several confidence thresholds.
- `visualization/plot_scene_shift.py` - diagnostic plots of the synchronisation
  (frame-difference signals, shift curve, final pairing) for one scene.

## Repository layout

```
configs/                 dataset configs (train / test)
scripts/
  paths.py               all absolute roots in one place (see "Paths")
  data_preparation/      anti_uav/ (sync, align, merge)   polygon/ (field data)
  training/              pretrain/  from_scratch/  transfer/  polygon_finetune/  shared/
  evaluation/            anti_uav/ (test metrics, speed)   polygon/ (custom TP/FP/FN evaluation)
  visualization/         side-by-side videos, synchronisation plots
DATA_SOURCES.md          links to every dataset used
requirements.txt
```

## Installation

Training relies on a custom fork of Ultralytics/YOLO that adds RGB / IR (grayscale) /
RGBT (4-channel mid-fusion) support:

- Repository: https://github.com/wandahangFY/YOLOv11-RGBT
- Commit used: `9cc2e208a3d3e8452e15b5ef47e1c536766aa1f7`

The fork's code is not included in this repo: clone it separately (it is used without
modifications) and install the dependencies:

```bash
git clone https://github.com/wandahangFY/YOLOv11-RGBT.git /home/YOLOv11-RGBT
cd /home/YOLOv11-RGBT && git checkout 9cc2e208a3d3e8452e15b5ef47e1c536766aa1f7
pip install -r /home/YOLOv11-RGBT/requirements.txt
pip install -r requirements.txt  # this repo's own scripts
```

`torch` / `torchvision` are not pinned in either requirements file -- install the build
matching your CUDA version from https://pytorch.org/get-started/locally/ first. The
server this project was developed on uses:

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```

(Python 3.11.12, NVIDIA driver 575.64.03 / CUDA 12.9, ffmpeg 4.4.2. `ffmpeg` must be on
`PATH` for the video rendering script.)

## Paths

Every absolute root the scripts need is defined once in `scripts/paths.py`. The defaults
are the paths of the original training server; override them with environment
variables to run on your own layout:

| Variable | Meaning | Default |
|---|---|---|
| `YOLO_REPO_ROOT` | clone of the YOLOv11-RGBT fork | `/home/YOLOv11-RGBT` |
| `DIPLOMA_ROOT` | where runs, checkpoints, configs and outputs are written | `/home/src/diploma` |
| `POLYGON_DATA_ROOT` | field (polygon) dataset | `/mnt/datasets/Maria_preprocess/mine_dpl` |
| `ANTIUAV_DATA_ROOT` | Anti-UAV dataset (raw and synchronised) | `/mnt/datasets/IR_DATA/ANTI-UAV` |

```bash
export YOLO_REPO_ROOT=/path/to/YOLOv11-RGBT
export ANTIUAV_DATA_ROOT=/path/to/ANTI-UAV
python -m scripts.training.pretrain.train_rgb_pretrain
```

Scripts are run as modules from the repo root. Each training script checks that the
imported `ultralytics` really comes from `YOLO_REPO_ROOT`, to avoid silently using a
system-installed package instead of the fork. Dataset YAML files under `configs/` still
contain the server paths and need to be edited for your data.
