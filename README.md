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
Ultralytics `val()`, see `scripts/evaluation/anti_uav/eval_anti_uav_test.py`.

| Model | Recall | Precision | AP@0.5 | F1 |
|---|---|---|---|---|
| YOLO11s RGB-only | 0.817 | 0.930 | 0.875 | 0.870 |
| YOLO11s IR-only | 0.848 | 0.963 | 0.907 | 0.902 |
| YOLO11s-RGBT from scratch | 0.893 | 0.986 | 0.951 | 0.937 |
| RGBT + frozen RGB/IR backbones + RGB neck/head | 0.938 | 0.987 | 0.977 | 0.962 |
| RGBT + frozen RGB/IR backbones + IR neck/head | 0.952 | 0.980 | 0.984 | 0.966 |

AP@0.5 by condition (test subsets in `configs/test/anti_uav/`):

| Model | Day | Night | Hard conditions |
|---|---|---|---|
| YOLO11s RGB-only | 0.915 | 0.267 | 0.720 |
| YOLO11s IR-only | 0.903 | 0.948 | 0.870 |
| YOLO11s-RGBT from scratch | 0.946 | 0.994 | 0.930 |
| RGBT + frozen RGB/IR backbones + RGB neck/head | 0.975 | 0.993 | 0.947 |
| RGBT + frozen RGB/IR backbones + IR neck/head | 0.983 | 0.994 | 0.966 |

Field validation: 659 pairs from separate video sequences, 1248x1248, after fine-tuning
on 1350 pairs. Ultralytics `val()`, see `scripts/evaluation/polygon/val_poly_models.py`.

| Model | Recall | Precision | AP@0.5 | F1 |
|---|---|---|---|---|
| YOLO11s RGB-only | 0.581 | 0.635 | 0.601 | 0.607 |
| YOLO11s IR-only | 0.813 | 0.870 | 0.853 | 0.841 |
| YOLO11s-RGBT from scratch | 0.893 | 0.939 | 0.947 | 0.916 |
| RGBT + frozen RGB/IR backbones + RGB neck/head | 0.826 | 0.853 | 0.860 | 0.839 |
| RGBT + frozen RGB/IR backbones + IR neck/head | 0.810 | 0.919 | 0.880 | 0.861 |

Mid-fusion helps most at night and in hard conditions (blur, glare, low contrast). On
the field data the from-scratch RGBT model is the best; the frozen-backbone variants drop.
Part of this drop comes from the grayscale RGB pretraining and the rest most likely from
the field sensor differing from the pretraining data (see the next section).

### Input modes and colour retraining

The fork decides how to read an image from its `use_simotm` argument. If the argument is
not set, the fork's default mode is used: the visible frame is read as grayscale and the
three input channels are filled with the gray image and two blurred copies. The RGB
pretraining and the RGB-only Anti-UAV model were trained without setting `use_simotm`, so
these two models, and the results above for them, use grayscale input. All other models
read their input as intended: the RGBT models take the visible frame in colour plus the
thermal frame (`use_simotm="RGBT"`), the IR models take the thermal frame (`"Gray"`), and
the field RGB-only model was fine-tuned in colour (`"RGB"`).

To see whether colour matters, these two models were retrained with
`use_simotm="RGB"` and nothing else changed (same data, hyperparameters and frozen layers),
and both RGBT transfer variants were re-initialised from the colour RGB pretrain, trained
on Anti-UAV-RGBT and fine-tuned on the field data in the same way as before:

| Model | `use_simotm` not set (default: grayscale input; results above) | `use_simotm="RGB"` (colour input) |
|---|---|---|
| RGB pretrain | `yolo11s_rgb_pretrain` / `rgb_pretrain.pt` | `yolo11s_rgb_pretrain_color` / `rgb_pretrain_color.pt` |
| RGB-only, Anti-UAV | `yolo11s_rgb_from_scratch` / `rgb_from_scratch.pt` | `yolo11s_rgb_from_scratch_color` / `rgb_from_scratch_color.pt` |
| RGBT transfer, RGB neck/head | `..._rgb_ir_pretrained_rgb_neck_head_freeze_backbones` / `rgbt_transfer_rgb_neckhead.pt` | `..._rgb_color_ir_pretrained_rgb_neck_head_freeze_backbones` / `rgbt_transfer_rgb_neckhead_color.pt` |
| RGBT transfer, IR neck/head | `..._rgb_ir_pretrained_ir_neck_head_freeze_backbones` / `rgbt_transfer_ir_neckhead.pt` | `..._rgb_color_ir_pretrained_ir_neck_head_freeze_backbones` / `rgbt_transfer_ir_neckhead_color.pt` |
| RGBT transfer, RGB neck/head, field | `..._rgb_neck_head_freeze_backbones_poly_finetune_crop_x287_1248` / `poly_rgbt_transfer_rgb_neckhead.pt` | `..._rgb_color_rgb_neck_head_freeze_backbones_poly_finetune_crop_x287_1248` / `poly_rgbt_transfer_rgb_neckhead_color.pt` |
| RGBT transfer, IR neck/head, field | `..._ir_neck_head_freeze_backbones_poly_finetune_crop_x287_1248` / `poly_rgbt_transfer_ir_neckhead.pt` | `..._rgb_color_ir_neck_head_freeze_backbones_poly_finetune_crop_x287_1248` / `poly_rgbt_transfer_ir_neckhead_color.pt` |

Run folders are under `train/runs/antiuav_resync/` (pretrain: `pretrain/runs/`). The
weights trained without `use_simotm` are in release v1.0, the colour ones in v1.1; both
sets are on Hugging Face.

Anti-UAV-RGBT test set, Recall / Precision / AP@0.5 / F1:

| Model | `use_simotm` not set (grayscale) | `use_simotm="RGB"` (colour) |
|---|---|---|
| RGB-only | 0.817 / 0.930 / 0.875 / 0.870 | 0.787 / 0.923 / 0.854 / 0.850 |
| RGBT transfer, RGB neck/head | 0.938 / 0.987 / 0.977 / 0.962 | 0.951 / 0.988 / 0.981 / 0.969 |
| RGBT transfer, IR neck/head | 0.952 / 0.980 / 0.984 / 0.966 | 0.947 / 0.987 / 0.982 / 0.966 |

AP@0.5 by condition, `use_simotm` not set → `use_simotm="RGB"`:

| Model | Day | Night | Hard conditions |
|---|---|---|---|
| RGB-only | 0.915 → 0.902 | 0.267 → 0.210 | 0.720 → 0.675 |
| RGBT transfer, RGB neck/head | 0.975 → 0.980 | 0.993 → 0.994 | 0.947 → 0.960 |
| RGBT transfer, IR neck/head | 0.983 → 0.980 | 0.994 → 0.994 | 0.966 → 0.963 |

Field validation (659 pairs), Recall / Precision / AP@0.5 / F1:

| Model | `use_simotm` not set (grayscale) | `use_simotm="RGB"` (colour) |
|---|---|---|
| RGBT transfer, RGB neck/head | 0.826 / 0.853 / 0.860 / 0.839 | 0.848 / 0.879 / 0.893 / 0.863 |
| RGBT transfer, IR neck/head | 0.810 / 0.919 / 0.880 / 0.861 | 0.870 / 0.875 / 0.908 / 0.872 |

Colour input does not help the RGB-only detector: it is worse than the grayscale one on
the full test set and on every subset, so the grayscale baseline above is not understated.
For weight transfer, the colour RGB pretrain matters more. On Anti-UAV-RGBT it improves
the RGB neck/head variant on every test subset (up to +0.013 AP@0.5 in hard conditions)
and leaves the IR neck/head variant unchanged. On the field data it improves both variants
by about 0.03 AP@0.5 and reduces missed detections, which narrows the gap to the
from-scratch RGBT model (AP@0.5 0.947) from 0.067-0.087 to 0.039-0.054. With grayscale
pretraining, the frozen RGB backbone received colour frames it had not been trained on;
this explains part of the field drop of the transfer variants. The from-scratch RGBT model
is still the best on the field data, and on Anti-UAV-RGBT RGBT with transferred weights
remains the best model.

## Pipeline

The work is organised as a chain of blocks. Each one maps to a folder under
`scripts/`.

### 1. Data preparation - `scripts/data_preparation/`

Anti-UAV-RGBT has RGB (1920x1080) and IR (640x512) videos that are not aligned in space
and whose time shift is not constant, so frames cannot be paired by frame number.
`anti_uav/videos_to_frames.py` first converts the raw videos and json annotations into
per-frame images with YOLO labels (`ANTI-UAV_each_frame`). `anti_uav/sync_anti_uav_rgbt.py`
then builds paired training data in three steps:

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

`anti_uav/group_by_scene.py` then regroups the synchronised test set by scene, the layout
used by the day / night / hard-conditions test configs.

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
- `evaluation/polygon/val_poly_models.py` - Ultralytics `val()` of all five models on the
  held-out field interval (659 pairs, 1248x1248): Recall, Precision, AP@0.5, F1. This
  gives the field-validation results.
- `evaluation/polygon/eval_rgbt_poly_models.py` - additional analysis: explicit TP / FP / FN
  matching at several confidence thresholds, with and without a one-detection-per-frame
  limit (frames contain 0-1 objects), plus mean IoU.
- `visualization/polygon/render_side_by_side_poly.py` - side-by-side RGB | IR videos of
  predictions at several confidence thresholds.
- `visualization/plot_scene_shift.py` - diagnostic plots of the synchronisation
  (frame-difference signals, shift curve, final pairing) for one scene.

### Other domains - `experiments/llvip/`, `experiments/flir/`

The same RGB / IR / RGBT mid-fusion comparison on other datasets:

- LLVIP (night street scenes, person), YOLO11s with the Anti-UAV settings. Test split:
  AP@0.5 0.873 (RGB), 0.955 (IR), 0.958 (RGBT); AP@0.5:0.95 0.488 / 0.617 / 0.625.
  Details: [experiments/llvip/README.md](experiments/llvip/README.md).
- FLIR ADAS aligned (person, car, bicycle), YOLO11n. Details:
  [experiments/flir/README.md](experiments/flir/README.md).

On both, RGBT is better than either single-spectrum model.

## Repository layout

```
configs/                 dataset configs (train / test / llvip / flir)
experiments/             other domains: llvip/ (LLVIP), flir/ (FLIR ADAS)
scripts/
  paths.py               all absolute roots in one place (see "Paths")
  data_preparation/      anti_uav/ (sync, align, merge)   polygon/ (field data)   llvip/
  training/              pretrain/  from_scratch/  transfer/  polygon_finetune/  shared/
  evaluation/            anti_uav/ (test metrics, speed)   polygon/ (field validation, TP/FP/FN analysis)
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

## Weights

All trained checkpoints (2 pretrain models, 5 Anti-UAV models, 5 field fine-tuned models)
are on Hugging Face: https://huggingface.co/marymikkey/multispectral_uav_detection

The folder layout there matches `DIPLOMA_ROOT`, so after downloading the evaluation
and transfer scripts run without path changes:

```bash
pip install huggingface_hub
hf download marymikkey/multispectral_uav_detection --local-dir $DIPLOMA_ROOT
```

The same files are attached to the GitHub release
[v1.0](https://github.com/marymikkey/multispectral_uav_detection/releases/tag/v1.0) under
flat names. To use them, save each one as `weights/best.pt` in the run folder below
(relative to `DIPLOMA_ROOT`):

| Release file | Run folder |
|---|---|
| `rgb_pretrain.pt` | `pretrain/runs/yolo11s_rgb_pretrain` |
| `ir_pretrain.pt` | `pretrain/runs/yolo11s_gray_ir_pretrain_100ep` |
| `rgb_from_scratch.pt` | `train/runs/antiuav_resync/yolo11s_rgb_from_scratch` |
| `ir_from_scratch.pt` | `train/runs/antiuav_resync/yolo11s_ir_from_scratch` |
| `rgbt_from_scratch.pt` | `train/runs/antiuav_resync/yolo11s_rgbt_midfusion_p3_from_scratch` |
| `rgbt_transfer_rgb_neckhead.pt` | `train/runs/antiuav_resync/yolo11s_rgbt_midfusion_p3_rgb_ir_pretrained_rgb_neck_head_freeze_backbones` |
| `rgbt_transfer_ir_neckhead.pt` | `train/runs/antiuav_resync/yolo11s_rgbt_midfusion_p3_rgb_ir_pretrained_ir_neck_head_freeze_backbones` |
| `poly_rgb_from_scratch.pt` | `train/runs/antiuav_resync/yolo11s_rgb_from_scratch_poly_finetune_crop_x287_1248` |
| `poly_ir_from_scratch.pt` | `train/runs/antiuav_resync/yolo11s_ir_from_scratch_poly_finetune_crop_x287_1248` |
| `poly_rgbt_from_scratch.pt` | `train/runs/antiuav_resync/yolo11s_rgbt_midfusion_p3_from_scratch_poly_finetune_crop_x287_1248` |
| `poly_rgbt_transfer_rgb_neckhead.pt` | `train/runs/antiuav_resync/yolo11s_rgbt_midfusion_p3_rgb_neck_head_freeze_backbones_poly_finetune_crop_x287_1248` |
| `poly_rgbt_transfer_ir_neckhead.pt` | `train/runs/antiuav_resync/yolo11s_rgbt_midfusion_p3_ir_neck_head_freeze_backbones_poly_finetune_crop_x287_1248` |

The colour-retrained models are in release
[v1.1](https://github.com/marymikkey/multispectral_uav_detection/releases/tag/v1.1) (see
"Input modes and colour retraining"), the LLVIP models in
[v1.2](https://github.com/marymikkey/multispectral_uav_detection/releases/tag/v1.2)
(`llvip_{rgb,ir,rgbt}.pt`, run folders `train/runs/llvip/yolo11s_{rgb,ir,rgbt}_llvip`).
Both are on Hugging Face as well, in the same layout.

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

## Run order

All commands are run from the repository root.

```bash
# 1. data (Anti-UAV)
python -m scripts.data_preparation.anti_uav.videos_to_frames
python -m scripts.data_preparation.anti_uav.sync_anti_uav_rgbt
python -m scripts.data_preparation.anti_uav.group_by_scene        # test subsets by scene

# 2. unimodal pretraining
python -m scripts.training.pretrain.train_rgb_pretrain
python -m scripts.training.pretrain.train_ir_pretrain

# 3. baselines from scratch
python -m scripts.training.from_scratch.train_rgb_from_scratch
python -m scripts.training.from_scratch.train_ir_from_scratch
python -m scripts.training.from_scratch.train_rgbt_from_scratch

# 4. weight transfer
python -m scripts.training.transfer.init_pretrain_ir_neckhead
python -m scripts.training.transfer.init_pretrain_rgb_neckhead
python -m scripts.training.transfer.init_pretrain_verify
python -m scripts.training.transfer.train_rgbt_transfer_ir_neckhead
python -m scripts.training.transfer.train_rgbt_transfer_rgb_neckhead

# 5. testing
python -m scripts.evaluation.anti_uav.eval_anti_uav_test
python -m scripts.evaluation.anti_uav.speed_benchmark

# 6. field data
python -m scripts.data_preparation.polygon.merge_cvat_labels_ir_rgb
python -m scripts.data_preparation.polygon.crop_pad_to_1248
python -m scripts.training.polygon_finetune.train_rgb_ir_poly_finetune
python -m scripts.training.polygon_finetune.train_rgbt_poly_finetune
python -m scripts.evaluation.polygon.val_poly_models
python -m scripts.evaluation.polygon.eval_rgbt_poly_models
python -m scripts.visualization.polygon.render_side_by_side_poly
```

Training scripts never overwrite an existing run folder (`exist_ok=False`): if the folder
already exists, Ultralytics writes to a new one with a numeric suffix (`name2`), and the
next stages will not find it under the expected name. To rerun a stage, point
`DIPLOMA_ROOT` at a fresh directory.
