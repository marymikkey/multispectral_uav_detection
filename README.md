# multispectral_detection
Multispectral (RGB + IR) UAV detection with YOLOv11 mid-fusion.

## External dependencies

Training relies on a custom fork of Ultralytics/YOLO that adds RGB / IR (grayscale) /
RGBT (4-channel mid-fusion) support:

- Repository: https://github.com/wandahangFY/YOLOv11-RGBT
- Commit/tag used: `9cc2e208a3d3e8452e15b5ef47e1c536766aa1f7`

This repo is used as-is (not modified) and is **not** vendored or added as a submodule
here — it's a separate external dependency. Clone it next to this project and point
training scripts at it:

```bash
git clone https://github.com/wandahangFY/YOLOv11-RGBT.git /home/YOLOv11-RGBT
cd /home/YOLOv11-RGBT && git checkout 9cc2e208a3d3e8452e15b5ef47e1c536766aa1f7
pip install -r /home/YOLOv11-RGBT/requirements.txt
pip install -r requirements.txt  # this repo's own scripts (see requirements.txt)
```

`torch`/`torchvision` aren't pinned in either requirements file -- install the build
matching your CUDA version from https://pytorch.org/get-started/locally/ first. The
training server this project was developed on uses:

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```

(Python 3.11.12, NVIDIA driver 575.64.03 / CUDA 12.9, ffmpeg 4.4.2 -- `ffmpeg` must be
on `PATH` for scripts/visualization/polygon/shared_poly_viz.py to encode videos.)

Every training script sets `REPO_ROOT = Path("/home/YOLOv11-RGBT")` before importing
`ultralytics`, and checks that the imported `ultralytics.__file__` actually resolves
inside `REPO_ROOT`, to guard against accidentally picking up a system-installed
`ultralytics` package instead of this fork.
