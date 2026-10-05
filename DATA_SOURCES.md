# Data sources

## Multispectral dataset (paired RGB + IR)

**Anti-UAV (Anti-UAV-RGBT)** - CVPR 2023 Anti-UAV Workshop & Challenge, 296,901 RGB + 296,901 IR images.
https://github.com/ZhaoJ9014/Anti-UAV

Used for training and testing the multispectral models. The raw videos are aligned and
synchronised by `scripts/data_preparation/anti_uav/sync_anti_uav_rgbt.py`; frames that
were not included in the synchronised pairs are used for unimodal pretraining.

## Unimodal pretraining - RGB (visible)

Data config: `configs/train/pretrain/rgb_cfg.yaml`.

| Dataset | Images | Source |
|---|---|---|
| Drone Detection Dataset | 54,000 | https://github.com/Maciullo/DroneDetectionDataset |
| DUT-Anti-UAV | 8,057 | https://github.com/wangdongdut/DUT-Anti-UAV (paper: https://arxiv.org/abs/2205.10851) |
| USC-MCL Drone Dataset | 706 | https://mcl.usc.edu/mcl-drone-dataset/ |
| UAV-Eagle | 947 | https://universe.roboflow.com/drone-detection-9oxnd/drone-detection-wypxm/dataset/2 |
| S-UAV-T (synthetic, Blender) | 52,500 | https://github.com/larics/synthetic-UAV |
| DroneDetectionPaper | 1,936 | https://github.com/chelicynly/A-Deep-Learning-Approach-to-Drone-Monitoring |
| RISL | 7,578 | https://universe.roboflow.com/drone-detection-9oxnd/drone-detection-wypxm |
| Halmstadt UAV Dataset (visible part) | 85,000 (RGB + IR) | https://zenodo.org/records/5500576 (DOI 10.5281/zenodo.5500575) |

An additional own RGB set (`Finetune_data_10_09`) is also part of the pretraining mix.

## Unimodal pretraining - IR (thermal)

Data config: `configs/train/pretrain/ir_cfg.yaml`.

| Dataset | Images | Source |
|---|---|---|
| Thermal Drones Dataset | 360 | https://universe.roboflow.com/korki14/drones-srdze/dataset/1 |
| Halmstadt UAV Dataset (infrared part) | 85,000 (RGB + IR) | https://zenodo.org/records/5500576 (DOI 10.5281/zenodo.5500575) |
