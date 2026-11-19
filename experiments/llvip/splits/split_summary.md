# LLVIP split

Validation scenes are taken from the official train set (seed 0, at least 12% of its frames, whole scenes only). The official test set is unchanged and is used only for the final evaluation. Built by `scripts/data_preparation/llvip/make_llvip_split.py`.

| Split | Folder | Scenes | Frames | Boxes |
|---|---|---|---:|---:|
| train | `train_split` | 01, 02, 03, 04, 05, 06, 07, 08, 09, 11, 13, 14, 15, 16, 17, 18, 25 | 10251 | 30092 |
| val | `val_split` | 10, 12 | 1774 | 4038 |
| test | `test` | 19, 20, 21, 22, 23, 24, 26 | 3463 | 8302 |
