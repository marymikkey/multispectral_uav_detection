#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate run_summary.md for training experiments.
"""

from pathlib import Path
from typing import Dict, Any


def save_run_markdown(save_dir: Path, info: Dict[str, Any]) -> None:
    """Save a Markdown summary of the training run."""
    md_path = save_dir / "run_summary.md"
    lines = [
        f"# {info['run_name']}",
        "",
        "## Run status",
        f"- resume_mode: `{info.get('resume_mode', False)}`",
        f"- completed: `{info['completed']}`",
        f"- stop_reason: `{info['stop_reason']}`",
        "",
        "## Best Model",
        f"- best_epoch: `{info['best_epoch']}`",
        f"- best_mAP50: `{info['best_mAP50']:.4f}`",
        f"- best_mAP50-95: `{info['best_mAP50_95']:.4f}`",
        f"- best_precision: `{info['best_precision']:.4f}`",
        f"- best_recall: `{info['best_recall']:.4f}`",
        "",
        "## Model",
        f"- model_cfg: `{info.get('model_cfg', '')}`",
        f"- init_checkpoint: `{info.get('init_checkpoint', '')}`",
        f"- data_yaml: `{info['data_yaml']}`",
        f"- actual_save_dir: `{info.get('actual_save_dir', '')}`",
        "",
        "## Training params",
        f"- imgsz: `{info['imgsz']}`",
        f"- epochs: `{info['epochs']}`",
        f"- batch: `{info['batch']}`",
        f"- workers: `{info['workers']}`",
        f"- device: `{info['device']}`",
        f"- optimizer: `{info['optimizer']}`",
        f"- lr0: `{info['lr0']}`",
        f"- lrf: `{info['lrf']}`",
        f"- cos_lr: `{info['cos_lr']}`",
        f"- patience: `{info['patience']}`",
        f"- single_cls: `{info['single_cls']}`",
        f"- use_simotm: `{info.get('use_simotm', '')}`",
        f"- channels: `{info.get('channels', '')}`",
        f"- pairs_rgb_ir: `{info.get('pairs_rgb_ir', '')}`",
        f"- freeze: `{info.get('freeze', 'none')}`",
        "",
        "## Augmentations",
        f"- hsv_h: `{info['hsv_h']}`",
        f"- hsv_s: `{info['hsv_s']}`",
        f"- hsv_v: `{info['hsv_v']}`",
        f"- bgr: `{info['bgr']}`",
        f"- degrees: `{info['degrees']}`",
        f"- translate: `{info['translate']}`",
        f"- scale: `{info['scale']}`",
        f"- shear: `{info['shear']}`",
        f"- perspective: `{info['perspective']}`",
        f"- flipud: `{info['flipud']}`",
        f"- fliplr: `{info['fliplr']}`",
        f"- mosaic: `{info['mosaic']}`",
        f"- close_mosaic: `{info['close_mosaic']}`",
        f"- mixup: `{info['mixup']}`",
        "",
        "## Dataset size",
        f"- train_images: `{info['train_images']}`",
        f"- val_images: `{info['val_images']}`",
        "",
        "## Timing",
        f"- start_time: `{info['start_time']}`",
        f"- end_time: `{info['end_time']}`",
        f"- duration_sec: `{info['duration_sec']:.2f}`",
        f"- duration_hms: `{info['duration_hms']}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved run summary to: {md_path}")