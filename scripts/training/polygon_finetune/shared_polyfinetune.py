#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers specific to the polygon fine-tuning stage: copying an init
checkpoint before fine-tuning, and writing the fine-tune run_summary.md /
multi-experiment pipeline summary.

General-purpose helpers (CUDA check, ultralytics import check, image
counting, best-epoch metrics) come from scripts.training.shared.utils --
this module only holds what's specific to fine-tune experiments (they carry
extra fields like source checkpoint / dataset root that the from_scratch /
pretrain / transfer summary format doesn't have).
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def copy_init_weight(source_pt: Path, init_pt: Path, init_copy_dir: Path) -> None:
    """Copy a source checkpoint into the fine-tune init dir (once)."""
    init_copy_dir.mkdir(parents=True, exist_ok=True)

    if not source_pt.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_pt}")

    if not init_pt.exists():
        shutil.copy2(source_pt, init_pt)
        print("[INFO] Copied init weight:")
        print(f"  from: {source_pt}")
        print(f"  to:   {init_pt}")
    else:
        print(f"[INFO] Init copy already exists: {init_pt}")


def save_run_markdown(save_dir: Path, info: Dict[str, Any]) -> None:
    """Save a Markdown summary of a single polygon fine-tune run."""
    md_path = save_dir / "run_summary.md"

    lines = [
        f"# {info['run_name']}",
        "",
        "## Run status",
        f"- completed: `{info.get('completed', True)}`",
        f"- stop_reason: `{info.get('stop_reason', 'finished')}`",
        "",
        "## Fine-tune setup",
        f"- experiment_tag: `{info['tag']}`",
        f"- modality: `{info.get('modality', '')}`",
        f"- source_checkpoint: `{info['source_checkpoint']}`",
        f"- init_checkpoint_copy: `{info['init_checkpoint']}`",
        f"- data_yaml: `{info['data_yaml']}`",
        f"- dataset_root: `{info['dataset_root']}`",
        "",
        "## Best Model",
        f"- best_epoch: `{info['best_epoch']}`",
        f"- best_mAP50: `{info['best_mAP50']:.4f}`",
        f"- best_mAP50-95: `{info['best_mAP50_95']:.4f}`",
        f"- best_precision: `{info['best_precision']:.4f}`",
        f"- best_recall: `{info['best_recall']:.4f}`",
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
        f"- degrees: `{info['degrees']}`",
        f"- translate: `{info['translate']}`",
        f"- scale: `{info['scale']}`",
        f"- fliplr: `{info['fliplr']}`",
        f"- mosaic: `{info['mosaic']}`",
        f"- close_mosaic: `{info['close_mosaic']}`",
        "",
        "## Dataset size",
        f"- train_images: `{info['train_images']}`",
        f"- val_images: `{info['val_images']}`",
        "",
        "## Timing",
        f"- start_time: `{info['start_time']}`",
        f"- end_time: `{info['end_time']}`",
        f"- duration_hms: `{info['duration_hms']}`",
        "",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved run summary to: {md_path}")


def save_pipeline_summary(
    infos: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    include_modality: bool = False,
) -> None:
    """Save a Markdown table summarizing several fine-tune experiments."""
    header_cols = ["tag"]
    if include_modality:
        header_cols.append("modality")
    header_cols += ["best_epoch", "mAP50", "mAP50-95", "precision", "recall", "save_dir"]

    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(header_cols) + " |",
        "|" + "|".join("---:" if c not in ("tag", "modality", "save_dir") else "---" for c in header_cols) + "|",
    ]

    for info in infos:
        row = [info["tag"]]
        if include_modality:
            row.append(info.get("modality", ""))
        row += [
            str(info["best_epoch"]),
            f"{info['best_mAP50']:.4f}",
            f"{info['best_mAP50_95']:.4f}",
            f"{info['best_precision']:.4f}",
            f"{info['best_recall']:.4f}",
            f"`{info['save_dir']}`",
        ]
        lines.append("| " + " | ".join(row) + " |")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved pipeline summary: {out_path}")
