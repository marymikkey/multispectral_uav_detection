#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alignment and synchronization of RGB/IR frames from the Anti-UAV dataset.

The script estimates local temporal shifts between visible and infrared
streams using frame-difference values, warps IR frames into the RGB coordinate
system, selects consistent RGB/IR pairs using bounding-box consistency checks,
and saves paired images with shared YOLO annotations.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from scripts.paths import ANTIUAV_DATA_ROOT


# =========================================================
# CONFIGURATION
# =========================================================

# Paths
ROOT = ANTIUAV_DATA_ROOT / "ANTI-UAV_each_frame"
OUT_ROOT = ANTIUAV_DATA_ROOT / "ANTI-UAV_synced_lite"

# Which splits to process
SPLITS = ["train", "val", "test"]

CLASS_ID = 0

# ---------- Stage 1: frame-diff ----------
USE_BLUR = False
BLUR_KSIZE = 5
MASK_BBOX = True
MASK_EXPAND_PX = 6
SMOOTH_KERNEL = 7

# ---------- Local shift by windows ----------
SHIFT_WINDOW_SIZE = 100
SHIFT_WINDOW_STEP = 50
MAX_SHIFT_FOR_WINDOW_CORR = 25
MIN_OVERLAP_RATIO = 0.60
MIN_WINDOW_CORR = 0.18

# ---------- Stage 2: alignment ----------
UNION_EXPAND_RATIO = 1.5

SCALE = 2.8515
ROT_DEG = -0.33
OFF_X = 10
OFF_Y = -245

IOU_MIN = 0.05

SEARCH_RADIUS_LOCAL = 3

# ---------- Output options ----------
SAVE_SHIFT_INFO = False      # Save per-scene shift estimation info


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def scene_from_stem(stem: str) -> str:
    return stem.rsplit("_f", 1)[0] if "_f" in stem else stem


def frame_index_from_stem(stem: str) -> Optional[int]:
    if "_f" not in stem:
        return None
    try:
        return int(stem.rsplit("_f", 1)[1])
    except Exception:
        return None


def read_single_yolo_box(txt_path: Path, class_id: int) -> Tuple[bool, Optional[Tuple[float, float, float, float]]]:
    if not txt_path.exists():
        return False, None
    s = txt_path.read_text(encoding="utf-8").strip()
    if not s:
        return False, None
    for line in s.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            if cls != class_id:
                continue
            return True, (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except Exception:
            continue
    return False, None


def yolo_to_xyxy_px(box_n: Tuple[float, float, float, float], W: int, H: int) -> Tuple[float, float, float, float]:
    cx_n, cy_n, w_n, h_n = box_n
    cx = cx_n * W
    cy = cy_n * H
    w = w_n * W
    h = h_n * H
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


def clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0.0, min(float(W), x1))
    y1 = max(0.0, min(float(H), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def clamp_xyxy_int(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def box_area(box_xyxy: Tuple[float, float, float, float]) -> float:
    return max(0.0, box_xyxy[2] - box_xyxy[0]) * max(0.0, box_xyxy[3] - box_xyxy[1])


def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def union_xyxy(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def build_scene_maps(label_dir: Path) -> Dict[str, List[str]]:
    scene_to_stems = defaultdict(list)
    for p in label_dir.glob("*.txt"):
        scene = scene_from_stem(p.stem)
        scene_to_stems[scene].append(p.stem)
    return scene_to_stems


def build_idx_to_stem(stems: List[str]) -> Dict[int, str]:
    idx_to_stem = {}
    for stem in stems:
        fi = frame_index_from_stem(stem)
        if fi is not None:
            idx_to_stem[fi] = stem
    return idx_to_stem


# =========================================================
# IR -> RGB TRANSFORM (geometric alignment)
# =========================================================

def rotate_expand_affine(w: int, h: int, rot_deg: float) -> Tuple[np.ndarray, int, int]:
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    rot = np.vstack([M, [0, 0, 1]]) @ corners
    xs, ys = rot[0, :], rot[1, :]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))
    M_adj = M.copy()
    M_adj[0, 2] -= min_x
    M_adj[1, 2] -= min_y
    return M_adj, new_w, new_h


def build_ir_to_rgb_pipeline(ir_w: int, ir_h: int, scale: float, rot_deg: float):
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    sw = int(ir_w * scale)
    sh = int(ir_h * scale)
    M_rot, rw, rh = rotate_expand_affine(sw, sh, rot_deg)
    R = np.vstack([M_rot, [0, 0, 1]]).astype(np.float64)
    A = R @ S
    return A, rw, rh


def warp_ir_to_rgb_canvas(ir_img: np.ndarray, rgb_w: int, rgb_h: int,
                          scale: float, rot_deg: float, off_x: float, off_y: float) -> np.ndarray:
    """Warp IR image to RGB geometry, return single-channel (grayscale)"""
    ir_h, ir_w = ir_img.shape[:2]
    A, rw, rh = build_ir_to_rgb_pipeline(ir_w, ir_h, scale, rot_deg)
    A2 = A[:2, :]
    
    ir_warp = cv2.warpAffine(ir_img, A2, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Convert to grayscale if needed
    if ir_warp.ndim == 3:
        ir_warp = cv2.cvtColor(ir_warp, cv2.COLOR_BGR2GRAY)
    
    canvas = np.zeros((rgb_h, rgb_w), dtype=np.uint8)
    x1, y1 = int(round(off_x)), int(round(off_y))
    x2, y2 = x1 + rw, y1 + rh
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(rgb_w, x2), min(rgb_h, y2)
    
    if cx1 < cx2 and cy1 < cy2:
        sx1, sy1 = cx1 - x1, cy1 - y1
        sx2, sy2 = sx1 + (cx2 - cx1), sy1 + (cy2 - cy1)
        canvas[cy1:cy2, cx1:cx2] = ir_warp[sy1:sy2, sx1:sx2]
    
    return canvas


def transform_ir_box_to_rgb_xyxy(ir_box_n, ir_w, ir_h, rgb_w, rgb_h, scale, rot_deg, off_x, off_y):
    x1, y1, x2, y2 = yolo_to_xyxy_px(ir_box_n, ir_w, ir_h)
    corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]], dtype=np.float64).T
    A, _, _ = build_ir_to_rgb_pipeline(ir_w, ir_h, scale, rot_deg)
    warped = A @ corners
    xs = warped[0, :] + off_x
    ys = warped[1, :] + off_y
    return clamp_xyxy(xs.min(), ys.min(), xs.max(), ys.max(), rgb_w, rgb_h)


# =========================================================
# FRAME-DIFF SIGNAL
# =========================================================

def union_bbox(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def mask_bbox_inplace(gray, bbox_xyxy, expand=0):
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 -= expand
    y1 -= expand
    x2 += expand
    y2 += expand
    x1, y1, x2, y2 = clamp_xyxy_int(x1, y1, x2, y2, W, H)
    if x2 > x1 and y2 > y1:
        gray[y1:y2, x1:x2] = 0


def compute_diff_score(prev_bgr, curr_bgr, bbox_prev=None, bbox_curr=None):
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    
    if USE_BLUR and BLUR_KSIZE >= 3 and BLUR_KSIZE % 2 == 1:
        prev = cv2.GaussianBlur(prev, (BLUR_KSIZE, BLUR_KSIZE), 0)
        curr = cv2.GaussianBlur(curr, (BLUR_KSIZE, BLUR_KSIZE), 0)
    
    if MASK_BBOX:
        ub = union_bbox(bbox_prev, bbox_curr)
        if ub is not None:
            mask_bbox_inplace(prev, ub, expand=MASK_EXPAND_PX)
            mask_bbox_inplace(curr, ub, expand=MASK_EXPAND_PX)
    
    diff = cv2.absdiff(curr, prev)
    return float(diff.mean())


def compute_scene_fdiff_signal(idx_to_stem, img_dir, lbl_dir, class_id):
    frame_indices = sorted(idx_to_stem.keys())
    if len(frame_indices) < 2:
        return {}
    
    boxes = {}
    for t in frame_indices:
        present, box = read_single_yolo_box(lbl_dir / f"{idx_to_stem[t]}.txt", class_id)
        if present and box:
            boxes[t] = box
    
    scores = {}
    for t in frame_indices[1:]:
        if t-1 not in idx_to_stem:
            continue
        
        prev_img = cv2.imread(str(img_dir / f"{idx_to_stem[t-1]}.jpg"))
        curr_img = cv2.imread(str(img_dir / f"{idx_to_stem[t]}.jpg"))
        if prev_img is None or curr_img is None:
            continue
        
        h, w = curr_img.shape[:2]
        
        bb_prev = None
        bb_curr = None
        if t-1 in boxes:
            bb_prev = clamp_xyxy_int(*map(int, map(round, yolo_to_xyxy_px(boxes[t-1], w, h))), w, h)
        if t in boxes:
            bb_curr = clamp_xyxy_int(*map(int, map(round, yolo_to_xyxy_px(boxes[t], w, h))), w, h)
        
        scores[t] = compute_diff_score(prev_img, curr_img, bbox_prev=bb_prev, bbox_curr=bb_curr)
    
    return scores


def smooth_signal(y, k=SMOOTH_KERNEL):
    y = y.astype(np.float32)
    if len(y) < 5 or k < 3:
        return y.copy()
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(y, kernel, mode='same')


def znorm(x):
    x = x.astype(np.float32)
    if len(x) < 3:
        return None
    mu, std = np.mean(x), np.std(x)
    if std < 1e-6:
        return None
    return (x - mu) / std


def normalized_corr_for_lag(a, b, lag, min_overlap):
    if lag >= 0:
        a_seg = a[lag:]
        b_seg = b[:len(b) - lag]
    else:
        s = -lag
        a_seg = a[:len(a) - s]
        b_seg = b[s:]
    
    if len(a_seg) < min_overlap or len(b_seg) < min_overlap or len(a_seg) != len(b_seg):
        return None
    
    za, zb = znorm(a_seg), znorm(b_seg)
    if za is None or zb is None:
        return None
    
    return float(np.mean(za * zb))


def build_shift_curve_from_window_centers(frame_ids, centers, lags):
    out = {}
    if len(frame_ids) == 0:
        return out
    if len(centers) == 0:
        for t in frame_ids:
            out[int(t)] = 0.0
        return out
    
    centers_np = np.array(centers, dtype=np.float64)
    lags_np = np.array(lags, dtype=np.float64)
    interp_vals = np.interp(frame_ids.astype(np.float64), centers_np, lags_np, left=lags_np[0], right=lags_np[-1])
    
    for t, v in zip(frame_ids, interp_vals):
        out[int(t)] = float(v)
    return out


def estimate_scene_local_shift_from_fdiff(vis_scores_by_frame, ir_scores_by_frame):
    all_frames = sorted(set(vis_scores_by_frame.keys()) | set(ir_scores_by_frame.keys()))
    if not all_frames:
        return {"avg_shift_rgb_minus_ir": 0, "reason": "no_frames", "window_estimates": [], "shift_curve_by_frame": {}}
    
    x = np.array(all_frames, dtype=np.int32)
    y_vis = np.array([vis_scores_by_frame.get(t, np.nan) for t in x], dtype=np.float32)
    y_ir = np.array([ir_scores_by_frame.get(t, np.nan) for t in x], dtype=np.float32)
    
    finite_mask = np.isfinite(y_vis) & np.isfinite(y_ir)
    if finite_mask.sum() < 20:
        return {"avg_shift_rgb_minus_ir": 0, "reason": "too_few_finite_points", "window_estimates": [], "shift_curve_by_frame": {}}
    
    x_f = x[finite_mask]
    y_vis_f = smooth_signal(y_vis[finite_mask], SMOOTH_KERNEL)
    y_ir_f = smooth_signal(y_ir[finite_mask], SMOOTH_KERNEL)
    
    min_f, max_f = int(x_f.min()), int(x_f.max())
    min_overlap_len = max(20, int(round(SHIFT_WINDOW_SIZE * MIN_OVERLAP_RATIO)))
    
    window_estimates = []
    valid_centers, valid_lags = [], []
    
    for start in range(min_f, max_f + 1, SHIFT_WINDOW_STEP):
        end = start + SHIFT_WINDOW_SIZE - 1
        center = start + SHIFT_WINDOW_SIZE // 2
        
        mask_w = (x_f >= start) & (x_f <= end)
        if mask_w.sum() < min_overlap_len:
            continue
        
        win_vis, win_ir = y_vis_f[mask_w], y_ir_f[mask_w]
        
        best_lag, best_corr = None, None
        for lag in range(-MAX_SHIFT_FOR_WINDOW_CORR, MAX_SHIFT_FOR_WINDOW_CORR + 1):
            corr = normalized_corr_for_lag(win_vis, win_ir, lag, min_overlap_len)
            if corr is None:
                continue
            if best_corr is None or corr > best_corr:
                best_corr, best_lag = corr, lag
        
        is_valid = best_lag is not None and best_corr is not None and best_corr >= MIN_WINDOW_CORR
        
        window_estimates.append({
            "window_start": int(start), "window_end": int(end), "window_center": int(center),
            "num_points": int(mask_w.sum()), "lag_rgb_minus_ir": int(best_lag) if best_lag is not None else None,
            "corr": float(best_corr) if best_corr is not None else None, "is_valid": bool(is_valid)
        })
        
        if is_valid:
            valid_centers.append(int(center))
            valid_lags.append(float(best_lag))
    
    if not valid_centers:
        shift_curve = {int(t): 0.0 for t in x_f}
        return {"avg_shift_rgb_minus_ir": 0, "reason": "no_valid_windows", "window_estimates": window_estimates, "shift_curve_by_frame": shift_curve}
    
    shift_curve = build_shift_curve_from_window_centers(frame_ids=x_f, centers=valid_centers, lags=valid_lags)
    avg_shift = int(round(np.mean(valid_lags)))
    
    return {"avg_shift_rgb_minus_ir": int(avg_shift), "reason": "ok", "window_estimates": window_estimates, "shift_curve_by_frame": shift_curve}


def choose_best_candidate_local_shift(t_ir, local_shift_rgb_minus_ir, vis_map, vis_lbl_dir,
                                       ir_xyxy_warp, rgb_w, rgb_h, search_radius):
    center = int(round(t_ir + local_shift_rgb_minus_ir))
    radius = max(0, int(search_radius))
    j_lo, j_hi = center - radius, center + radius
    
    best = None
    found_any_index = False
    found_any_label = False

    fail_reasons = set()   

    for j in range(j_lo, j_hi + 1):
        if j not in vis_map:
            continue
        found_any_index = True

        rgb_present, rgb_box_n = read_single_yolo_box(vis_lbl_dir / f"{vis_map[j]}.txt", CLASS_ID)
        if not rgb_present or rgb_box_n is None:
            continue
        found_any_label = True

        rgb_xyxy = yolo_to_xyxy_px(rgb_box_n, rgb_w, rgb_h)
        iou = box_iou(rgb_xyxy, ir_xyxy_warp)

        union = union_xyxy(rgb_xyxy, ir_xyxy_warp)
        area_union = box_area(union)
        area_ref = max(box_area(rgb_xyxy), box_area(ir_xyxy_warp))

        # --- FILTERS ---
        if area_ref > 0 and area_union >= UNION_EXPAND_RATIO * area_ref:
            fail_reasons.add("union_too_large")
            continue

        if IOU_MIN > 0 and iou < IOU_MIN:
            fail_reasons.add("iou_below_min")
            continue

        # --- candidate ---
        score = 1.0 - iou
        cand = (score, j, iou, rgb_xyxy, union)

        if best is None or cand[0] < best[0]:
            best = cand


    # --- RESULT ---
    if best is not None:
        return best, "ok"

    if not found_any_index:
        return None, "rgb_index_missing"

    if not found_any_label:
        return None, "rgb_label_missing"

    if radius == 0:
        return None, "exact_local_shift_failed"

    return None, ",".join(sorted(fail_reasons)) if fail_reasons else "no_valid_candidate"


# =========================================================
# SYNCHRONIZATION MAIN
# =========================================================

def sync_split(root: Path, split: str, out_root: Path) -> Dict[str, Any]:
    print(f"\n{'='*60}\nProcessing split: {split}\n{'='*60}")
    
    vis_img_src = root / "visible" / "images" / split
    ir_img_src = root / "infrared" / "images" / split
    vis_lbl_src = root / "visible" / "labels" / split
    ir_lbl_src = root / "infrared" / "labels" / split
    
    out_vis_img = out_root / split / "visible" / "images"
    out_ir_img = out_root / split / "infrared" / "images"
    out_vis_lbl = out_root / split / "visible" / "labels"
    out_ir_lbl = out_root / split / "infrared" / "labels"

    ensure_dir(out_vis_img)
    ensure_dir(out_ir_img)
    ensure_dir(out_vis_lbl)
    ensure_dir(out_ir_lbl)
    
    vis_scenes = build_scene_maps(vis_lbl_src)
    ir_scenes = build_scene_maps(ir_lbl_src)
    common_scenes = sorted(set(vis_scenes.keys()) & set(ir_scenes.keys()))
    
    if not common_scenes:
        print(f"  No common scenes found in {split}")
        return {"split": split, "scenes": [], "skip_reasons": {}}
    
    sample_vis = next(iter(vis_img_src.glob("*.jpg")), None)
    sample_ir = next(iter(ir_img_src.glob("*.jpg")), None)
    if sample_vis is None or sample_ir is None:
        print(f"  No images found in {split}")
        return {"split": split, "scenes": [], "skip_reasons": {}}
    
    rgb = cv2.imread(str(sample_vis))
    ir = cv2.imread(str(sample_ir))
    rgb_h, rgb_w = rgb.shape[:2]
    ir_h, ir_w = ir.shape[:2]
    
    print(f"  RGB: {rgb_w}x{rgb_h}, IR: {ir_w}x{ir_h}")
    print(f"  Found {len(common_scenes)} scenes")
    
    total_vis_frames = 0
    total_ir_frames = 0
    total_synced_frames = 0
    scene_results = []
    global_skip_reasons = defaultdict(int)
    
    for scene in tqdm(common_scenes, desc=f"  Processing {split}"):
        vis_stems = sorted(vis_scenes[scene], key=lambda s: frame_index_from_stem(s) or 0)
        ir_stems = sorted(ir_scenes[scene], key=lambda s: frame_index_from_stem(s) or 0)
        
        vis_map = build_idx_to_stem(vis_stems)
        ir_map = build_idx_to_stem(ir_stems)
        
        vis_frame_count = len(vis_map)
        ir_frame_count = len(ir_map)
        total_vis_frames += vis_frame_count
        total_ir_frames += ir_frame_count
        
        vis_scores = compute_scene_fdiff_signal(vis_map, vis_img_src, vis_lbl_src, CLASS_ID)
        ir_scores = compute_scene_fdiff_signal(ir_map, ir_img_src, ir_lbl_src, CLASS_ID)
        
        shift_info = estimate_scene_local_shift_from_fdiff(vis_scores, ir_scores)
        avg_shift = shift_info["avg_shift_rgb_minus_ir"]
        shift_curve = shift_info.get("shift_curve_by_frame", {})
        
        if SAVE_SHIFT_INFO:
            scene_out_dir = out_root / "shift_info" / split / scene
            ensure_dir(scene_out_dir)
            (scene_out_dir / "shift_info.json").write_text(json.dumps(shift_info, indent=2), encoding="utf-8")
        
        synced_count = 0
        scene_skip_reasons = defaultdict(int)
        ir_indices = sorted(ir_map.keys())
        
        for t in ir_indices:
            ir_present, ir_box = read_single_yolo_box(ir_lbl_src / f"{ir_map[t]}.txt", CLASS_ID)
            if not ir_present or ir_box is None:
                continue
            
            if t not in shift_curve:
                scene_skip_reasons["no_local_shift_for_frame"] += 1
                global_skip_reasons["no_local_shift_for_frame"] += 1
                continue

            local_shift = int(round(shift_curve[t]))
                        
            ir_xyxy_warp = transform_ir_box_to_rgb_xyxy(ir_box, ir_w, ir_h, rgb_w, rgb_h, SCALE, ROT_DEG, OFF_X, OFF_Y)
            
            candidate, reason = choose_best_candidate_local_shift(
                t_ir=t, local_shift_rgb_minus_ir=local_shift, vis_map=vis_map, vis_lbl_dir=vis_lbl_src,
                ir_xyxy_warp=ir_xyxy_warp, rgb_w=rgb_w, rgb_h=rgb_h, search_radius=SEARCH_RADIUS_LOCAL
            )
            
            if candidate is None:
                if reason:
                    scene_skip_reasons[reason] += 1
                    global_skip_reasons[reason] += 1
                continue
            
            _, j_star, iou_best, rgb_xyxy_best, union_box = candidate
            union_box = clamp_xyxy(*union_box, rgb_w, rgb_h)
            
            # Convert union to YOLO
            cx = (union_box[0] + union_box[2]) / 2.0 / rgb_w
            cy = (union_box[1] + union_box[3]) / 2.0 / rgb_h
            w = (union_box[2] - union_box[0]) / rgb_w
            h = (union_box[3] - union_box[1]) / rgb_h
            
            if not (0 < cx < 1 and 0 < cy < 1 and 0 < w <= 1 and 0 < h <= 1):
                scene_skip_reasons["invalid_yolo"] += 1
                global_skip_reasons["invalid_yolo"] += 1
                continue
            
            # Read and warp IR image
            ir_img = cv2.imread(str(ir_img_src / f"{ir_map[t]}.jpg"), cv2.IMREAD_GRAYSCALE)
            if ir_img is None:
                scene_skip_reasons["ir_img_read_fail"] += 1
                global_skip_reasons["ir_img_read_fail"] += 1
                continue
            ir_warped = warp_ir_to_rgb_canvas(ir_img, rgb_w, rgb_h, SCALE, ROT_DEG, OFF_X, OFF_Y)
            
            # Read RGB image
            rgb_img = cv2.imread(str(vis_img_src / f"{vis_map[j_star]}.jpg"))
            if rgb_img is None:
                scene_skip_reasons["rgb_img_read_fail"] += 1
                global_skip_reasons["rgb_img_read_fail"] += 1
                continue
            
            stem = f"{scene}_ir{t:06d}_rgb{j_star:06d}"
            cv2.imwrite(str(out_vis_img / f"{stem}.jpg"), rgb_img)
            (out_vis_lbl / f"{stem}.txt").write_text(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")

            cv2.imwrite(str(out_ir_img / f"{stem}.jpg"), ir_warped)
            (out_ir_lbl / f"{stem}.txt").write_text(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
            synced_count += 1
        
        total_synced_frames += synced_count
        scene_results.append({
            "scene": scene, "original_rgb_frames": vis_frame_count, "original_ir_frames": ir_frame_count,
            "synced_frames": synced_count, "estimated_avg_shift": avg_shift, "shift_reason": shift_info["reason"],
            "skip_reasons": dict(scene_skip_reasons)
        })
        print(f"    {scene}: {synced_count}/{min(vis_frame_count, ir_frame_count)} synced frames")
    
    return {
        "split": split, "rgb_size": [rgb_w, rgb_h], "ir_size": [ir_w, ir_h],
        "total_original_rgb_frames": total_vis_frames, "total_original_ir_frames": total_ir_frames,
        "total_synced_frames": total_synced_frames, "scenes": scene_results, "skip_reasons": dict(global_skip_reasons)
    }


def main():
    print(f"\n{'='*60}")
    print(f"Anti-UAV Dataset Synchronization")
    print(f"{'='*60}")
    print(f"Source: {ROOT}")
    print(f"Output: {OUT_ROOT}")
    print(f"Splits: {SPLITS}")
    print(f"Search radius: {SEARCH_RADIUS_LOCAL}")
    print(f"{'='*60}\n")

    for split in SPLITS:
        ensure_dir(OUT_ROOT / split / "visible" / "images")
        ensure_dir(OUT_ROOT / split / "visible" / "labels")
        ensure_dir(OUT_ROOT / split / "infrared" / "images")
        ensure_dir(OUT_ROOT / split / "infrared" / "labels")

    if SAVE_SHIFT_INFO:
        ensure_dir(OUT_ROOT / "shift_info")

    all_results = []
    total_orig_rgb = 0
    total_orig_ir = 0
    total_synced = 0

    all_skip_reasons = defaultdict(int)

    for split in SPLITS:
        result = sync_split(ROOT, split, OUT_ROOT)
        all_results.append(result)

        total_orig_rgb += result.get("total_original_rgb_frames", 0)
        total_orig_ir += result.get("total_original_ir_frames", 0)
        total_synced += result.get("total_synced_frames", 0)

        for reason, count in result.get("skip_reasons", {}).items():
            all_skip_reasons[reason] += count

    rgb_ret = total_synced / total_orig_rgb if total_orig_rgb > 0 else 0
    ir_ret = total_synced / total_orig_ir if total_orig_ir > 0 else 0
    combined_ret = (
        (2.0 * total_synced) / (total_orig_rgb + total_orig_ir)
        if (total_orig_rgb + total_orig_ir) > 0 else 0
    )

    summary = {
        "source": str(ROOT),
        "output": str(OUT_ROOT),
        "splits_processed": SPLITS,
        "search_radius_local": SEARCH_RADIUS_LOCAL,

        "total_original_rgb_frames": total_orig_rgb,
        "total_original_ir_frames": total_orig_ir,
        "total_synced_frames": total_synced,

        "rgb_retention_rate": rgb_ret,
        "ir_retention_rate": ir_ret,
        "combined_retention_rate": combined_ret,

        "skip_reasons": dict(all_skip_reasons),
        "results_by_split": all_results
    }

    (OUT_ROOT / "sync_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print("SYNCHRONIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {OUT_ROOT}")

    print(f"\nOriginal RGB frames total: {total_orig_rgb:,}")
    print(f"Original IR frames total:  {total_orig_ir:,}")
    print(f"Synced frames total:       {total_synced:,}")

    print(f"\nRetention rates:")
    print(f"  RGB:      {rgb_ret*100:.2f}%")
    print(f"  IR:       {ir_ret*100:.2f}%")
    print(f"  Combined: {combined_ret*100:.2f}%")

    if all_skip_reasons:
        print(f"\nTop skip reasons:")
        for k, v in sorted(all_skip_reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"  {k}: {v}")

    print(f"\nSummary saved to: {OUT_ROOT / 'sync_summary.json'}")

if __name__ == "__main__":
    main()