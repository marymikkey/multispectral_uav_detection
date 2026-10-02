#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize RGB/IR synchronization for a selected Anti-UAV scene.
The script computes frame-difference values for visible and infrared streams,
estimates local temporal shifts, builds an interpolated shift curve, performs
final RGB/IR pair matching, and saves diagnostic plots for synchronization
analysis.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.paths import ANTIUAV_DATA_ROOT, DIPLOMA_ROOT

# =========================================================
# CONFIG
# =========================================================

ROOT = ANTIUAV_DATA_ROOT / "ANTI-UAV_each_frame"
SPLIT = "test"

OUT_DIR = DIPLOMA_ROOT / "anti_uav" / "results" / "test_lite_alignment" / "single_scene_plots"

CLASS_ID = 0

# ---------- Typography ----------
LEGEND_FONT_SIZE = 11
AXIS_LABEL_FONT_SIZE = 12
AXIS_TICK_FONT_SIZE_X = 10
AXIS_TICK_FONT_SIZE_Y = 9
TITLE_FONT_SIZE = 12
ANNOT_TEXT_FONT_SIZE = 10
LEGEND_LINEWIDTH = 2.0

# Target scene 
TARGET_SCENE = "20190925_111757_1_6"

# ---------- Stage 1: frame-difference ----------
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

# ---------- Visualization ----------
WINDOW_BORDER_COLOR = "red"
WINDOW_BORDER_ALPHA = 0.65
WINDOW_BORDER_LINEWIDTH = 0.8

SHIFT_CURVE_LINEWIDTH = 1.5
SHIFT_POINTS_SIZE = 18

FIGURE_WIDTH = 20
FIGURE_HEIGHT = 4

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


def read_single_yolo_box(
    txt_path: Path,
    class_id: int
) -> Tuple[bool, Optional[Tuple[float, float, float, float]]]:
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


def yolo_to_xyxy_px(
    box_n: Tuple[float, float, float, float],
    W: int,
    H: int
) -> Tuple[float, float, float, float]:
    cx_n, cy_n, w_n, h_n = box_n
    cx = cx_n * W
    cy = cy_n * H
    w = w_n * W
    h = h_n * H
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def clamp_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    W: int,
    H: int
) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(W), x1))
    y1 = max(0.0, min(float(H), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def clamp_xyxy_int(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    W: int,
    H: int
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def box_center_x(box_xyxy: Tuple[float, float, float, float]) -> float:
    return 0.5 * (box_xyxy[0] + box_xyxy[2])


def box_area(box_xyxy: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    a_area = box_area(a)
    b_area = box_area(b)
    union = a_area + b_area - inter

    return inter / union if union > 0 else 0.0


def union_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))


def build_scene_maps(label_dir: Path) -> Dict[str, List[str]]:
    scene_to_stems: Dict[str, List[str]] = {}
    for p in sorted(label_dir.glob("*.txt")):
        scene = scene_from_stem(p.stem)
        scene_to_stems.setdefault(scene, []).append(p.stem)
    return scene_to_stems


def build_idx_to_stem(stems: List[str]) -> Dict[int, str]:
    idx_to_stem = {}
    for stem in stems:
        fi = frame_index_from_stem(stem)
        if fi is not None:
            idx_to_stem[fi] = stem
    return idx_to_stem


# =========================================================
# IR -> RGB TRANSFORM
# =========================================================

def rotate_expand_affine(w: int, h: int, rot_deg: float) -> Tuple[np.ndarray, int, int]:
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)

    corners = np.array(
        [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]],
        dtype=np.float64
    ).T

    rot = np.vstack([M, [0, 0, 1]]) @ corners
    xs = rot[0, :]
    ys = rot[1, :]

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


def transform_ir_box_to_rgb_xyxy(
    ir_box_n: Tuple[float, float, float, float],
    ir_w: int,
    ir_h: int,
    rgb_w: int,
    rgb_h: int,
    scale: float,
    rot_deg: float,
    off_x: float,
    off_y: float
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = yolo_to_xyxy_px(ir_box_n, ir_w, ir_h)

    corners = np.array(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=np.float64
    ).T

    A, _, _ = build_ir_to_rgb_pipeline(ir_w, ir_h, scale, rot_deg)
    warped = A @ corners

    xs = warped[0, :] + off_x
    ys = warped[1, :] + off_y

    return clamp_xyxy(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), rgb_w, rgb_h)


# =========================================================
# FRAME-DIFFERENCE VALUES
# =========================================================

def union_bbox(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]]
) -> Optional[Tuple[int, int, int, int]]:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def mask_bbox_inplace(
    gray: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    expand: int = 0
) -> None:
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy

    x1 -= expand
    y1 -= expand
    x2 += expand
    y2 += expand

    x1, y1, x2, y2 = clamp_xyxy_int(x1, y1, x2, y2, W, H)

    if x2 > x1 and y2 > y1:
        gray[y1:y2, x1:x2] = 0


def compute_diff_score(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    bbox_prev: Optional[Tuple[int, int, int, int]] = None,
    bbox_curr: Optional[Tuple[int, int, int, int]] = None
) -> float:
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    if USE_BLUR and BLUR_KSIZE >= 3 and (BLUR_KSIZE % 2 == 1):
        prev = cv2.GaussianBlur(prev, (BLUR_KSIZE, BLUR_KSIZE), 0)
        curr = cv2.GaussianBlur(curr, (BLUR_KSIZE, BLUR_KSIZE), 0)

    if MASK_BBOX:
        ub = union_bbox(bbox_prev, bbox_curr)
        if ub is not None:
            mask_bbox_inplace(prev, ub, expand=MASK_EXPAND_PX)
            mask_bbox_inplace(curr, ub, expand=MASK_EXPAND_PX)

    diff = cv2.absdiff(curr, prev)
    return float(diff.mean())


def compute_scene_fdiff_signal(
    idx_to_stem: Dict[int, str],
    img_dir: Path,
    lbl_dir: Path,
    class_id: int
) -> Dict[int, float]:
    frame_indices = sorted(idx_to_stem.keys())
    scores_by_frame: Dict[int, float] = {t: float("nan") for t in frame_indices}
    boxes_n: Dict[int, Tuple[float, float, float, float]] = {}

    for t in frame_indices:
        stem = idx_to_stem[t]
        present, box_n = read_single_yolo_box(lbl_dir / f"{stem}.txt", class_id)
        if present and box_n is not None:
            boxes_n[t] = box_n

    for t in frame_indices:
        if (t - 1) not in idx_to_stem:
            continue

        stem_prev = idx_to_stem[t - 1]
        stem_curr = idx_to_stem[t]

        img_prev = cv2.imread(str(img_dir / f"{stem_prev}.jpg"), cv2.IMREAD_COLOR)
        img_curr = cv2.imread(str(img_dir / f"{stem_curr}.jpg"), cv2.IMREAD_COLOR)

        if img_prev is None or img_curr is None:
            continue

        H, W = img_curr.shape[:2]

        bb_prev = None
        bb_curr = None

        if (t - 1) in boxes_n:
            bb_prev = clamp_xyxy_int(
                *map(int, map(round, yolo_to_xyxy_px(boxes_n[t - 1], W, H))),
                W, H
            )

        if t in boxes_n:
            bb_curr = clamp_xyxy_int(
                *map(int, map(round, yolo_to_xyxy_px(boxes_n[t], W, H))),
                W, H
            )

        scores_by_frame[t] = compute_diff_score(
            img_prev,
            img_curr,
            bbox_prev=bb_prev,
            bbox_curr=bb_curr
        )

    return scores_by_frame


def smooth_signal(y: np.ndarray, k: int = SMOOTH_KERNEL) -> np.ndarray:
    y = y.astype(np.float32)
    if y.size < 5 or k < 3:
        return y.copy()
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(y, kernel, mode="same")


# =========================================================
# LOCAL WINDOW SHIFTS + INTERPOLATED SHIFT CURVE
# =========================================================

def znorm(x: np.ndarray) -> Optional[np.ndarray]:
    x = x.astype(np.float32)
    if x.size < 3:
        return None
    mu = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-6:
        return None
    return (x - mu) / std


def normalized_corr_for_lag(
    a: np.ndarray,
    b: np.ndarray,
    lag_rgb_minus_ir: int,
    min_overlap_len: int
) -> Optional[float]:
    if lag_rgb_minus_ir >= 0:
        a_seg = a[lag_rgb_minus_ir:]
        b_seg = b[:len(b) - lag_rgb_minus_ir]
    else:
        s = -lag_rgb_minus_ir
        a_seg = a[:len(a) - s]
        b_seg = b[s:]

    if len(a_seg) < min_overlap_len or len(b_seg) < min_overlap_len:
        return None
    if len(a_seg) != len(b_seg):
        return None

    za = znorm(a_seg)
    zb = znorm(b_seg)
    if za is None or zb is None:
        return None

    return float(np.mean(za * zb))


def build_shift_curve_from_window_centers(
    frame_ids: np.ndarray,
    centers: List[int],
    lags: List[float]
) -> Dict[int, float]:
    assert len(centers) == len(lags)
    out: Dict[int, float] = {}

    if len(frame_ids) == 0:
        return out

    if len(centers) == 0:
        for t in frame_ids:
            out[int(t)] = 0.0
        return out

    centers_np = np.array(centers, dtype=np.float64)
    lags_np = np.array(lags, dtype=np.float64)

    interp_vals = np.interp(
        frame_ids.astype(np.float64),
        centers_np,
        lags_np,
        left=lags_np[0],
        right=lags_np[-1]
    )

    for t, v in zip(frame_ids, interp_vals):
        out[int(t)] = float(v)

    return out


def estimate_scene_local_shift_from_fdiff(
    vis_scores_by_frame: Dict[int, float],
    ir_scores_by_frame: Dict[int, float]
) -> Dict[str, Any]:
    all_frames = sorted(set(vis_scores_by_frame.keys()) | set(ir_scores_by_frame.keys()))
    if not all_frames:
        return {
            "avg_shift_rgb_minus_ir": 0,
            "reason": "no_frames",
            "window_estimates": [],
            "shift_curve_by_frame": {},
        }

    x = np.array(all_frames, dtype=np.int32)
    y_vis = np.array([vis_scores_by_frame.get(int(t), np.nan) for t in x], dtype=np.float32)
    y_ir = np.array([ir_scores_by_frame.get(int(t), np.nan) for t in x], dtype=np.float32)

    finite_mask = np.isfinite(y_vis) & np.isfinite(y_ir)
    if finite_mask.sum() < 20:
        return {
            "avg_shift_rgb_minus_ir": 0,
            "reason": "too_few_finite_points",
            "window_estimates": [],
            "shift_curve_by_frame": {},
        }

    x_f = x[finite_mask]
    y_vis_f = smooth_signal(y_vis[finite_mask], SMOOTH_KERNEL)
    y_ir_f = smooth_signal(y_ir[finite_mask], SMOOTH_KERNEL)

    min_f = int(x_f.min())
    max_f = int(x_f.max())

    min_overlap_len = max(20, int(round(SHIFT_WINDOW_SIZE * MIN_OVERLAP_RATIO)))

    window_estimates: List[Dict[str, Any]] = []
    valid_centers: List[int] = []
    valid_lags: List[float] = []

    for start in range(min_f, max_f + 1, SHIFT_WINDOW_STEP):
        end = start + SHIFT_WINDOW_SIZE - 1
        center = start + SHIFT_WINDOW_SIZE // 2

        mask_w = (x_f >= start) & (x_f <= end)
        n_pts = int(mask_w.sum())

        best_lag = None
        best_corr = None

        if n_pts >= min_overlap_len:
            win_vis = y_vis_f[mask_w]
            win_ir = y_ir_f[mask_w]

            for lag in range(-MAX_SHIFT_FOR_WINDOW_CORR, MAX_SHIFT_FOR_WINDOW_CORR + 1):
                corr = normalized_corr_for_lag(
                    a=win_vis,
                    b=win_ir,
                    lag_rgb_minus_ir=lag,
                    min_overlap_len=min_overlap_len
                )
                if corr is None:
                    continue

                if best_corr is None or corr > best_corr:
                    best_corr = corr
                    best_lag = lag

        is_valid = (
            best_lag is not None and
            best_corr is not None and
            best_corr >= MIN_WINDOW_CORR
        )

        est = {
            "window_start": int(start),
            "window_end": int(end),
            "window_center": int(center),
            "num_points": int(n_pts),
            "lag_rgb_minus_ir": int(best_lag) if best_lag is not None else None,
            "corr": float(best_corr) if best_corr is not None else None,
            "is_valid": bool(is_valid),
        }
        window_estimates.append(est)

        if is_valid:
            valid_centers.append(int(center))
            valid_lags.append(float(best_lag))

    if not valid_lags:
        shift_curve = {int(t): 0.0 for t in x_f}
        return {
            "avg_shift_rgb_minus_ir": 0,
            "reason": "no_valid_windows",
            "window_estimates": window_estimates,
            "shift_curve_by_frame": shift_curve,
        }

    shift_curve = build_shift_curve_from_window_centers(
        frame_ids=x_f,
        centers=valid_centers,
        lags=valid_lags
    )

    avg_shift = int(round(np.mean(valid_lags)))

    return {
        "avg_shift_rgb_minus_ir": int(avg_shift),
        "reason": "ok",
        "window_estimates": window_estimates,
        "shift_curve_by_frame": shift_curve,
    }


# =========================================================
# FINAL MATCHING
# =========================================================

def choose_best_candidate_local_shift(
    t_ir: int,
    local_shift_rgb_minus_ir: int,
    vis_map: Dict[int, str],
    vis_lbl_dir: Path,
    ir_xyxy_warp: Tuple[float, float, float, float],
    rgb_w: int,
    rgb_h: int,
    search_radius: int,
) -> Tuple[Optional[Tuple], str]:
    center = int(round(t_ir + local_shift_rgb_minus_ir))
    radius = max(0, int(search_radius))

    j_lo = center - radius
    j_hi = center + radius

    best = None
    found_any_index = False
    found_any_label = False

    for j in range(j_lo, j_hi + 1):
        if j not in vis_map:
            continue
        found_any_index = True

        rgb_stem = vis_map[j]
        rgb_txt = vis_lbl_dir / f"{rgb_stem}.txt"

        rgb_present, rgb_box_n = read_single_yolo_box(rgb_txt, CLASS_ID)
        if not rgb_present or rgb_box_n is None:
            continue
        found_any_label = True

        rgb_xyxy = yolo_to_xyxy_px(rgb_box_n, rgb_w, rgb_h)
        iou = box_iou(rgb_xyxy, ir_xyxy_warp)

        union_box = union_xyxy(rgb_xyxy, ir_xyxy_warp)
        area_union = box_area(union_box)
        area_ref = max(box_area(rgb_xyxy), box_area(ir_xyxy_warp))

        if area_ref > 0 and area_union > UNION_EXPAND_RATIO * area_ref:
            continue

        if IOU_MIN > 0 and iou < IOU_MIN:
            continue

        score = 1.0 - iou
        cand = (score, j, iou, union_box)

        if best is None or cand[0] < best[0]:
            best = cand

    if best is not None:
        return best, "ok"

    if not found_any_index:
        return None, "rgb_index_missing"
    if not found_any_label:
        return None, "rgb_label_missing"

    if radius == 0:
        return None, "exact_local_shift_failed"
    return None, "no_valid_candidate_in_local_window"


def build_final_match_mapping(
    ir_map: Dict[int, str],
    vis_map: Dict[int, str],
    ir_lbl_dir: Path,
    vis_lbl_dir: Path,
    shift_curve_by_frame: Dict[int, float],
    ir_w: int,
    ir_h: int,
    rgb_w: int,
    rgb_h: int,
) -> Dict[int, int]:
    final_map: Dict[int, int] = {}

    for t in sorted(ir_map.keys()):
        if int(t) not in shift_curve_by_frame:
            continue

        ir_stem = ir_map[t]
        ir_txt = ir_lbl_dir / f"{ir_stem}.txt"

        ir_present, ir_box_n = read_single_yolo_box(ir_txt, CLASS_ID)
        if not ir_present or ir_box_n is None:
            continue

        ir_xyxy_warp = transform_ir_box_to_rgb_xyxy(
            ir_box_n,
            ir_w,
            ir_h,
            rgb_w,
            rgb_h,
            SCALE,
            ROT_DEG,
            OFF_X,
            OFF_Y
        )

        local_shift = int(round(shift_curve_by_frame[int(t)]))

        candidate, _ = choose_best_candidate_local_shift(
            t_ir=t,
            local_shift_rgb_minus_ir=local_shift,
            vis_map=vis_map,
            vis_lbl_dir=vis_lbl_dir,
            ir_xyxy_warp=ir_xyxy_warp,
            rgb_w=rgb_w,
            rgb_h=rgb_h,
            search_radius=SEARCH_RADIUS_LOCAL,
        )

        if candidate is None:
            continue

        _, j_star, _, _ = candidate
        final_map[int(t)] = int(j_star)

    return final_map


# =========================================================
# VISUALIZATION
# =========================================================

def save_fdiff_plot(
    scene: str,
    vis_scores: Dict[int, float],
    ir_scores: Dict[int, float],
    shift_info: Dict[str, Any],
    out_path: Path
):
    all_frames = sorted(set(vis_scores.keys()) | set(ir_scores.keys()))
    if not all_frames:
        return

    x = np.array(all_frames, dtype=np.int32)
    y_vis = np.array([vis_scores.get(int(t), np.nan) for t in x], dtype=np.float32)
    y_ir = np.array([ir_scores.get(int(t), np.nan) for t in x], dtype=np.float32)

    finite_mask = np.isfinite(y_vis) & np.isfinite(y_ir)
    if finite_mask.sum() >= 3:
        xs = x[finite_mask]
        yvs = smooth_signal(y_vis[finite_mask], SMOOTH_KERNEL)
        yis = smooth_signal(y_ir[finite_mask], SMOOTH_KERNEL)
    else:
        xs = x
        yvs = y_vis
        yis = y_ir

    shift_curve_by_frame = shift_info.get("shift_curve_by_frame", {})
    shift_vals = np.array([shift_curve_by_frame.get(int(t), np.nan) for t in xs], dtype=np.float32)
    window_estimates = shift_info.get("window_estimates", [])
    valid_windows = [w for w in window_estimates if w.get("is_valid", False)]

    fig, ax1 = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    ax1.plot(xs, yvs, linewidth=1.0, label="RGB frame-diff")
    ax1.plot(xs, yis, linewidth=1.0, label="IR frame-diff")

    for w in valid_windows:
        ax1.axvline(
            w["window_start"],
            color=WINDOW_BORDER_COLOR,
            alpha=WINDOW_BORDER_ALPHA,
            linewidth=WINDOW_BORDER_LINEWIDTH
        )
        ax1.axvline(
            w["window_end"],
            color=WINDOW_BORDER_COLOR,
            alpha=WINDOW_BORDER_ALPHA,
            linewidth=WINDOW_BORDER_LINEWIDTH
        )

    ax1.set_xlabel("Frame index", fontsize=AXIS_LABEL_FONT_SIZE)
    ax1.set_ylabel("Frame-diff score", fontsize=AXIS_LABEL_FONT_SIZE)
    ax1.set_title(scene, fontsize=TITLE_FONT_SIZE)

    xmin = int(xs.min())
    xmax = int(xs.max())
    major_x = 50
    minor_x = 10
    ax1.set_xticks(np.arange((xmin // major_x) * major_x, xmax + major_x, major_x))
    ax1.set_xticks(np.arange((xmin // minor_x) * minor_x, xmax + minor_x, minor_x), minor=True)

    finite_y = np.concatenate([yvs[np.isfinite(yvs)], yis[np.isfinite(yis)]])
    ymax = float(np.max(finite_y)) if finite_y.size > 0 else 1.0
    ymax = max(1.0, float(np.ceil(ymax)))
    ax1.set_yticks(np.arange(0.0, ymax + 1.0, 1.0))
    ax1.set_yticks(np.arange(0.0, ymax + 0.5, 0.5), minor=True)

    ax1.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.8)
    ax1.grid(True, which="minor", linestyle="--", linewidth=0.35, alpha=0.45)

    ax2 = ax1.twinx()
    ax2.plot(
        xs,
        shift_vals,
        linewidth=SHIFT_CURVE_LINEWIDTH,
        label="Interpolated local shift"
    )

    centers = [w["window_center"] for w in valid_windows]
    lags = [w["lag_rgb_minus_ir"] for w in valid_windows]
    if centers:
        ax2.scatter(
            centers,
            lags,
            s=SHIFT_POINTS_SIZE,
            c="black",
            marker="o",
            label="Valid window centers"
        )

    ax2.set_ylabel("Shift (RGB - IR)", fontsize=AXIS_LABEL_FONT_SIZE)

    avg_shift = int(shift_info["avg_shift_rgb_minus_ir"])
    n_windows_total = len(window_estimates)
    n_windows_valid = len(valid_windows)

    text = (
        f"Average shift: {avg_shift}\n"
        f"Windows total: {n_windows_total}\n"
        f"Windows valid: {n_windows_valid}\n"
        f"Window size: {SHIFT_WINDOW_SIZE}, step: {SHIFT_WINDOW_STEP}, lag search: ±{MAX_SHIFT_FOR_WINDOW_CORR}"
    )
    ax1.text(
        0.995, 0.98,
        text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOT_TEXT_FONT_SIZE,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.95
    )
    for legline in legend.get_lines():
        legline.set_linewidth(LEGEND_LINEWIDTH)

    ax1.tick_params(axis="x", labelrotation=90, labelsize=AXIS_TICK_FONT_SIZE_X)
    ax1.tick_params(axis="y", labelsize=AXIS_TICK_FONT_SIZE_Y)
    ax2.tick_params(axis="y", labelsize=AXIS_TICK_FONT_SIZE_Y)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_shift_only_plot(
    scene: str,
    vis_scores: Dict[int, float],
    ir_scores: Dict[int, float],
    shift_info: Dict[str, Any],
    out_path: Path,
    mode: str,
    final_match_map: Optional[Dict[int, int]] = None,
):
    all_frames = sorted(set(vis_scores.keys()) | set(ir_scores.keys()))
    if not all_frames:
        return

    x = np.array(all_frames, dtype=np.int32)
    y_vis = np.array([vis_scores.get(int(t), np.nan) for t in x], dtype=np.float32)
    y_ir = np.array([ir_scores.get(int(t), np.nan) for t in x], dtype=np.float32)

    finite_mask = np.isfinite(y_vis) & np.isfinite(y_ir)
    if finite_mask.sum() < 3:
        return

    xs = x[finite_mask]
    yvs = smooth_signal(y_vis[finite_mask], SMOOTH_KERNEL)
    yis = smooth_signal(y_ir[finite_mask], SMOOTH_KERNEL)

    window_estimates = shift_info.get("window_estimates", [])
    valid_windows = [w for w in window_estimates if w.get("is_valid", False)]
    invalid_windows = [w for w in window_estimates if not w.get("is_valid", False)]

    shift_curve_by_frame = shift_info.get("shift_curve_by_frame", {})
    shift_vals = np.array([shift_curve_by_frame.get(int(t), np.nan) for t in xs], dtype=np.float32)

    fig, ax1 = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax2 = ax1.twinx()

    for w in valid_windows:
        ax1.axvline(w["window_start"], color="red", alpha=0.65, linewidth=0.9)
        ax1.axvline(w["window_end"], color="red", alpha=0.65, linewidth=0.9)

    for w in invalid_windows:
        ax1.axvline(w["window_start"], color="green", alpha=0.55, linewidth=0.8, linestyle="--")
        ax1.axvline(w["window_end"], color="green", alpha=0.55, linewidth=0.8, linestyle="--")

    title_suffix = ""

    if mode == "window":
        first_rgb_shifted = True
        first_ir_shifted = True

        for w in valid_windows:
            start = int(w["window_start"])
            end = int(w["window_end"])
            lag = int(w["lag_rgb_minus_ir"])

            mask_w = (xs >= start) & (xs <= end)
            if mask_w.sum() < 3:
                continue

            xw = xs[mask_w]
            rgb_wv = yvs[mask_w]
            ir_wv = yis[mask_w]

            if lag >= 0:
                rgb_seg = rgb_wv[lag:]
                ir_seg = ir_wv[:len(ir_wv) - lag]
                x_seg = xw[:len(ir_wv) - lag]
            else:
                s = -lag
                rgb_seg = rgb_wv[:len(rgb_wv) - s]
                ir_seg = ir_wv[s:]
                x_seg = xw[s:]

            if len(x_seg) < 3 or len(rgb_seg) != len(ir_seg):
                continue

            ax1.plot(
                x_seg,
                rgb_seg,
                color="blue",
                alpha=0.75,
                linewidth=1.2,
                label="RGB shifted in window" if first_rgb_shifted else None
            )
            ax1.plot(
                x_seg,
                ir_seg,
                color="orange",
                alpha=0.75,
                linewidth=1.2,
                label="IR aligned in window" if first_ir_shifted else None
            )
            first_rgb_shifted = False
            first_ir_shifted = False

        title_suffix = "window-wise alignment"

    elif mode == "interpolated":
        rgb_interp = np.full_like(yvs, np.nan, dtype=np.float32)
        ir_interp = np.full_like(yis, np.nan, dtype=np.float32)

        x_to_idx = {int(t): i for i, t in enumerate(xs)}

        for i, t in enumerate(xs):
            lag_val = shift_curve_by_frame.get(int(t), np.nan)
            if not np.isfinite(lag_val):
                continue

            lag = int(round(lag_val))
            rgb_frame = int(t + lag)

            j = x_to_idx.get(rgb_frame, None)
            if j is None:
                continue

            rgb_interp[i] = yvs[j]
            ir_interp[i] = yis[i]

        valid_interp = np.isfinite(rgb_interp) & np.isfinite(ir_interp)

        if valid_interp.sum() >= 3:
            ax1.plot(
                xs[valid_interp],
                rgb_interp[valid_interp],
                color="blue",
                alpha=0.95,
                linewidth=1.8,
                label="RGB aligned by interpolated shift"
            )
            ax1.plot(
                xs[valid_interp],
                ir_interp[valid_interp],
                color="orange",
                alpha=0.95,
                linewidth=1.8,
                label="IR reference"
            )

        title_suffix = "interpolated-shift alignment"

    elif mode == "final":
        rgb_final = np.full_like(yvs, np.nan, dtype=np.float32)
        ir_final = np.full_like(yis, np.nan, dtype=np.float32)

        x_to_idx = {int(t): i for i, t in enumerate(xs)}

        if final_match_map is not None:
            for i, t_ir in enumerate(xs):
                t_ir_int = int(t_ir)
                t_rgb = final_match_map.get(t_ir_int, None)
                if t_rgb is None:
                    continue

                j = x_to_idx.get(int(t_rgb), None)
                if j is None:
                    continue

                rgb_final[i] = yvs[j]
                ir_final[i] = yis[i]

        valid_final = np.isfinite(rgb_final) & np.isfinite(ir_final)

        if valid_final.sum() >= 3:
            ax1.plot(
                xs[valid_final],
                rgb_final[valid_final],
                color="blue",
                alpha=0.95,
                linewidth=2.0,
                label="RGB after final pair matching"
            )
            ax1.plot(
                xs[valid_final],
                ir_final[valid_final],
                color="orange",
                alpha=0.95,
                linewidth=2.0,
                label="IR after final pair matching"
            )

        title_suffix = "final pair matching"

    else:
        raise ValueError(f"Unknown shift-only mode: {mode}")

    ax2.plot(
        xs,
        shift_vals,
        color="black",
        linewidth=1.5,
        label="Interpolated local shift"
    )

    valid_centers = [w["window_center"] for w in valid_windows]
    valid_lags = [w["lag_rgb_minus_ir"] for w in valid_windows]
    if valid_centers:
        ax2.scatter(
            valid_centers,
            valid_lags,
            s=22,
            c="black",
            marker="o",
            label="Valid window centers"
        )

    invalid_centers = [w["window_center"] for w in invalid_windows]
    invalid_lags = [w.get("lag_rgb_minus_ir", None) for w in invalid_windows]

    invalid_xy = []
    for c, l in zip(invalid_centers, invalid_lags):
        if l is None:
            continue
        if np.isfinite(float(l)):
            invalid_xy.append((c, float(l)))

    if invalid_xy:
        ax2.scatter(
            [p[0] for p in invalid_xy],
            [p[1] for p in invalid_xy],
            s=26,
            c="green",
            marker="x",
            label="Invalid window centers"
        )

    xmin = int(xs.min())
    xmax = int(xs.max())
    major_x = 50
    minor_x = 10
    ax1.set_xticks(np.arange((xmin // major_x) * major_x, xmax + major_x, major_x))
    ax1.set_xticks(np.arange((xmin // minor_x) * minor_x, xmax + minor_x, minor_x), minor=True)

    finite_y = np.concatenate([yvs[np.isfinite(yvs)], yis[np.isfinite(yis)]])
    ymax = float(np.max(finite_y)) if finite_y.size > 0 else 1.0
    ymax = max(1.0, float(np.ceil(ymax)))
    ax1.set_yticks(np.arange(0.0, ymax + 1.0, 1.0))
    ax1.set_yticks(np.arange(0.0, ymax + 0.5, 0.5), minor=True)

    ax1.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.8)
    ax1.grid(True, which="minor", linestyle="--", linewidth=0.35, alpha=0.45)

    avg_shift = int(shift_info["avg_shift_rgb_minus_ir"])
    n_windows_total = len(window_estimates)
    n_windows_valid = len(valid_windows)

    text = (
        f"Average shift: {avg_shift}\n"
        f"Windows total: {n_windows_total}\n"
        f"Windows valid: {n_windows_valid}"
    )
    ax1.text(
        0.995, 0.98,
        text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax1.set_title(f"{scene} — shift-only: {title_suffix}", fontsize=TITLE_FONT_SIZE)
    ax1.set_xlabel("Frame index", fontsize=AXIS_LABEL_FONT_SIZE)
    ax1.set_ylabel("Frame-diff score", fontsize=AXIS_LABEL_FONT_SIZE)
    ax2.set_ylabel("Shift (RGB - IR)", fontsize=AXIS_LABEL_FONT_SIZE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.95
    )
    for legline in legend.get_lines():
        legline.set_linewidth(LEGEND_LINEWIDTH)

    ax1.tick_params(axis="x", labelrotation=90, labelsize=AXIS_TICK_FONT_SIZE_X)
    ax1.tick_params(axis="y", labelsize=AXIS_TICK_FONT_SIZE_Y)
    ax2.tick_params(axis="y", labelsize=AXIS_TICK_FONT_SIZE_Y)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():
    vis_img_root = ROOT / "visible" / "images" / SPLIT
    ir_img_root = ROOT / "infrared" / "images" / SPLIT
    vis_lbl_root = ROOT / "visible" / "labels" / SPLIT
    ir_lbl_root = ROOT / "infrared" / "labels" / SPLIT

    assert vis_img_root.exists(), f"Visible images dir not found: {vis_img_root}"
    assert ir_img_root.exists(), f"Infrared images dir not found: {ir_img_root}"
    assert vis_lbl_root.exists(), f"Visible labels dir not found: {vis_lbl_root}"
    assert ir_lbl_root.exists(), f"Infrared labels dir not found: {ir_lbl_root}"

    ensure_dir(OUT_DIR)

    scene_to_vis_stems = build_scene_maps(vis_lbl_root)
    scene_to_ir_stems = build_scene_maps(ir_lbl_root)

    any_vis = next(iter(sorted(vis_img_root.glob("*.jpg"))), None)
    any_ir = next(iter(sorted(ir_img_root.glob("*.jpg"))), None)
    assert any_vis is not None, "No visible images found"
    assert any_ir is not None, "No infrared images found"

    vis0 = cv2.imread(str(any_vis), cv2.IMREAD_COLOR)
    ir0 = cv2.imread(str(any_ir), cv2.IMREAD_COLOR)
    assert vis0 is not None, "Failed to read a sample visible image"
    assert ir0 is not None, "Failed to read a sample infrared image"

    rgb_h, rgb_w = vis0.shape[:2]
    ir_h, ir_w = ir0.shape[:2]

    if TARGET_SCENE not in scene_to_vis_stems or TARGET_SCENE not in scene_to_ir_stems:
        print(f"[ERROR] Target scene {TARGET_SCENE} not found in both modalities!")
        return

    scene = TARGET_SCENE
    print(f"[INFO] Processing scene: {scene}")

    scene_dir = OUT_DIR / scene
    plots_dir = scene_dir / "plots"
    ensure_dir(scene_dir)
    ensure_dir(plots_dir)

    vis_stems = sorted(
        scene_to_vis_stems.get(scene, []),
        key=lambda s: frame_index_from_stem(s) if frame_index_from_stem(s) is not None else -1
    )
    ir_stems = sorted(
        scene_to_ir_stems.get(scene, []),
        key=lambda s: frame_index_from_stem(s) if frame_index_from_stem(s) is not None else -1
    )

    vis_map = build_idx_to_stem(vis_stems)
    ir_map = build_idx_to_stem(ir_stems)

    if len(vis_map) < 5 or len(ir_map) < 5:
        print(f"[WARN] Too few frames in scene {scene}, skipping")
        return

    vis_scores = compute_scene_fdiff_signal(
        idx_to_stem=vis_map,
        img_dir=vis_img_root,
        lbl_dir=vis_lbl_root,
        class_id=CLASS_ID
    )
    ir_scores = compute_scene_fdiff_signal(
        idx_to_stem=ir_map,
        img_dir=ir_img_root,
        lbl_dir=ir_lbl_root,
        class_id=CLASS_ID
    )

    shift_info = estimate_scene_local_shift_from_fdiff(vis_scores, ir_scores)

    final_match_map = build_final_match_mapping(
        ir_map=ir_map,
        vis_map=vis_map,
        ir_lbl_dir=ir_lbl_root,
        vis_lbl_dir=vis_lbl_root,
        shift_curve_by_frame=shift_info.get("shift_curve_by_frame", {}),
        ir_w=ir_w,
        ir_h=ir_h,
        rgb_w=rgb_w,
        rgb_h=rgb_h,
    )

    print(f"[INFO] Saving plots for scene: {scene}")
    print(f"      avg_shift={shift_info['avg_shift_rgb_minus_ir']}, final_matches={len(final_match_map)}")

    save_fdiff_plot(
        scene=scene,
        vis_scores=vis_scores,
        ir_scores=ir_scores,
        shift_info=shift_info,
        out_path=plots_dir / f"{scene}_frame_diff_values_plus_shift.png"
    )

    save_shift_only_plot(
        scene=scene,
        vis_scores=vis_scores,
        ir_scores=ir_scores,
        shift_info=shift_info,
        out_path=plots_dir / f"{scene}_shift_only_window.png",
        mode="window",
        final_match_map=None,
    )

    save_shift_only_plot(
        scene=scene,
        vis_scores=vis_scores,
        ir_scores=ir_scores,
        shift_info=shift_info,
        out_path=plots_dir / f"{scene}_shift_only_interpolated.png",
        mode="interpolated",
        final_match_map=None,
    )

    save_shift_only_plot(
        scene=scene,
        vis_scores=vis_scores,
        ir_scores=ir_scores,
        shift_info=shift_info,
        out_path=plots_dir / f"{scene}_shift_only_final.png",
        mode="final",
        final_match_map=final_match_map,
    )

    print(f"[INFO] Done. Plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()