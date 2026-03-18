#!/usr/bin/env python3
"""Test floor tile penalty on the 3 problem images."""

import sys
import json
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from detect_board_v5 import find_candidates, checkerboard_score, order_points

DATA_DIR = Path(__file__).parent / "data"

# Load GT
with open(DATA_DIR / "annotations.json") as f:
    data = json.load(f)
id_to_file = {}
for img in data["images"]:
    fname = img.get("file_name") or img.get("path", "")
    if fname: id_to_file[img["id"]] = fname
id_to_corners = {}
for ann in data["annotations"]["corners"]:
    c = ann["corners"]
    id_to_corners[ann["image_id"]] = np.array(
        [c["top_left"], c["top_right"], c["bottom_right"], c["bottom_left"]], dtype=np.float32)
file_to_gt = {}
for img_id, fname in id_to_file.items():
    if img_id in id_to_corners:
        file_to_gt[Path(fname).name] = id_to_corners[img_id]


def translation_invariance_score(image, corners):
    """Check if shifting the quad by ~1 square still gives high scores.
    Returns a penalty factor (low = likely floor tiles, high = likely real board).
    """
    h, w = image.shape[:2]
    base_score = checkerboard_score(image, corners)
    if base_score < 0.1:
        return 1.0

    # Estimate square size from quad dimensions
    w1 = np.linalg.norm(corners[1] - corners[0])
    h1 = np.linalg.norm(corners[3] - corners[0])
    sq_w = w1 / 8
    sq_h = h1 / 8

    shift_scores = []
    for dx, dy in [(sq_w, 0), (-sq_w, 0), (0, sq_h), (0, -sq_h),
                   (sq_w, sq_h), (-sq_w, -sq_h)]:
        shifted = corners + np.array([dx, dy], dtype=np.float32)
        # Check bounds
        if (shifted[:, 0].min() < 0 or shifted[:, 0].max() >= w or
            shifted[:, 1].min() < 0 or shifted[:, 1].max() >= h):
            continue
        sc = checkerboard_score(image, shifted)
        shift_scores.append(sc)

    if not shift_scores:
        return 1.0

    avg_shifted = np.mean(shift_scores)
    max_shifted = np.max(shift_scores)
    ratio = avg_shifted / base_score

    return base_score, avg_shifted, max_shifted, ratio


IMAGES = [
    DATA_DIR / "chessred2k/images/41/G041_IMG008.jpg",
    DATA_DIR / "chessred2k/images/19/G019_IMG080.jpg",
    DATA_DIR / "chessred2k/images/22/G022_IMG042.jpg",
]

for img_path in IMAGES:
    if not img_path.exists():
        continue
    print(f"\n{'='*60}")
    print(f"{img_path.name}")
    print(f"{'='*60}")

    orig = cv2.imread(str(img_path))
    h, w = orig.shape[:2]
    gt = file_to_gt.get(img_path.name)

    max_dim = 1500
    scale = min(max_dim / max(w, h), 1.0)
    resized = cv2.resize(orig, (int(w * scale), int(h * scale))) if scale < 1.0 else orig.copy()
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    candidates = find_candidates(gray, resized)

    proc_h, proc_w = gray.shape
    img_center = np.array([proc_w / 2, proc_h / 2])
    img_diag = np.sqrt(proc_w ** 2 + proc_h ** 2)

    all_scored = []
    for corners in candidates:
        corners_orig = corners / scale
        cb_score = checkerboard_score(orig, corners_orig)
        quad_center = np.mean(corners, axis=0)
        center_dist = np.linalg.norm(quad_center - img_center) / img_diag
        center_weight = max(0.3, np.exp(-3.0 * center_dist ** 2))
        quad_area = cv2.contourArea(corners.reshape(-1, 1, 2))
        img_area = proc_w * proc_h
        area_ratio = quad_area / img_area
        if area_ratio > 0.85: size_weight = 0.2
        elif area_ratio > 0.7: size_weight = 0.5
        else: size_weight = 1.0
        score = cb_score * center_weight * size_weight

        # Test translation invariance
        ti_result = translation_invariance_score(orig, corners_orig)
        if isinstance(ti_result, tuple):
            base, avg_sh, max_sh, ratio = ti_result
            all_scored.append((score, cb_score, ratio, area_ratio, corners_orig))
        else:
            all_scored.append((score, cb_score, -1, area_ratio, corners_orig))

    all_scored.sort(key=lambda x: -x[0])
    print(f"\n  Top 8 candidates (score, cb, shift_ratio, area):")
    for i, (score, cb, ratio, ar, co) in enumerate(all_scored[:8]):
        marker = ""
        if gt is not None:
            d = [np.linalg.norm(co[j] - gt[j]) for j in range(4)]
            rmse = np.sqrt(np.mean(np.array(d)**2))
            marker = f"  RMSE={rmse:.0f}"
        print(f"    #{i+1}: score={score:.3f}  cb={cb:.3f}  shift_ratio={ratio:.3f}  area={ar:.1%}{marker}")

    # Also test GT corners
    if gt is not None:
        gt_ti = translation_invariance_score(orig, gt)
        if isinstance(gt_ti, tuple):
            base, avg_sh, max_sh, ratio = gt_ti
            print(f"\n  GT: cb={base:.3f}  shift_ratio={ratio:.3f}")

print("\nDone!")
