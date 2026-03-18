#!/usr/bin/env python3
"""
Evaluate board detection against ground truth corners from ChessReD2K.
Reports per-image RMSE, IoU of warped board region, and distributions.
Usage: python3 training/eval_gt_corners.py [--max N]
"""

import sys
import json
import random
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from detect_board_v5 import detect_board_corners, order_points

DATA_DIR = Path(__file__).parent / "data"
ANN_FILE = DATA_DIR / "annotations.json"


def load_annotations():
    """Load ChessReD2K annotations and build image_id -> corners + filename mapping."""
    with open(ANN_FILE) as f:
        data = json.load(f)

    # Build image_id -> filename
    id_to_file = {}
    for img in data["images"]:
        fname = img.get("file_name") or img.get("path", "")
        if fname:
            id_to_file[img["id"]] = fname

    # Build image_id -> corners (ordered TL, TR, BR, BL)
    id_to_corners = {}
    for ann in data["annotations"]["corners"]:
        c = ann["corners"]
        corners = np.array([
            c["top_left"], c["top_right"], c["bottom_right"], c["bottom_left"]
        ], dtype=np.float32)
        id_to_corners[ann["image_id"]] = corners

    return id_to_file, id_to_corners, data.get("splits", {})


def match_corners(pred, gt):
    """
    Match predicted corners to GT corners using nearest-neighbor.
    Returns per-corner distances and mean RMSE.
    """
    # Try both the predicted order and all permutations of matching
    # Best approach: for each GT corner, find nearest predicted corner
    dists = []
    used = set()
    for g in gt:
        best_d = float("inf")
        best_j = -1
        for j, p in enumerate(pred):
            if j in used:
                continue
            d = np.linalg.norm(p - g)
            if d < best_d:
                best_d = d
                best_j = j
        used.add(best_j)
        dists.append(best_d)
    return np.array(dists), np.sqrt(np.mean(np.array(dists) ** 2))


def compute_iou(pred_corners, gt_corners, img_size=3072):
    """Compute IoU between predicted and GT quadrilaterals."""
    # Create masks
    mask_pred = np.zeros((img_size, img_size), dtype=np.uint8)
    mask_gt = np.zeros((img_size, img_size), dtype=np.uint8)

    pts_pred = pred_corners.astype(np.int32).reshape((-1, 1, 2))
    pts_gt = gt_corners.astype(np.int32).reshape((-1, 1, 2))

    # Clip to image bounds
    pts_pred = np.clip(pts_pred, 0, img_size - 1)
    pts_gt = np.clip(pts_gt, 0, img_size - 1)

    cv2.fillPoly(mask_pred, [pts_pred], 1)
    cv2.fillPoly(mask_gt, [pts_gt], 1)

    intersection = np.sum(mask_pred & mask_gt)
    union = np.sum(mask_pred | mask_gt)

    return intersection / union if union > 0 else 0.0


def main():
    max_n = 200
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        if idx + 1 < len(sys.argv):
            max_n = int(sys.argv[idx + 1])

    print(f"Loading annotations...")
    id_to_file, id_to_corners, splits = load_annotations()

    # Use all images that have both corners and exist on disk
    img_dir = DATA_DIR / "chessred2k" / "images"

    # Build a quick lookup of files on disk
    disk_files = {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            disk_files[p.name] = p

    available = []
    for img_id in id_to_corners:
        if img_id not in id_to_file:
            continue
        fname = Path(id_to_file[img_id]).name
        if fname in disk_files:
            available.append((img_id, disk_files[fname]))

    random.seed(42)
    random.shuffle(available)
    available = available[:max_n]

    print(f"Evaluating {len(available)} images (val+test split)\n")

    rmses = []
    ious = []
    failures = []
    corner_dists_all = []

    for i, (img_id, img_path) in enumerate(available):
        gt = id_to_corners[img_id]
        diag = np.sqrt(3072**2 + 3072**2)

        try:
            pred = detect_board_corners(str(img_path))
        except Exception as e:
            failures.append((img_path.name, f"error: {e}"))
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(available)}...", flush=True)
            continue

        if pred is None:
            failures.append((img_path.name, "no detection"))
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(available)}...", flush=True)
            continue

        dists, rmse = match_corners(pred, gt)
        corner_dists_all.extend(dists.tolist())
        rmses.append(rmse)

        iou = compute_iou(pred, gt)
        ious.append(iou)

        pct = rmse / diag * 100
        status = "GOOD" if pct < 2 else "OK" if pct < 5 else "BAD"
        if status == "BAD":
            print(f"  [{i+1}] BAD  {img_path.name}: RMSE={rmse:.0f}px ({pct:.1f}% diag), IoU={iou:.3f}")
        elif (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(available)}...", flush=True)

    print(f"\n{'='*60}")
    print(f"Ground Truth Corner Evaluation — ChessReD2K ({len(available)} images)")
    print(f"{'='*60}")
    print(f"  Detected: {len(rmses)}/{len(available)} ({len(rmses)/len(available)*100:.1f}%)")
    print(f"  Failed:   {len(failures)}/{len(available)}")

    if rmses:
        rmses = np.array(rmses)
        ious = np.array(ious)
        diag = np.sqrt(3072**2 + 3072**2)

        print(f"\n  Corner RMSE (pixels, on 3072x3072 images):")
        print(f"    Mean:   {np.mean(rmses):.1f}px ({np.mean(rmses)/diag*100:.2f}% diagonal)")
        print(f"    Median: {np.median(rmses):.1f}px ({np.median(rmses)/diag*100:.2f}% diagonal)")
        print(f"    Std:    {np.std(rmses):.1f}px")
        print(f"    Min:    {np.min(rmses):.1f}px")
        print(f"    Max:    {np.max(rmses):.1f}px")

        for thresh_pct in [1, 2, 3, 5, 10]:
            thresh_px = diag * thresh_pct / 100
            within = np.sum(rmses < thresh_px)
            print(f"    <{thresh_pct}% diagonal ({thresh_px:.0f}px): {within}/{len(rmses)} ({within/len(rmses)*100:.1f}%)")

        print(f"\n  Quad IoU:")
        print(f"    Mean:   {np.mean(ious):.3f}")
        print(f"    Median: {np.median(ious):.3f}")
        print(f"    >0.9:   {np.sum(ious > 0.9)}/{len(ious)} ({np.sum(ious > 0.9)/len(ious)*100:.1f}%)")
        print(f"    >0.8:   {np.sum(ious > 0.8)}/{len(ious)} ({np.sum(ious > 0.8)/len(ious)*100:.1f}%)")
        print(f"    >0.7:   {np.sum(ious > 0.7)}/{len(ious)} ({np.sum(ious > 0.7)/len(ious)*100:.1f}%)")

        # Worst 10
        sorted_idx = np.argsort(rmses)[::-1]
        print(f"\n  Worst 10:")
        for idx in sorted_idx[:10]:
            img_id, img_path = available[idx]
            pct = rmses[idx] / diag * 100
            print(f"    {img_path.name}: RMSE={rmses[idx]:.0f}px ({pct:.1f}%), IoU={ious[idx]:.3f}")

    if failures:
        print(f"\n  Failures:")
        for name, reason in failures[:10]:
            print(f"    {name}: {reason}")


if __name__ == "__main__":
    main()
