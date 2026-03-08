#!/usr/bin/env python3
"""
Evaluate board detection on dataset images.
Usage: python3 training/eval_board_detection.py [--dataset chessred2k|chessred|kaggle] [--max N]
"""

import sys
import json
import random
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from detect_board_v5 import detect_board_corners

DATA_DIR = Path(__file__).parent / "data"


def get_chessred_images(variant="chessred2k", max_n=50):
    """Get image paths from ChessReD dataset."""
    img_dir = DATA_DIR / variant / "images"
    images = []
    for subdir in sorted(img_dir.iterdir()):
        if subdir.is_dir():
            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    images.append(f)
    random.seed(42)
    random.shuffle(images)
    return images[:max_n]


def get_chessred_corners(image_path, variant="chessred2k"):
    """Get ground truth corners from ChessReD annotations."""
    ann_file = DATA_DIR / variant / "annotations.json" if (DATA_DIR / variant / "annotations.json").exists() else Path("annotations.json")
    # Try to find annotations
    for p in [DATA_DIR / variant / "annotations.json", Path("annotations.json")]:
        if p.exists():
            ann_file = p
            break
    else:
        return None

    with open(ann_file) as f:
        annotations = json.load(f)

    # Find by filename
    fname = image_path.name
    for ann in annotations:
        if ann.get("file_name", "").endswith(fname) or ann.get("image", "").endswith(fname):
            corners = ann.get("corners")
            if corners:
                return np.array(corners, dtype=np.float32)
    return None


def get_kaggle_images(max_n=50):
    """Get image paths from Kaggle dataset."""
    img_dir = DATA_DIR / "kaggle" / "data"
    images = sorted([f for f in img_dir.iterdir() if f.suffix == '.jpg'])
    random.seed(42)
    random.shuffle(images)
    return images[:max_n]


def get_kaggle_corners(image_path):
    """Get ground truth from Kaggle JSON sidecar."""
    json_path = image_path.with_suffix('.json')
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    corners = data.get("corners")
    if corners:
        return np.array(corners, dtype=np.float32)
    return None


def corner_distance(pred, gt):
    """Average distance between predicted and ground truth corners (normalized by image diagonal)."""
    # Match corners: for each gt corner, find nearest pred corner
    dists = []
    for g in gt:
        d = np.linalg.norm(pred - g, axis=1)
        dists.append(np.min(d))
    return np.mean(dists)


def check_warp_quality(image_path, corners):
    """Check if warped board has checkerboard pattern (alternating light/dark)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    warp_size = 400
    dst = np.float32([[0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]])
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (warp_size, warp_size))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    sq = warp_size // 8
    # Sample center of each square, check alternating pattern
    values = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            cx = c * sq + sq // 2
            cy = r * sq + sq // 2
            region = gray[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
            values[r, c] = np.mean(region)

    # Check if alternating: adjacent squares should have different brightness
    diffs = 0
    total = 0
    for r in range(8):
        for c in range(7):
            diff = abs(float(values[r, c]) - float(values[r, c+1]))
            if diff > 20:  # Significant brightness difference
                diffs += 1
            total += 1
    for r in range(7):
        for c in range(8):
            diff = abs(float(values[r, c]) - float(values[r+1, c]))
            if diff > 20:
                diffs += 1
            total += 1

    # On a perfect checkerboard, all adjacent squares differ
    ratio = diffs / total
    return ratio


def main():
    dataset = "chessred2k"
    max_n = 50

    for arg in sys.argv[1:]:
        if arg.startswith("--dataset"):
            continue
        if arg in ("chessred2k", "chessred", "kaggle"):
            dataset = arg
        elif arg == "--max":
            continue
        else:
            try:
                max_n = int(arg)
            except ValueError:
                pass

    if "--dataset" in sys.argv:
        idx = sys.argv.index("--dataset")
        if idx + 1 < len(sys.argv):
            dataset = sys.argv[idx + 1]
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        if idx + 1 < len(sys.argv):
            max_n = int(sys.argv[idx + 1])

    print(f"Dataset: {dataset}, max images: {max_n}")

    if dataset == "kaggle":
        images = get_kaggle_images(max_n)
    else:
        images = get_chessred_images(dataset, max_n)

    print(f"Testing {len(images)} images...\n")

    detected = 0
    failed = 0
    warp_scores = []
    corner_errors = []

    for i, img_path in enumerate(images):
        try:
            corners = detect_board_corners(str(img_path))
        except Exception as e:
            print(f"  [{i+1}] ERROR {img_path.name}: {e}")
            failed += 1
            continue

        if corners is None:
            print(f"  [{i+1}] FAIL  {img_path.name}")
            failed += 1
            continue

        detected += 1
        score = check_warp_quality(img_path, corners)
        warp_scores.append(score)

        status = "GOOD" if score > 0.5 else "WEAK" if score > 0.3 else "BAD"
        if status != "GOOD":
            print(f"  [{i+1}] {status}  {img_path.name} (checkerboard={score:.2f})")

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(images)} done ({detected} detected, {failed} failed)")

    print(f"\n{'='*50}")
    print(f"Results: {dataset} ({len(images)} images)")
    print(f"  Detected: {detected}/{len(images)} ({detected/len(images)*100:.1f}%)")
    print(f"  Failed:   {failed}/{len(images)} ({failed/len(images)*100:.1f}%)")
    if warp_scores:
        good = sum(1 for s in warp_scores if s > 0.5)
        weak = sum(1 for s in warp_scores if 0.3 < s <= 0.5)
        bad = sum(1 for s in warp_scores if s <= 0.3)
        print(f"  Warp quality (of detected):")
        print(f"    GOOD (>0.5): {good} ({good/len(warp_scores)*100:.1f}%)")
        print(f"    WEAK (0.3-0.5): {weak} ({weak/len(warp_scores)*100:.1f}%)")
        print(f"    BAD  (<0.3): {bad} ({bad/len(warp_scores)*100:.1f}%)")
        print(f"    Mean checkerboard score: {np.mean(warp_scores):.3f}")


if __name__ == "__main__":
    main()
