#!/usr/bin/env python3
"""
Go through all test images, classify each as top-down or not,
and check if board detection succeeded.
Saves a grid of failed/weak images for visual inspection.
"""

import sys
import random
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from detect_board_v5 import detect_board_corners

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "debug_output"


def get_images(dataset, max_n):
    if dataset == "kaggle":
        img_dir = DATA_DIR / "kaggle" / "data"
        images = sorted([f for f in img_dir.iterdir() if f.suffix == '.jpg'])
    else:
        img_dir = DATA_DIR / dataset / "images"
        images = []
        for subdir in sorted(img_dir.iterdir()):
            if subdir.is_dir():
                for f in sorted(subdir.iterdir()):
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        images.append(f)
    random.seed(42)
    random.shuffle(images)
    return images[:max_n]


def check_warp_quality(image_path, corners):
    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0, None
    warp_size = 400
    dst = np.float32([[0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]])
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (warp_size, warp_size))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sq = warp_size // 8
    values = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            cx = c * sq + sq // 2
            cy = r * sq + sq // 2
            region = gray[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
            values[r, c] = np.mean(region)
    diffs = 0
    total = 0
    for r in range(8):
        for c in range(7):
            if abs(float(values[r, c]) - float(values[r, c+1])) > 20:
                diffs += 1
            total += 1
    for r in range(7):
        for c in range(8):
            if abs(float(values[r, c]) - float(values[r+1, c])) > 20:
                diffs += 1
            total += 1
    return diffs / total, warped


def make_thumbnail(img, size=300):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [("chessred2k", 50), ("kaggle", 50), ("chessred", 50)]

    for dataset, max_n in datasets:
        images = get_images(dataset, max_n)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({len(images)} images)")
        print(f"{'='*60}")

        failed_imgs = []  # (path, original_img, reason)
        weak_imgs = []    # (path, original_img, warped_img, score)
        good_count = 0

        for i, img_path in enumerate(images):
            orig = cv2.imread(str(img_path))
            if orig is None:
                continue

            try:
                corners = detect_board_corners(str(img_path))
            except Exception:
                corners = None

            if corners is None:
                failed_imgs.append((img_path, orig, "no detection"))
                continue

            score, warped = check_warp_quality(img_path, corners)
            if score > 0.5:
                good_count += 1
            elif score > 0.3:
                weak_imgs.append((img_path, orig, warped, score))
            else:
                failed_imgs.append((img_path, orig, f"bad warp ({score:.2f})"))

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(images)} processed...")

        print(f"\n  GOOD: {good_count}")
        print(f"  WEAK: {len(weak_imgs)}")
        print(f"  FAIL: {len(failed_imgs)}")

        # Save grid of failed images (original only)
        if failed_imgs:
            cols = min(5, len(failed_imgs))
            rows = (len(failed_imgs) + cols - 1) // cols
            cell = 300
            grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
            for idx, (path, orig, reason) in enumerate(failed_imgs):
                r, c = idx // cols, idx % cols
                thumb = make_thumbnail(orig, cell)
                cv2.putText(thumb, path.name[:25], (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(thumb, reason, (5, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                grid[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = thumb
            cv2.imwrite(str(OUT_DIR / f"failures_{dataset}.jpg"), grid)
            print(f"  Saved: debug_output/failures_{dataset}.jpg")

        # Save grid of weak images (original + warped side by side)
        if weak_imgs:
            cols = min(4, len(weak_imgs))
            rows = (len(weak_imgs) + cols - 1) // cols
            cell = 300
            grid = np.zeros((rows * cell, cols * cell * 2, 3), dtype=np.uint8)
            for idx, (path, orig, warped, score) in enumerate(weak_imgs):
                r, c = idx // cols, idx % cols
                thumb_orig = make_thumbnail(orig, cell)
                thumb_warp = make_thumbnail(warped, cell)
                cv2.putText(thumb_orig, f"{path.name[:25]}", (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(thumb_orig, f"score={score:.2f}", (5, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                grid[r*cell:(r+1)*cell, c*cell*2:(c*2+1)*cell] = thumb_orig
                grid[r*cell:(r+1)*cell, (c*2+1)*cell:(c+1)*cell*2] = thumb_warp
            cv2.imwrite(str(OUT_DIR / f"weak_{dataset}.jpg"), grid)
            print(f"  Saved: debug_output/weak_{dataset}.jpg")


if __name__ == "__main__":
    main()
