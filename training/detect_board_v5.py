#!/usr/bin/env python3
"""
Board detection using contour finding + checkerboard validation.

Algorithm:
1. Generate multiple binary images (Canny, adaptive, OTSU, CLAHE, etc.)
2. Find all quadrilateral contours that could be boards
3. For each candidate, warp to square and score by checkerboard pattern
4. Pick the candidate with the highest checkerboard score
5. Optionally refine to inner playing area

Usage: python3 training/detect_board_v5.py <image_path>
"""

import sys
import math
from pathlib import Path
import numpy as np
import cv2


def order_points(pts):
    """Order points as TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # TR
    rect[3] = pts[np.argmax(d)]   # BL
    return rect


def checkerboard_score(image, corners):
    """
    Warp the image using corners and check if it looks like a checkerboard.
    Returns a score from 0 (not a checkerboard) to 1 (perfect checkerboard).
    """
    warp_size = 400
    dst = np.float32([[0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]])

    try:
        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, M, (warp_size, warp_size))
    except cv2.error:
        return 0.0

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    sq = warp_size // 8

    # Sample center of each square
    values = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            cx = c * sq + sq // 2
            cy = r * sq + sq // 2
            margin = sq // 6  # sample a small region around center
            region = gray[max(0, cy - margin):cy + margin, max(0, cx - margin):cx + margin]
            if region.size == 0:
                return 0.0
            values[r, c] = np.mean(region)

    # Check 1: Adjacent squares should alternate brightness
    adj_diffs = 0
    adj_total = 0
    for r in range(8):
        for c in range(7):
            if abs(float(values[r, c]) - float(values[r, c + 1])) > 15:
                adj_diffs += 1
            adj_total += 1
    for r in range(7):
        for c in range(8):
            if abs(float(values[r, c]) - float(values[r + 1, c])) > 15:
                adj_diffs += 1
            adj_total += 1
    alternating_ratio = adj_diffs / adj_total if adj_total > 0 else 0

    # Check 2: Same-parity squares should have similar brightness
    # (light squares should all be light, dark squares all dark)
    light_vals = []
    dark_vals = []
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 0:
                light_vals.append(values[r, c])
            else:
                dark_vals.append(values[r, c])

    light_std = np.std(light_vals)
    dark_std = np.std(dark_vals)
    light_mean = np.mean(light_vals)
    dark_mean = np.mean(dark_vals)

    # Consistency: low std within each group = good
    max_range = max(np.max(values) - np.min(values), 1)
    consistency = 1.0 - (light_std + dark_std) / max_range
    consistency = max(0, consistency)

    # Separation: big difference between light and dark means = good
    separation = abs(light_mean - dark_mean) / 255.0

    # Combined score
    score = alternating_ratio * 0.5 + consistency * 0.25 + separation * 0.25
    return score


def get_binary_images(gray, color_image=None):
    """Generate multiple binary images for contour detection."""
    h, w = gray.shape
    kernel3 = np.ones((3, 3), np.uint8)

    # Canny with different params
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for lo, hi in [(30, 100), (50, 150), (20, 80)]:
        yield cv2.dilate(cv2.Canny(blurred, lo, hi), kernel3, iterations=2)

    # Adaptive threshold
    yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 5)

    # OTSU
    yield cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # CLAHE + Canny
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    for lo, hi in [(30, 100), (50, 150)]:
        yield cv2.dilate(cv2.Canny(enhanced, lo, hi), kernel3, iterations=2)

    # Bilateral filter + Canny (preserves edges, reduces texture)
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
    yield cv2.dilate(cv2.Canny(bilateral, 30, 100), kernel3, iterations=2)

    # Morphological gradient
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel5)
    yield cv2.threshold(morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Color-based: HSV and LAB channels can detect board edges invisible in grayscale
    # (e.g. wooden board on wooden floor — same brightness, different hue/saturation)
    if color_image is not None:
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
        for channel in [hsv[:, :, 0], hsv[:, :, 1], lab[:, :, 1], lab[:, :, 2]]:
            blurred_ch = cv2.GaussianBlur(channel, (5, 5), 0)
            for lo, hi in [(30, 100), (20, 80)]:
                yield cv2.dilate(cv2.Canny(blurred_ch, lo, hi), kernel3, iterations=2)

        # Brightness segmentation: board is often lighter/darker than background
        # Use morphological close+open to get a clean blob, then find its contour
        kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        l_ch = lab[:, :, 0]
        for thresh_val in [90, 100, 110, 120]:
            for inv in [False, True]:
                flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
                _, binary = cv2.threshold(l_ch, thresh_val, 255, flag)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=3)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel5, iterations=2)
                yield binary


def find_candidates(gray, image):
    """Find all quadrilateral contours that could be a chess board."""
    h, w = gray.shape
    seen = set()  # Deduplicate similar candidates
    candidates = []

    for binary in get_binary_images(gray, color_image=image):
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Board should be at least 5% of image (lowered from 10% for zoomed-out shots)
            if area < h * w * 0.05 or area > h * w * 0.98:
                continue
            peri = cv2.arcLength(cnt, True)
            found_quad = False
            for eps_mult in [0.02, 0.03, 0.05, 0.08]:
                approx = cv2.approxPolyDP(cnt, eps_mult * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    ordered = order_points(pts)

                    # Check aspect ratio
                    w1 = np.linalg.norm(ordered[1] - ordered[0])
                    w2 = np.linalg.norm(ordered[2] - ordered[3])
                    h1 = np.linalg.norm(ordered[3] - ordered[0])
                    h2 = np.linalg.norm(ordered[2] - ordered[1])
                    avg_w = (w1 + w2) / 2
                    avg_h = (h1 + h2) / 2
                    if max(avg_w, avg_h) < 1:
                        break
                    aspect = min(avg_w, avg_h) / max(avg_w, avg_h)

                    if aspect > 0.55:
                        # Deduplicate: round corners to nearest 20px
                        key = tuple(np.round(ordered.flatten() / 20).astype(int))
                        if key not in seen:
                            seen.add(key)
                            candidates.append(ordered)
                    found_quad = True
                    break

            # Fallback: use minAreaRect for large blobs that don't approxPolyDP cleanly
            if not found_quad and area > h * w * 0.15:
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if max(rw, rh) > 0 and min(rw, rh) / max(rw, rh) > 0.55:
                    box = cv2.boxPoints(rect).astype(np.float32)
                    # Clip to image bounds
                    box[:, 0] = np.clip(box[:, 0], 0, w - 1)
                    box[:, 1] = np.clip(box[:, 1], 0, h - 1)
                    ordered = order_points(box)
                    key = tuple(np.round(ordered.flatten() / 20).astype(int))
                    if key not in seen:
                        seen.add(key)
                        candidates.append(ordered)

    return candidates


def find_inner_border(warped_gray, warp_size):
    """Find inner playing area in warped image using variance-based detection."""
    s = warp_size

    row_std = np.array([np.std(warped_gray[r, :]) for r in range(s)])
    col_std = np.array([np.std(warped_gray[:, c]) for c in range(s)])

    kernel_size = max(3, s // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    row_std_smooth = np.convolve(row_std, np.ones(kernel_size) / kernel_size, mode='same')
    col_std_smooth = np.convolve(col_std, np.ones(kernel_size) / kernel_size, mode='same')

    mid_start, mid_end = s // 4, 3 * s // 4
    row_threshold = np.median(row_std_smooth[mid_start:mid_end]) * 0.5
    col_threshold = np.median(col_std_smooth[mid_start:mid_end]) * 0.5

    top = 0
    for r in range(s // 4):
        if row_std_smooth[r] > row_threshold:
            top = r
            break

    bottom = s - 1
    for r in range(s - 1, 3 * s // 4, -1):
        if row_std_smooth[r] > row_threshold:
            bottom = r
            break

    left = 0
    for c in range(s // 4):
        if col_std_smooth[c] > col_threshold:
            left = c
            break

    right = s - 1
    for c in range(s - 1, 3 * s // 4, -1):
        if col_std_smooth[c] > col_threshold:
            right = c
            break

    return left, top, right, bottom


def find_inner_contour(warped_gray, warp_size):
    """Find inner playing area rectangle via contour detection in warped image."""
    s = warp_size
    blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_corners = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < s * s * 0.50 or area > s * s * 0.98:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps_mult in [0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps_mult * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                ordered = order_points(pts)
                w1 = np.linalg.norm(ordered[1] - ordered[0])
                h1 = np.linalg.norm(ordered[3] - ordered[0])
                aspect = min(w1, h1) / max(w1, h1)
                if aspect > 0.8:
                    score = area * aspect
                    if score > best_score:
                        best_score = score
                        best_corners = ordered
                break

    return best_corners


def refine_to_inner(orig, outer_corners):
    """Stage 2: Refine outer board corners to inner playing area."""
    warp_size = 1000
    dst = np.float32([[0, 0], [warp_size, 0],
                      [warp_size, warp_size], [0, warp_size]])
    M_outer = cv2.getPerspectiveTransform(outer_corners, dst)
    warped = cv2.warpPerspective(orig, M_outer, (warp_size, warp_size))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Try variance-based
    left, top, right, bottom = find_inner_border(warped_gray, warp_size)
    margin_pct = [left / warp_size, top / warp_size,
                  (warp_size - right) / warp_size, (warp_size - bottom) / warp_size]
    margins_ok = all(0.005 < m < 0.15 for m in margin_pct)

    # Try contour-based
    inner_contour = find_inner_contour(warped_gray, warp_size)

    if inner_contour is not None:
        inner_pts = inner_contour
    elif margins_ok:
        inner_pts = np.float32([
            [left, top], [right, top], [right, bottom], [left, bottom]
        ])
    else:
        # No refinement needed / possible
        return outer_corners

    # Map inner points back to original coordinates
    M_inv = cv2.getPerspectiveTransform(dst, outer_corners)
    inner_pts_h = np.hstack([inner_pts, np.ones((4, 1))]).T
    orig_pts = M_inv @ inner_pts_h
    orig_pts = (orig_pts[:2] / orig_pts[2]).T

    return orig_pts.astype(np.float32)


def detect_board_corners(image_path, debug_dir=None):
    """
    Detect chess board corners.
    1. Find all quadrilateral candidates
    2. Score each by checkerboard pattern
    3. Pick the best, refine to inner playing area
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read {image_path}")

    orig = image.copy()
    h, w = image.shape[:2]

    # Resize for processing
    max_dim = 1500
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Stage 1: Find all candidates
    candidates = find_candidates(gray, image)
    if not candidates:
        print("No candidates found!")
        return None

    print(f"Found {len(candidates)} candidates")

    # Stage 2: Score each candidate by checkerboard pattern
    best_score = 0
    best_corners = None

    for corners in candidates:
        # Scale corners back to original image coordinates
        corners_orig = corners / scale
        score = checkerboard_score(orig, corners_orig)
        if score > best_score:
            best_score = score
            best_corners = corners_orig

    if best_corners is None or best_score < 0.15:
        print(f"No good checkerboard found (best score: {best_score:.3f})")
        return None

    print(f"Best checkerboard score: {best_score:.3f}")

    # Stage 3: Refine to inner playing area
    final_corners = refine_to_inner(orig, best_corners)

    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Show final corners on original
        debug_orig = orig.copy()
        labels = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(final_corners):
            cv2.circle(debug_orig, (int(x), int(y)), 15, (0, 255, 0), -1)
            cv2.putText(debug_orig, labels[i], (int(x) + 20, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i in range(4):
            pt1 = tuple(final_corners[i].astype(int))
            pt2 = tuple(final_corners[(i + 1) % 4].astype(int))
            cv2.line(debug_orig, pt1, pt2, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "v5_corners.jpg"), debug_orig)

        # Final warp with grid
        warp_size = 800
        dst = np.float32([[0, 0], [warp_size, 0],
                          [warp_size, warp_size], [0, warp_size]])
        M = cv2.getPerspectiveTransform(final_corners, dst)
        warped = cv2.warpPerspective(orig, M, (warp_size, warp_size))
        sq = warp_size // 8
        for i in range(9):
            cv2.line(warped, (0, i * sq), (warp_size, i * sq), (0, 0, 255), 2)
            cv2.line(warped, (i * sq, 0), (i * sq, warp_size), (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "v5_warped.jpg"), warped)

    return final_corners


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 training/detect_board_v5.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    debug_dir = Path("training/debug_output")
    debug_dir.mkdir(parents=True, exist_ok=True)

    corners = detect_board_corners(image_path, debug_dir)
    if corners is not None:
        print(f"\nDetected corners:")
        labels = ["TL", "TR", "BR", "BL"]
        for label, (x, y) in zip(labels, corners):
            print(f"  {label}: ({x:.0f}, {y:.0f})")
    else:
        print("Failed!")


if __name__ == "__main__":
    main()
