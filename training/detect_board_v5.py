#!/usr/bin/env python3
"""
Two-stage board detection:
1. Find outer board rectangle via contour detection (reliable)
2. Warp to square, then find inner playing area boundary

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


def get_binary_images(gray):
    """Generate multiple binary images."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    yield cv2.dilate(canny, kernel, iterations=2)

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, 5)
    yield adaptive

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    yield otsu

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    canny2 = cv2.Canny(enhanced, 50, 150)
    yield cv2.dilate(canny2, np.ones((3, 3), np.uint8), iterations=2)

    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
    canny3 = cv2.Canny(bilateral, 30, 100)
    yield cv2.dilate(canny3, np.ones((3, 3), np.uint8), iterations=2)


def find_outer_board(gray, image):
    """Stage 1: Find the outer board rectangle."""
    h, w = gray.shape
    results = []

    for binary in get_binary_images(gray):
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.10 or area > h * w * 0.95:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps_mult in [0.02, 0.03, 0.05, 0.08]:
                approx = cv2.approxPolyDP(cnt, eps_mult * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    ordered = order_points(pts)
                    w1 = np.linalg.norm(ordered[1] - ordered[0])
                    w2 = np.linalg.norm(ordered[2] - ordered[3])
                    h1 = np.linalg.norm(ordered[3] - ordered[0])
                    h2 = np.linalg.norm(ordered[2] - ordered[1])
                    avg_w = (w1 + w2) / 2
                    avg_h = (h1 + h2) / 2
                    aspect = min(avg_w, avg_h) / max(avg_w, avg_h)
                    if aspect > 0.6:
                        results.append((area * aspect, ordered))
                    break

    if not results:
        return None
    results.sort(key=lambda x: x[0], reverse=True)
    return results[0][1]


def find_inner_border(warped_gray, warp_size, debug_dir=None):
    """
    Stage 2: In the warped image, find where the playing area starts.
    Scan from each edge inward looking for the dark border line or
    the first checkerboard square edge.
    """
    s = warp_size

    # The warped image should have the board roughly centered.
    # The wooden border creates a light-colored margin around the playing area.
    # The playing area starts with either a light or dark square at each corner.
    # The dark border line (if present) is a thin dark line separating border from grid.

    # Strategy: for each edge, find the strongest vertical/horizontal edge
    # (the boundary between wooden border and playing area)

    # Use Sobel gradients
    sobel_x = cv2.Sobel(warped_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(warped_gray, cv2.CV_64F, 0, 1, ksize=3)

    # For left edge: scan columns from left, looking for strong vertical gradient
    # For top edge: scan rows from top, looking for strong horizontal gradient
    # etc.

    margins = [0, 0, 0, 0]  # left, top, right, bottom

    # Sample the central 60% to avoid corners
    c_start = int(s * 0.2)
    c_end = int(s * 0.8)

    # Left margin: scan columns, look at vertical gradient
    for col in range(s // 4):
        grad_sum = np.mean(np.abs(sobel_x[c_start:c_end, col]))
        if grad_sum > 30:
            margins[0] = col
            break

    # Right margin: scan from right
    for col in range(s - 1, s * 3 // 4, -1):
        grad_sum = np.mean(np.abs(sobel_x[c_start:c_end, col]))
        if grad_sum > 30:
            margins[2] = s - 1 - col
            break

    # Top margin: scan rows
    for row in range(s // 4):
        grad_sum = np.mean(np.abs(sobel_y[row, c_start:c_end]))
        if grad_sum > 30:
            margins[1] = row
            break

    # Bottom margin: scan from bottom
    for row in range(s - 1, s * 3 // 4, -1):
        grad_sum = np.mean(np.abs(sobel_y[row, c_start:c_end]))
        if grad_sum > 30:
            margins[3] = s - 1 - row
            break

    return margins


def find_inner_border_v2(warped_gray, warp_size, debug_dir=None):
    """
    Better approach: detect the checkerboard pattern to find where it starts.
    A chess board has alternating light/dark squares. The variance along
    a row/column shows peaks at square boundaries. We can detect where
    these peaks first appear (= start of playing area).
    """
    s = warp_size

    # Compute row-wise and column-wise variance in a sliding window
    # The checkerboard creates a periodic high-low-high-low pattern
    # The wooden border is mostly uniform (low variance)

    # For each row, compute the standard deviation of pixel values
    row_std = np.array([np.std(warped_gray[r, :]) for r in range(s)])
    col_std = np.array([np.std(warped_gray[:, c]) for c in range(s)])

    # The playing area rows/columns will have high std (alternating light/dark)
    # The border rows/columns will have low std (uniform wood)

    # Smooth to reduce noise
    kernel_size = max(3, s // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    row_std_smooth = np.convolve(row_std, np.ones(kernel_size)/kernel_size, mode='same')
    col_std_smooth = np.convolve(col_std, np.ones(kernel_size)/kernel_size, mode='same')

    # Find threshold: the playing area should have std > some threshold
    # Use the median of the middle portion as reference
    mid_start, mid_end = s // 4, 3 * s // 4
    row_threshold = np.median(row_std_smooth[mid_start:mid_end]) * 0.5
    col_threshold = np.median(col_std_smooth[mid_start:mid_end]) * 0.5

    # Find first/last row/col above threshold
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

    if debug_dir:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.plot(row_std_smooth, label='row std')
        ax1.axhline(y=row_threshold, color='r', linestyle='--', label='threshold')
        ax1.axvline(x=top, color='g', linestyle='--', label=f'top={top}')
        ax1.axvline(x=bottom, color='g', linestyle='--', label=f'bottom={bottom}')
        ax1.set_title('Row-wise std')
        ax1.legend()
        ax2.plot(col_std_smooth, label='col std')
        ax2.axhline(y=col_threshold, color='r', linestyle='--', label='threshold')
        ax2.axvline(x=left, color='g', linestyle='--', label=f'left={left}')
        ax2.axvline(x=right, color='g', linestyle='--', label=f'right={right}')
        ax2.set_title('Col-wise std')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(str(debug_dir / "v5_variance_profile.png"))
        plt.close()

    return left, top, right, bottom


def find_inner_border_v3(warped_gray, warp_size, debug_dir=None):
    """
    Most robust: find the dark border line using edge detection in warped image.
    After warping to the outer board, look for the largest inner rectangle.
    """
    s = warp_size

    # Look for the inner border line
    blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_corners = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Inner rectangle should be 50-95% of warped image
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
                if aspect > 0.8:  # Should be very square for inner playing area
                    score = area * aspect
                    if score > best_score:
                        best_score = score
                        best_corners = ordered
                break

    if debug_dir and best_corners is not None:
        debug = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
        for i in range(4):
            pt1 = tuple(best_corners[i].astype(int))
            pt2 = tuple(best_corners[(i+1) % 4].astype(int))
            cv2.line(debug, pt1, pt2, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "v5_inner_contour.jpg"), debug)

    return best_corners


def detect_board_corners(image_path, debug_dir=None):
    """Two-stage board detection."""
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

    # Stage 1: Find outer board rectangle
    outer_corners = find_outer_board(gray, image)
    if outer_corners is None:
        print("Stage 1 failed: could not find outer board!")
        return None

    outer_corners_orig = outer_corners / scale
    print(f"Stage 1: outer board found")

    # Warp to square using outer corners
    warp_size = 1000
    dst = np.float32([[0, 0], [warp_size, 0],
                      [warp_size, warp_size], [0, warp_size]])
    M_outer = cv2.getPerspectiveTransform(outer_corners_orig, dst)
    warped = cv2.warpPerspective(orig, M_outer, (warp_size, warp_size))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "v5_01_warped_outer.jpg"), warped)

    # Stage 2: Find inner playing area
    # Try multiple methods and pick the best

    # Method A: Variance-based (checkerboard pattern detection)
    left, top, right, bottom = find_inner_border_v2(warped_gray, warp_size, debug_dir)
    print(f"Stage 2 (variance): margins L={left} T={top} R={warp_size-right} B={warp_size-bottom}")

    # Method B: Contour-based (find inner rectangle)
    inner_contour = find_inner_border_v3(warped_gray, warp_size, debug_dir)
    if inner_contour is not None:
        ic_left = min(inner_contour[0][0], inner_contour[3][0])
        ic_top = min(inner_contour[0][1], inner_contour[1][1])
        ic_right = max(inner_contour[1][0], inner_contour[2][0])
        ic_bottom = max(inner_contour[2][1], inner_contour[3][1])
        print(f"Stage 2 (contour): L={ic_left:.0f} T={ic_top:.0f} R={ic_right:.0f} B={ic_bottom:.0f}")

    # Use the variance-based margins if they seem reasonable, else contour
    # Reasonable = margins are 1-15% of image size
    margin_pct = [left / warp_size, top / warp_size,
                  (warp_size - right) / warp_size, (warp_size - bottom) / warp_size]
    margins_ok = all(0.005 < m < 0.15 for m in margin_pct)

    if inner_contour is not None:
        # Use contour corners directly
        inner_pts = inner_contour
        print("Using contour-based inner border")
    elif margins_ok:
        inner_pts = np.float32([
            [left, top], [right, top], [right, bottom], [left, bottom]
        ])
        print("Using variance-based inner border")
    else:
        # Fall back to outer with a small inset
        inset = warp_size * 0.03
        inner_pts = np.float32([
            [inset, inset], [warp_size - inset, inset],
            [warp_size - inset, warp_size - inset], [inset, warp_size - inset]
        ])
        print("Using default inset (3%)")

    # Map inner points back to original coordinates
    # inner_pts are in warped space, need to go back via inverse of M_outer
    M_inv = cv2.getPerspectiveTransform(dst, outer_corners_orig)
    inner_pts_h = np.hstack([inner_pts, np.ones((4, 1))]).T  # 3x4
    orig_pts = M_inv @ inner_pts_h  # 3x4
    orig_pts = (orig_pts[:2] / orig_pts[2]).T  # 4x2

    final_corners = orig_pts.astype(np.float32)

    if debug_dir:
        # Show inner border on warped image
        debug_warped = warped.copy()
        for i in range(4):
            pt1 = tuple(inner_pts[i].astype(int))
            pt2 = tuple(inner_pts[(i+1) % 4].astype(int))
            cv2.line(debug_warped, pt1, pt2, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "v5_02_inner_border.jpg"), debug_warped)

        # Show final corners on original
        debug_orig = orig.copy()
        labels = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(final_corners):
            cv2.circle(debug_orig, (int(x), int(y)), 15, (0, 255, 0), -1)
            cv2.putText(debug_orig, labels[i], (int(x)+20, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i in range(4):
            pt1 = tuple(final_corners[i].astype(int))
            pt2 = tuple(final_corners[(i+1) % 4].astype(int))
            cv2.line(debug_orig, pt1, pt2, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "v5_03_final_corners.jpg"), debug_orig)

        # Final warp with grid
        M_final = cv2.getPerspectiveTransform(final_corners, dst)
        warped_final = cv2.warpPerspective(orig, M_final, (warp_size, warp_size))
        sq = warp_size // 8
        for i in range(9):
            cv2.line(warped_final, (0, i*sq), (warp_size, i*sq), (0, 0, 255), 2)
            cv2.line(warped_final, (i*sq, 0), (i*sq, warp_size), (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "v5_04_final_warped.jpg"), warped_final)

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
