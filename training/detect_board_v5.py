#!/usr/bin/env python3
"""
Board detection using contour finding + checkerboard validation.

Algorithm:
1. Generate multiple binary images (Canny, adaptive, OTSU, CLAHE, etc.)
2. Find all quadrilateral contours that could be boards
3. For each candidate, warp to square and score by checkerboard pattern
4. Pick the candidate with the highest checkerboard score
5. Refine using Sobel edge grid detection to find precise 8x8 grid lines

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
            margin = sq // 6
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

    max_range = max(np.max(values) - np.min(values), 1)
    consistency = 1.0 - (light_std + dark_std) / max_range
    consistency = max(0, consistency)

    separation = abs(light_mean - dark_mean) / 255.0

    score = alternating_ratio * 0.5 + consistency * 0.25 + separation * 0.25
    return score


def get_binary_images(gray, color_image=None):
    """Generate multiple binary images for contour detection."""
    h, w = gray.shape
    kernel3 = np.ones((3, 3), np.uint8)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for lo, hi in [(30, 100), (50, 150), (20, 80)]:
        yield cv2.dilate(cv2.Canny(blurred, lo, hi), kernel3, iterations=2)

    yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 5)

    yield cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    for lo, hi in [(30, 100), (50, 150)]:
        yield cv2.dilate(cv2.Canny(enhanced, lo, hi), kernel3, iterations=2)

    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
    yield cv2.dilate(cv2.Canny(bilateral, 30, 100), kernel3, iterations=2)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel5)
    yield cv2.threshold(morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if color_image is not None:
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
        for channel in [hsv[:, :, 0], hsv[:, :, 1], lab[:, :, 1], lab[:, :, 2]]:
            blurred_ch = cv2.GaussianBlur(channel, (5, 5), 0)
            for lo, hi in [(30, 100), (20, 80)]:
                yield cv2.dilate(cv2.Canny(blurred_ch, lo, hi), kernel3, iterations=2)

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
    seen = set()
    candidates = []

    for binary in get_binary_images(gray, color_image=image):
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.05 or area > h * w * 0.98:
                continue
            peri = cv2.arcLength(cnt, True)
            found_quad = False
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
                    if max(avg_w, avg_h) < 1:
                        break
                    aspect = min(avg_w, avg_h) / max(avg_w, avg_h)

                    if aspect > 0.55:
                        key = tuple(np.round(ordered.flatten() / 20).astype(int))
                        if key not in seen:
                            seen.add(key)
                            candidates.append(ordered)
                    found_quad = True
                    break

            if not found_quad and area > h * w * 0.15:
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if max(rw, rh) > 0 and min(rw, rh) / max(rw, rh) > 0.55:
                    box = cv2.boxPoints(rect).astype(np.float32)
                    box[:, 0] = np.clip(box[:, 0], 0, w - 1)
                    box[:, 1] = np.clip(box[:, 1], 0, h - 1)
                    ordered = order_points(box)
                    key = tuple(np.round(ordered.flatten() / 20).astype(int))
                    if key not in seen:
                        seen.add(key)
                        candidates.append(ordered)

    return candidates


# --- Sobel edge-based grid detection ---

def find_grid_lines(warped_gray, warp_size):
    """
    Find 9 evenly-spaced grid lines per axis using Sobel edge energy.
    Uses autocorrelation to find grid spacing, then brute-force for origin.
    Returns (h_origin, h_spacing, v_origin, v_spacing) or None.
    """
    gray = warped_gray.astype(np.float32)
    kernel = np.ones(5) / 5

    energy_profiles = []
    blur_configs = [
        (gray, np.mean, 3),
        (cv2.GaussianBlur(gray, (7, 7), 0), np.mean, 3),
    ]
    for src, agg_fn, ksize in blur_configs:
        sobel_h = np.abs(cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=ksize))
        sobel_v = np.abs(cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=ksize))
        eh = np.convolve(agg_fn(sobel_h, axis=1), kernel, mode='same')
        ev = np.convolve(agg_fn(sobel_v, axis=0), kernel, mode='same')
        energy_profiles.append((eh, ev))

    def find_spacing_autocorr(energy, size):
        e = energy - np.mean(energy)
        norm = np.sqrt(np.sum(e ** 2))
        if norm < 1e-8:
            return None
        e = e / norm
        acf = np.correlate(e, e, mode='full')
        acf = acf[len(e):]
        min_spacing = int(size * 0.03)
        max_spacing = int(size * 0.14)
        candidates = []
        for lag in range(min_spacing, min(max_spacing + 1, len(acf) - 1)):
            if acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1]:
                candidates.append((acf[lag], lag))
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        return [c[1] for c in candidates[:3]]

    def score_grid(energy, size, origin, spacing):
        board_span = spacing * 8
        if origin < 0 or origin + board_span > size:
            return -1
        line_energies = []
        for k in range(9):
            pos = origin + k * spacing
            lo = max(0, pos - 2)
            hi = min(size, pos + 3)
            line_energies.append(np.max(energy[lo:hi]))
        line_energy = sum(line_energies)
        mid_energy = 0
        for k in range(8):
            mid = origin + k * spacing + spacing // 2
            lo = max(0, mid - 2)
            hi = min(size, mid + 3)
            mid_energy += np.mean(energy[lo:hi])

        # Consistency bonus: grid lines should all have significant energy
        if len(line_energies) > 0 and max(line_energies) > 0:
            min_line = min(line_energies)
            max_line = max(line_energies)
            consistency = min_line / max_line
        else:
            consistency = 0

        return (line_energy - 0.5 * mid_energy) * (0.85 + 0.15 * consistency)

    def find_best_grid(energy, size):
        best_score = -1
        best_origin = 0
        best_spacing = size // 8

        spacing_candidates = find_spacing_autocorr(energy, size) or []

        # Search spacings: autocorrelation candidates + range around size/8
        expected = size // 8
        focused_spacings = list(range(max(3, int(expected * 0.5)),
                                      min(size // 8 + expected, int(expected * 1.5) + 1)))
        all_spacings = list(dict.fromkeys(spacing_candidates + focused_spacings))

        for spacing in all_spacings:
            board_span = spacing * 8
            if board_span > size:
                continue
            max_origin = size - board_span
            # Coarse sweep with step=4
            for origin in range(0, max_origin + 1, 4):
                score = score_grid(energy, size, origin, spacing)
                if score > best_score:
                    best_score = score
                    best_origin = origin
                    best_spacing = spacing

        # Fine-tune around best
        for ds in range(-3, 4):
            spacing = best_spacing + ds
            if spacing < 3:
                continue
            board_span = spacing * 8
            if board_span > size:
                continue
            for origin in range(max(0, best_origin - 6),
                               min(size - board_span + 1, best_origin + 7)):
                score = score_grid(energy, size, origin, spacing)
                if score > best_score:
                    best_score = score
                    best_origin = origin
                    best_spacing = spacing

        return best_origin, best_spacing, best_score

    best_result = None
    best_total_score = -1

    for energy_h, energy_v in energy_profiles:
        h_origin, h_spacing, h_score = find_best_grid(energy_h, warp_size)
        v_origin, v_spacing, v_score = find_best_grid(energy_v, warp_size)

        if h_spacing > 0 and v_spacing > 0:
            ratio = min(h_spacing, v_spacing) / max(h_spacing, v_spacing)
            if ratio < 0.5:
                continue

        total_score = h_score + v_score
        if total_score > best_total_score:
            best_total_score = total_score
            best_result = (h_origin, h_spacing, v_origin, v_spacing)

    return best_result


def refine_to_inner_edges(orig, outer_corners):
    """
    Refine using Sobel edge energy grid fitting with rotation search.
    Tries small rotations in warped space to handle angle errors in initial quad.
    Returns (corners, checkerboard_score) or (None, 0).
    """
    warp_size = 800
    dst = np.float32([[0, 0], [warp_size, 0],
                      [warp_size, warp_size], [0, warp_size]])
    M_outer = cv2.getPerspectiveTransform(outer_corners, dst)
    warped = cv2.warpPerspective(orig, M_outer, (warp_size, warp_size))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    center_ws = (warp_size / 2, warp_size / 2)

    def _try_angle(gray, angle):
        if angle == 0:
            gray_rot = gray
        else:
            R = cv2.getRotationMatrix2D(center_ws, angle, 1.0)
            gray_rot = cv2.warpAffine(gray, R, (warp_size, warp_size),
                                       borderMode=cv2.BORDER_REPLICATE)
        result = find_grid_lines(gray_rot, warp_size)
        if result is None:
            return None, 0
        h_origin, h_spacing, v_origin, v_spacing = result
        inner_pts_rot = np.float32([
            [v_origin, h_origin],
            [v_origin + v_spacing * 8, h_origin],
            [v_origin + v_spacing * 8, h_origin + h_spacing * 8],
            [v_origin, h_origin + h_spacing * 8]])
        if angle != 0:
            R_inv = cv2.getRotationMatrix2D(center_ws, -angle, 1.0)
            pts_h = np.hstack([inner_pts_rot, np.ones((4, 1))])
            inner_pts = (R_inv @ pts_h.T).T.astype(np.float32)
        else:
            inner_pts = inner_pts_rot
        M_inv = cv2.getPerspectiveTransform(dst, outer_corners)
        inner_h = np.hstack([inner_pts, np.ones((4, 1))]).T
        orig_pts = M_inv @ inner_h
        orig_pts = (orig_pts[:2] / orig_pts[2]).T.astype(np.float32)
        score = checkerboard_score(orig, orig_pts)
        return orig_pts, score

    # First try 0° — if good enough, skip rotation search
    best_pts, best_score = _try_angle(warped_gray, 0)
    zero_score = best_score if best_score else 0

    if zero_score < 0.55:
        # Try rotations to handle angle errors in initial quad
        # Require meaningful improvement over 0° to avoid false positives
        min_improvement = 0.08
        for angle in [-5, 5, -10, 10, -15, 15]:
            pts, score = _try_angle(warped_gray, angle)
            if score > best_score and score > zero_score + min_improvement:
                best_score = score
                best_pts = pts
            if best_score >= 0.65:
                break  # Good enough

    if best_pts is None:
        return None, 0
    return best_pts, best_score


# --- Variance-based fallback refinement ---

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


def refine_shrink_and_score(orig, corners):
    """Systematically shrink the quad to maximize checkerboard score."""
    center = np.mean(corners, axis=0)
    best_pts = corners.copy()
    best_score = checkerboard_score(orig, corners)

    for shrink_pct in range(1, 16):
        factor = shrink_pct / 100.0
        shrunk = corners + factor * (center - corners)
        score = checkerboard_score(orig, shrunk.astype(np.float32))
        if score > best_score:
            best_score = score
            best_pts = shrunk.astype(np.float32)

    return best_pts, best_score


def refine_to_inner(orig, outer_corners):
    """
    Refine outer board corners to inner playing area.
    Strategy: prefer edge-based grid detection (geometrically precise).
    Fall back to variance-based or shrink-and-score methods.
    """
    outer_score = checkerboard_score(orig, outer_corners)

    # Try edge-based grid detection
    edges_pts, edges_score = refine_to_inner_edges(orig, outer_corners)
    edges_valid = False
    if edges_pts is not None and edges_score > 0.25:
        outer_area = cv2.contourArea(outer_corners.reshape(-1, 1, 2))
        edges_area = cv2.contourArea(edges_pts.reshape(-1, 1, 2))
        if outer_area > 0:
            area_ratio = edges_area / outer_area
            if 0.20 < area_ratio < 0.95:
                edges_valid = True

    if edges_valid:
        return edges_pts

    # Fallback: variance-based border detection
    warp_size = 1000
    dst = np.float32([[0, 0], [warp_size, 0],
                      [warp_size, warp_size], [0, warp_size]])
    M_outer = cv2.getPerspectiveTransform(outer_corners, dst)
    warped = cv2.warpPerspective(orig, M_outer, (warp_size, warp_size))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    left, top, right, bottom = find_inner_border(warped_gray, warp_size)
    margin_pct = [left / warp_size, top / warp_size,
                  (warp_size - right) / warp_size, (warp_size - bottom) / warp_size]
    margins_ok = all(0.005 < m < 0.15 for m in margin_pct)

    best_pts = outer_corners
    best_score = outer_score

    if margins_ok:
        inner_pts = np.float32([
            [left, top], [right, top], [right, bottom], [left, bottom]
        ])
        M_inv = cv2.getPerspectiveTransform(dst, outer_corners)
        inner_pts_h = np.hstack([inner_pts, np.ones((4, 1))]).T
        orig_pts = M_inv @ inner_pts_h
        var_pts = (orig_pts[:2] / orig_pts[2]).T.astype(np.float32)
        var_score = checkerboard_score(orig, var_pts)
        if var_score > best_score:
            best_pts = var_pts
            best_score = var_score

    # Shrink-and-score as last resort
    shrunk_pts, shrunk_score = refine_shrink_and_score(orig, outer_corners)
    if shrunk_score > best_score:
        best_pts = shrunk_pts
        best_score = shrunk_score

    return best_pts


# --- Empty square detection for board localization ---

def _find_empty_squares(image):
    """
    Detect empty squares (both dark and light) on the original image using
    L-channel thresholding. Empty squares have high solidity and low internal
    brightness variance.
    Returns (positions, median_square_side) or (None, None).
    positions is np.array of (cx, cy).
    """
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    min_side = min(h, w) * 0.015
    max_side = min(h, w) * 0.12

    best_pts = None
    best_areas = None
    best_score = 0

    for thresh in [70, 80, 90, 100, 110, 120]:
        for flag in [cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY]:
            _, binary = cv2.threshold(l_ch, thresh, 255, flag)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            pts = []
            areas = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                side = np.sqrt(area)
                if side < min_side or side > max_side:
                    continue
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if max(rw, rh) < 1:
                    continue
                if min(rw, rh) / max(rw, rh) < 0.45:
                    continue
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                if solidity < 0.85:
                    continue
                # Fast variance check using small mask within bounding rect
                bx, by, bw_r, bh_r = cv2.boundingRect(cnt)
                roi = l_ch[by:by + bh_r, bx:bx + bw_r]
                if roi.size == 0:
                    continue
                roi_mask = np.zeros((bh_r, bw_r), dtype=np.uint8)
                cnt_shifted = cnt.copy()
                cnt_shifted[:, :, 0] -= bx
                cnt_shifted[:, :, 1] -= by
                cv2.drawContours(roi_mask, [cnt_shifted], 0, 255, -1)
                pixels = roi[roi_mask > 0]
                if len(pixels) == 0 or np.std(pixels) > 18:
                    continue
                pts.append(np.array([rect[0][0], rect[0][1]], dtype=np.float32))
                areas.append(area)

            if len(pts) < 4:
                continue

            # Filter by size consistency: remove squares that differ >2x from median
            areas_arr = np.array(areas)
            med_area = np.median(areas_arr)
            keep = (areas_arr > med_area * 0.4) & (areas_arr < med_area * 2.5)
            pts_filt = [p for p, k in zip(pts, keep) if k]
            areas_filt = areas_arr[keep]

            if len(pts_filt) < 4:
                continue

            # Score: prefer many squares with consistent sizes
            size_cv = np.std(areas_filt) / np.mean(areas_filt)
            consistency = max(0, 1.0 - size_cv)
            score = len(pts_filt) * (0.5 + 0.5 * consistency)

            if score > best_score:
                best_score = score
                best_pts = pts_filt
                best_areas = areas_filt

    if best_pts is None or len(best_pts) < 4:
        return None, None

    med_sq_side = np.sqrt(np.median(best_areas))
    return np.array(best_pts), med_sq_side


def _remove_outlier_squares(pts):
    """Remove outlier squares based on nearest-neighbor distance."""
    if len(pts) < 4:
        return pts
    nn_dists = []
    for i in range(len(pts)):
        dists = np.linalg.norm(pts - pts[i], axis=1)
        dists[i] = float('inf')
        nn_dists.append(np.min(dists))
    nn_dists = np.array(nn_dists)
    med_nn = np.median(nn_dists)
    return pts[nn_dists < med_nn * 3.0]


def _find_grid_vectors(pts, sq_side):
    """
    Find two roughly-perpendicular grid vectors (u, v) from detected square centers.
    Uses median nearest-neighbor distance as the grid step, then angle clustering
    on pairwise displacement vectors normalized to that step.
    Returns (u, v) or None. u and v are one grid-step long.
    """
    n = len(pts)
    if n < 3:
        return None

    # Compute median nearest-neighbor distance = grid step
    nn_dists = []
    for i in range(n):
        min_d = float('inf')
        for j in range(n):
            if i != j:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_d:
                    min_d = d
        nn_dists.append(min_d)
    grid_step = np.median(nn_dists)

    # Collect displacement vectors normalized to one grid_step
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[j] - pts[i]
            dist = np.linalg.norm(d)
            ratio = dist / grid_step
            nearest_int = round(ratio)
            if nearest_int < 1 or nearest_int > 4:
                continue
            snap_err = abs(ratio - nearest_int) / nearest_int
            if snap_err < 0.25:
                unit_d = d / nearest_int
                candidates.append(unit_d)

    if len(candidates) < 2:
        return None

    # Cluster by angle (mod 180°) to find two perpendicular directions
    angles = np.array([np.degrees(np.arctan2(d[1], d[0])) % 180 for d in candidates])

    best_pair = None
    best_count = 0
    angle_tol = 15

    for i in range(len(candidates)):
        a1 = angles[i]
        cluster1 = [candidates[j] for j in range(len(candidates))
                     if min(abs(angles[j] - a1), 180 - abs(angles[j] - a1)) < angle_tol]

        a2_target = (a1 + 90) % 180
        cluster2 = [candidates[j] for j in range(len(candidates))
                     if min(abs(angles[j] - a2_target), 180 - abs(angles[j] - a2_target)) < angle_tol]

        count = len(cluster1) + len(cluster2)
        if count > best_count and len(cluster1) >= 1 and len(cluster2) >= 1:
            best_count = count
            u_arr = np.array(cluster1)
            ref_u = u_arr[0]
            for k in range(len(u_arr)):
                if np.dot(u_arr[k], ref_u) < 0:
                    u_arr[k] = -u_arr[k]
            v_arr = np.array(cluster2)
            ref_v = v_arr[0]
            for k in range(len(v_arr)):
                if np.dot(v_arr[k], ref_v) < 0:
                    v_arr[k] = -v_arr[k]
            best_pair = (np.mean(u_arr, axis=0), np.mean(v_arr, axis=0))

    if best_pair is None:
        return None

    return best_pair[0], best_pair[1]


def _assign_grid_coords(pts, u, v):
    """
    Assign grid coordinates to all points by projecting onto (u, v) basis,
    then iteratively refine using homography to handle perspective.

    Returns (grid_coords_dict, centroid, median_residual) or (None, None, None).
    grid_coords_dict maps point index -> (gi, gj).
    """
    n = len(pts)
    A = np.column_stack([u, v])
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None, None, None

    centroid = np.mean(pts, axis=0)

    # Initial assignment: project onto (u, v) from centroid
    coords = {}
    for i in range(n):
        ij = A_inv @ (pts[i] - centroid)
        coords[i] = (round(ij[0]), round(ij[1]))

    # Iterative homography refinement: use H^(-1) to re-project and correct
    for iteration in range(3):
        # Check we have enough unique coords
        unique_coords = set(coords.values())
        if len(unique_coords) < 4:
            return None, None, None

        # Fit homography grid -> pixel
        indices = list(coords.keys())
        grid_pts = np.array([[coords[i][0], coords[i][1]] for i in indices], dtype=np.float64)
        pixel_pts = np.array([pts[i] for i in indices], dtype=np.float64)

        if len(indices) < 4:
            return None, None, None

        try:
            if len(indices) == 4:
                H, _ = cv2.findHomography(grid_pts, pixel_pts, 0)
            else:
                H, mask = cv2.findHomography(grid_pts, pixel_pts, cv2.RANSAC,
                                              ransacReprojThreshold=np.linalg.norm(u) * 0.3)
        except cv2.error:
            break
        if H is None:
            break

        # Use inverse homography to refine grid coords
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            break

        changed = False
        for i in range(n):
            pt_h = np.array([pts[i][0], pts[i][1], 1.0], dtype=np.float64)
            proj = H_inv @ pt_h
            if abs(proj[2]) < 1e-10:
                continue
            gi_new = round(proj[0] / proj[2])
            gj_new = round(proj[1] / proj[2])
            if coords[i] != (gi_new, gj_new):
                coords[i] = (gi_new, gj_new)
                changed = True

        if not changed:
            break

    # Validate: check homography reprojection error (not constant-vector error)
    # Re-fit final homography
    indices = list(coords.keys())
    grid_pts = np.array([[coords[i][0], coords[i][1]] for i in indices], dtype=np.float64)
    pixel_pts = np.array([pts[i] for i in indices], dtype=np.float64)

    unique_coords = set(coords.values())
    if len(unique_coords) < 4:
        return None, None, None

    try:
        if len(indices) == 4:
            H_final, _ = cv2.findHomography(grid_pts, pixel_pts, 0)
        else:
            H_final, _ = cv2.findHomography(grid_pts, pixel_pts, cv2.RANSAC,
                                             ransacReprojThreshold=np.linalg.norm(u) * 0.3)
    except cv2.error:
        return None, None, None
    if H_final is None:
        return None, None, None

    residuals = []
    for i in range(n):
        gi, gj = coords[i]
        pt_h = np.array([gi, gj, 1.0], dtype=np.float64)
        proj = H_final @ pt_h
        if abs(proj[2]) > 1e-10:
            predicted = np.array([proj[0] / proj[2], proj[1] / proj[2]])
            residuals.append(np.linalg.norm(predicted - pts[i]))
    med_residual = np.median(residuals) if residuals else float('inf')

    step = np.linalg.norm(u)
    if med_residual > step * 0.3:
        return None, None, None

    return coords, centroid, med_residual


def _fit_homography(pts, coords):
    """
    Fit a homography from grid coordinates to pixel coordinates.
    coords: dict mapping point index -> (gi, gj)
    Returns (H, inlier_mask) or (None, None).
    """
    indices = list(coords.keys())
    n = len(indices)
    if n < 4:
        return None, None

    grid_pts = np.array([[coords[i][0], coords[i][1]] for i in indices], dtype=np.float64)
    pixel_pts = np.array([pts[i] for i in indices], dtype=np.float64)

    if n == 4:
        H, _ = cv2.findHomography(grid_pts, pixel_pts, 0)
        return H, np.ones(n, dtype=bool)
    else:
        avg_dist = np.mean([np.linalg.norm(pts[indices[i]] - pts[indices[j]])
                           for i in range(min(n, 5)) for j in range(i+1, min(n, 5))])
        thresh = max(avg_dist * 0.1, 5.0)
        H, mask = cv2.findHomography(grid_pts, pixel_pts, cv2.RANSAC,
                                      ransacReprojThreshold=thresh)
        if H is None:
            return None, None
        return H, mask.ravel().astype(bool) if mask is not None else np.ones(n, dtype=bool)


def _project_grid_to_pixel(H, gi, gj):
    """Project grid coordinate (gi, gj) to pixel coordinate using homography H."""
    pt = np.array([gi, gj, 1.0], dtype=np.float64)
    proj = H @ pt
    if abs(proj[2]) < 1e-10:
        return None
    return np.array([proj[0] / proj[2], proj[1] / proj[2]], dtype=np.float32)


def _fill_grid(image, pts, coords, H, sq_side):
    """
    Grow the detected square set by checking predicted grid positions.
    Validates each candidate position with:
    - Brightness consistency with detected squares
    - Low patch variance (uniform square interior)
    - Edge check: brightness transition at border with neighboring square
    - Limited extent: max 2 steps beyond initial detected range

    Returns expanded (pts_array, coords_dict) with new validated positions added.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    img_h, img_w = l_ch.shape

    # Measure brightness of each detected square
    brightnesses = {}
    for idx in coords:
        gi, gj = coords[idx]
        x, y = int(pts[idx][0]), int(pts[idx][1])
        half = max(3, int(sq_side * 0.15))
        y1, y2 = max(0, y - half), min(img_h, y + half + 1)
        x1, x2 = max(0, x - half), min(img_w, x + half + 1)
        if y2 > y1 and x2 > x1:
            brightnesses[(gi, gj)] = np.mean(l_ch[y1:y2, x1:x2])

    if not brightnesses:
        return pts, coords

    # Determine brightness acceptance range
    bright_vals = list(brightnesses.values())
    mean_bright = np.mean(bright_vals)
    std_bright = np.std(bright_vals)

    if std_bright < 15:
        accept_min = mean_bright - 30
        accept_max = mean_bright + 30
    else:
        accept_min = min(bright_vals) - 20
        accept_max = max(bright_vals) + 20

    # Limit fill extent: max 2 grid steps beyond initial detected range
    initial_coords = set(coords.values())
    gi_vals = [c[0] for c in initial_coords]
    gj_vals = [c[1] for c in initial_coords]
    gi_min, gi_max = min(gi_vals) - 2, max(gi_vals) + 2
    gj_min, gj_max = min(gj_vals) - 2, max(gj_vals) + 2

    # Build set of occupied grid positions
    filled_positions = {}
    for idx in coords:
        filled_positions[coords[idx]] = pts[idx].copy()

    # BFS outward from detected squares
    frontier = set(coords.values())
    for iteration in range(4):
        new_frontier = set()
        for (gi, gj) in frontier:
            for dgi, dgj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ngi, ngj = gi + dgi, gj + dgj
                if (ngi, ngj) in filled_positions:
                    continue
                # Extent limit
                if ngi < gi_min or ngi > gi_max or ngj < gj_min or ngj > gj_max:
                    continue

                pos = _project_grid_to_pixel(H, ngi, ngj)
                if pos is None:
                    continue
                x, y = int(pos[0]), int(pos[1])
                margin = int(sq_side * 0.4)
                if x < margin or x >= img_w - margin or y < margin or y >= img_h - margin:
                    continue

                # Check center brightness
                half = max(3, int(sq_side * 0.15))
                y1, y2 = max(0, y - half), min(img_h, y + half + 1)
                x1, x2 = max(0, x - half), min(img_w, x + half + 1)
                patch = l_ch[y1:y2, x1:x2].astype(float)
                patch_mean = np.mean(patch)
                patch_var = np.var(patch)

                # Reject: noisy patch (not a uniform square)
                if patch_var > 800:
                    continue

                # Reject: brightness outside expected range
                if patch_mean < accept_min or patch_mean > accept_max:
                    continue

                # Edge check: verify brightness transition at border with neighbor
                # Sample midpoint between this position and the neighbor we came from
                neighbor_pos = filled_positions.get((gi, gj))
                if neighbor_pos is not None:
                    mid_x = int((pos[0] + neighbor_pos[0]) / 2)
                    mid_y = int((pos[1] + neighbor_pos[1]) / 2)
                    edge_half = max(2, int(sq_side * 0.08))
                    ey1, ey2 = max(0, mid_y - edge_half), min(img_h, mid_y + edge_half + 1)
                    ex1, ex2 = max(0, mid_x - edge_half), min(img_w, mid_x + edge_half + 1)
                    edge_patch = l_ch[ey1:ey2, ex1:ex2].astype(float)
                    edge_var = np.var(edge_patch) if edge_patch.size > 0 else 0

                    # At a grid line, expect higher variance (brightness transition)
                    # If edge_var is very low, the two positions are on the same uniform surface
                    # (e.g., table) — reject
                    neighbor_bright = brightnesses.get((gi, gj),
                        np.mean(l_ch[max(0, int(neighbor_pos[1])-half):min(img_h, int(neighbor_pos[1])+half+1),
                                     max(0, int(neighbor_pos[0])-half):min(img_w, int(neighbor_pos[0])+half+1)].astype(float)))
                    bright_diff = abs(patch_mean - neighbor_bright)
                    # If neighbor and candidate have similar brightness but the edge
                    # between them also has similar brightness → likely same surface
                    if bright_diff < 10 and edge_var < 100:
                        continue

                filled_positions[(ngi, ngj)] = np.array([pos[0], pos[1]], dtype=np.float32)
                brightnesses[(ngi, ngj)] = patch_mean
                new_frontier.add((ngi, ngj))

        if not new_frontier:
            break
        frontier = new_frontier

    # Build expanded arrays
    new_pts = list(pts)
    new_coords = dict(coords)
    for (gi, gj), pos in filled_positions.items():
        if (gi, gj) not in initial_coords:
            new_idx = len(new_pts)
            new_pts.append(pos)
            new_coords[new_idx] = (gi, gj)

    return np.array(new_pts, dtype=np.float32), new_coords


def _grid_based_candidates(image, pts, sq_side, img_shape):
    """
    Grid-based board detection:
    1. Find grid vectors (u, v) — one grid-step each
    2. Assign raw grid coordinates
    3. Try 3 interpretations of how grid maps to chess coords:
       - direct: grid = chess (both colors detected)
       - diagonal: chess = (gi+gj, gi-gj) (single-color diagonal neighbors)
       - double: chess = (2*gi, 2*gj) (single-color same-row neighbors)
    4. For each interpretation, fit homography in chess space,
       search 8×8 placements, score by checkerboard + size match + coverage
    """
    h, w = img_shape[:2]
    n = len(pts)
    if n < 4 or sq_side <= 0:
        return []

    # Step 1: Find grid vectors
    result = _find_grid_vectors(pts, sq_side)
    if result is None:
        return []
    u, v = result

    # Step 2: Assign grid coordinates in raw basis
    coords, centroid, med_residual = _assign_grid_coords(pts, u, v)
    if coords is None:
        return []

    # Step 2.5: Fill grid — grow detected set using brightness validation
    H_init, _ = _fit_homography(pts, coords)
    if H_init is not None:
        pts, coords = _fill_grid(image, pts, coords, H_init, sq_side)

    indices = list(coords.keys())
    expected_board_px = 8 * sq_side

    # Step 3: Try all interpretations
    transforms = [
        ("direct", lambda gi, gj: (gi, gj)),
        ("diagonal", lambda gi, gj: (gi + gj, gi - gj)),
        ("double", lambda gi, gj: (2 * gi, 2 * gj)),
    ]

    candidates = []
    seen = set()

    for t_name, t_fn in transforms:
        chess = {idx: t_fn(*coords[idx]) for idx in indices}

        # Fit homography chess → pixel
        chess_pts = np.array([[chess[i][0], chess[i][1]] for i in indices], dtype=np.float64)
        pixel_pts = np.array([pts[i] for i in indices], dtype=np.float64)

        unique_chess = set(map(tuple, chess_pts.tolist()))
        if len(unique_chess) < 4:
            continue

        try:
            if len(indices) == 4:
                H_c, _ = cv2.findHomography(chess_pts, pixel_pts, 0)
            else:
                H_c, _ = cv2.findHomography(chess_pts, pixel_pts, cv2.RANSAC,
                                             ransacReprojThreshold=sq_side * 0.3)
        except cv2.error:
            continue
        if H_c is None:
            continue

        # Validate homography
        max_err = 0
        for idx in indices:
            ci, cj = chess[idx]
            proj = _project_grid_to_pixel(H_c, ci, cj)
            if proj is not None:
                err = np.linalg.norm(proj - pts[idx])
                max_err = max(max_err, err)
        if max_err > sq_side * 0.5:
            continue

        chess_assigned = [chess[i] for i in indices]
        ci_vals = [c[0] for c in chess_assigned]
        cj_vals = [c[1] for c in chess_assigned]
        ci_min, ci_max = min(ci_vals), max(ci_vals)
        cj_min, cj_max = min(cj_vals), max(cj_vals)
        occupied = set(chess_assigned)

        for bs in [7, 8, 9]:
            for ci_off in range(ci_min - bs + 1, ci_max + 1):
                for cj_off in range(cj_min - bs + 1, cj_max + 1):
                    covered = 0
                    for gc in occupied:
                        if ci_off <= gc[0] < ci_off + bs and cj_off <= gc[1] < cj_off + bs:
                            covered += 1

                    if covered < max(2, len(occupied) * 0.4):
                        continue

                    c_tl = _project_grid_to_pixel(H_c, ci_off, cj_off)
                    c_tr = _project_grid_to_pixel(H_c, ci_off + bs, cj_off)
                    c_br = _project_grid_to_pixel(H_c, ci_off + bs, cj_off + bs)
                    c_bl = _project_grid_to_pixel(H_c, ci_off, cj_off + bs)

                    if any(c is None for c in [c_tl, c_tr, c_br, c_bl]):
                        continue

                    corners = np.array([c_tl, c_tr, c_br, c_bl], dtype=np.float32)

                    if (corners[:, 0].min() < -w * 0.3 or corners[:, 0].max() > w * 1.3 or
                        corners[:, 1].min() < -h * 0.3 or corners[:, 1].max() > h * 1.3):
                        continue

                    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
                    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
                    ordered = order_points(corners)

                    key = tuple(np.round(ordered.flatten() / 30).astype(int))
                    if key in seen:
                        continue
                    seen.add(key)

                    # Size match penalty
                    side_lens = [np.linalg.norm(ordered[i] - ordered[(i+1) % 4]) for i in range(4)]
                    actual_board_px = np.mean(side_lens)
                    size_ratio = actual_board_px / expected_board_px if expected_board_px > 0 else 1.0
                    size_score = np.exp(-0.5 * ((size_ratio - 1.0) / 0.25) ** 2)

                    cb_score = checkerboard_score(image, ordered)
                    coverage_ratio = covered / len(occupied) if len(occupied) > 0 else 0
                    combined = cb_score * (0.3 + 0.3 * coverage_ratio + 0.4 * size_score)

                    candidates.append((combined, ordered))

    candidates.sort(key=lambda x: -x[0])
    return candidates


def find_board_from_empty_squares(image):
    """
    Locate the chess board using empty square detection + grid-based fitting.
    1. Find empty squares (high solidity, low variance) at multiple thresholds
    2. Remove outliers by nearest-neighbor distance
    3. Build neighbor graph, BFS grid coords, fit homography
    4. Enumerate 8×8 placements scored by checkerboard + coverage

    Returns (best_corners, board_center, board_radius) or (None, None, None).
    """
    pts, sq_side = _find_empty_squares(image)
    if pts is None:
        return None, None, None

    filtered = _remove_outlier_squares(pts)
    if len(filtered) < 4:
        return None, None, None

    h, w = image.shape[:2]
    center = np.mean(filtered, axis=0)

    # Try grid-based approach (homography)
    scored = _grid_based_candidates(image, filtered, sq_side, image.shape)

    best_corners = None
    best_score = 0
    if scored and scored[0][0] > 0.15:
        best_score = scored[0][0]
        best_corners = scored[0][1]

    # Board radius from square size
    if sq_side is not None:
        radius = sq_side * 8 * 0.7
    else:
        extent = max(filtered[:, 0].max() - filtered[:, 0].min(),
                     filtered[:, 1].max() - filtered[:, 1].min())
        radius = extent * 0.8

    return best_corners, center, radius


def detect_board_corners(image_path, debug_dir=None):
    """
    Detect chess board corners.
    1. Find empty squares (dark/light) to locate the board region
    2. Generate rough quads from empty square positions
    3. If needed, run contour detection filtered to the detected region
    4. Refine all candidates with Sobel edge grid detection
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read {image_path}")

    orig = image.copy()
    h, w = image.shape[:2]

    # --- Step 1: Empty square detection (primary) ---
    ds_corners, board_center, board_radius = find_board_from_empty_squares(orig)
    best_raw = None
    best_raw_score = 0

    if ds_corners is not None:
        ds_score = checkerboard_score(orig, ds_corners)
        best_raw = ds_corners
        best_raw_score = ds_score

    # --- Step 2: Contour candidates filtered by empty-square region ---
    max_dim = 1500
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        resized = image.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    candidates = find_candidates(gray, resized)

    if candidates:
        proc_h, proc_w = gray.shape
        if board_center is not None:
            ref_center = board_center * scale
            ref_radius = board_radius * scale if board_radius else None
        else:
            ref_center = np.array([proc_w / 2, proc_h / 2])
            ref_radius = None

        img_diag = np.sqrt(proc_w ** 2 + proc_h ** 2)

        for corners in candidates:
            corners_orig = corners / scale
            cb_score = checkerboard_score(orig, corners_orig)

            quad_center = np.mean(corners, axis=0)
            center_dist = np.linalg.norm(quad_center - ref_center) / img_diag
            center_weight = max(0.3, np.exp(-3.0 * center_dist ** 2))

            if ref_radius is not None:
                if np.linalg.norm(quad_center - ref_center) > ref_radius * 1.5:
                    continue

            quad_area = cv2.contourArea(corners.reshape(-1, 1, 2))
            img_area = proc_w * proc_h
            area_ratio = quad_area / img_area
            if area_ratio > 0.85:
                size_weight = 0.2
            elif area_ratio > 0.7:
                size_weight = 0.5
            else:
                size_weight = 1.0

            score = cb_score * center_weight * size_weight
            if score > best_raw_score:
                best_raw_score = score
                best_raw = corners_orig

    if best_raw is None:
        return None

    # --- Step 3: Collect top candidates and refine each ---
    # Gather all scored candidates (dark squares + contours)
    all_scored = []
    if ds_corners is not None:
        all_scored.append((checkerboard_score(orig, ds_corners), ds_corners))
    if candidates:
        for corners in candidates:
            corners_orig = corners / scale
            cb_score = checkerboard_score(orig, corners_orig)
            quad_center = np.mean(corners, axis=0)
            if board_center is not None:
                ref_c = board_center * scale
                ref_r = board_radius * scale if board_radius else None
                if ref_r and np.linalg.norm(quad_center - ref_c) > ref_r * 1.5:
                    continue
            all_scored.append((cb_score, corners_orig))

    all_scored.sort(key=lambda x: -x[0])

    # Refine top-5 and pick the best refined result
    best_final = best_raw
    best_final_score = checkerboard_score(orig, best_raw)
    for raw_score, raw_corners in all_scored[:5]:
        if raw_score < all_scored[0][0] * 0.3:
            break
        refined = refine_to_inner(orig, raw_corners)
        ref_score = checkerboard_score(orig, refined)
        if ref_score > best_final_score:
            best_final_score = ref_score
            best_final = refined

    # Iterative refinement: expand best result slightly and re-refine
    # This helps when the first pass partially corrected the angle
    if best_final_score < 0.65:
        center_pt = np.mean(best_final, axis=0)
        expanded = best_final + 0.12 * (best_final - center_pt)
        expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
        re_refined = refine_to_inner(orig, expanded.astype(np.float32))
        re_score = checkerboard_score(orig, re_refined)
        # Validate: re-refined corners should be near original (prevent jumping to wrong region)
        corner_shift = np.mean([np.linalg.norm(re_refined[j] - best_final[j]) for j in range(4)])
        board_side = np.mean([np.linalg.norm(best_final[1] - best_final[0]),
                              np.linalg.norm(best_final[2] - best_final[1])])
        if re_score > best_final_score and corner_shift < board_side * 0.4:
            best_final_score = re_score
            best_final = re_refined

    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        debug_orig = orig.copy()
        labels = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(best_final):
            cv2.circle(debug_orig, (int(x), int(y)), 15, (0, 255, 0), -1)
            cv2.putText(debug_orig, labels[i], (int(x) + 20, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i in range(4):
            pt1 = tuple(best_final[i].astype(int))
            pt2 = tuple(best_final[(i + 1) % 4].astype(int))
            cv2.line(debug_orig, pt1, pt2, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "v5_corners.jpg"), debug_orig)

        warp_size = 800
        dst = np.float32([[0, 0], [warp_size, 0],
                          [warp_size, warp_size], [0, warp_size]])
        M = cv2.getPerspectiveTransform(best_final, dst)
        warped = cv2.warpPerspective(orig, M, (warp_size, warp_size))
        sq = warp_size // 8
        for i in range(9):
            cv2.line(warped, (0, i * sq), (warp_size, i * sq), (0, 0, 255), 2)
            cv2.line(warped, (i * sq, 0), (i * sq, warp_size), (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "v5_warped.jpg"), warped)

    return best_final


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
