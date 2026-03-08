#!/usr/bin/env python3
"""
Prepare training data by warping board images and splitting into 64 labeled squares.

Uses ChessReD2K corner annotations for perspective warping (adapted from chesscog).
Each square is cropped with extra height to capture piece tops.

Output structure:
    data/squares/
        train/
            empty/ white_pawn/ ... black_king/
        val/
            (same classes)

Usage:
    python3 training/prepare_squares.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).parent / "data"
SQUARES_DIR = BASE_DIR / "squares"

SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE * 2
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2

# Chesscog-style: extend crop upward to capture piece height
MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = 0.25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)

# 13 classes
PIECE_CLASSES = [
    "empty",
    "white_pawn", "white_knight", "white_bishop",
    "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop",
    "black_rook", "black_queen", "black_king",
]

# ChessReD category names to our class names
CHESSRED_TO_CLASS = {
    "white-pawn": "white_pawn", "white-knight": "white_knight",
    "white-bishop": "white_bishop", "white-rook": "white_rook",
    "white-queen": "white_queen", "white-king": "white_king",
    "black-pawn": "black_pawn", "black-knight": "black_knight",
    "black-bishop": "black_bishop", "black-rook": "black_rook",
    "black-queen": "black_queen", "black-king": "black_king",
    "empty": "empty",
}


def sort_corner_points(points: np.ndarray) -> np.ndarray:
    """Order corners as: top-left, top-right, bottom-right, bottom-left."""
    points = points[points[:, 1].argsort()]
    points[:2] = points[:2][points[:2, 0].argsort()]
    points[2:] = points[2:][points[2:, 0].argsort()[::-1]]
    return points


def warp_board(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-warp the board to a regular grid with margin for piece tops."""
    src = sort_corner_points(corners)
    dst = np.array([
        [MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN],
        [MARGIN, BOARD_SIZE + MARGIN],
    ], dtype=np.float32)
    M, _ = cv2.findHomography(src, dst)
    return cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE))


def crop_square(warped: np.ndarray, row: int, col: int) -> np.ndarray:
    """Crop a square from the warped image with extra height for piece tops.

    row=0 is top of board (rank 8), col=0 is left (file a).
    Adapted from chesscog: extends upward more for pieces further from camera.
    """
    height_increase = MIN_HEIGHT_INCREASE + \
        (MAX_HEIGHT_INCREASE - MIN_HEIGHT_INCREASE) * ((7 - row) / 7)
    left_increase = 0 if col >= 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((3 - col) / 3)
    right_increase = 0 if col < 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((col - 4) / 3)

    x1 = int(MARGIN + SQUARE_SIZE * (col - left_increase))
    x2 = int(MARGIN + SQUARE_SIZE * (col + 1 + right_increase))
    y1 = int(MARGIN + SQUARE_SIZE * (row - height_increase))
    y2 = int(MARGIN + SQUARE_SIZE * (row + 1))

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(IMG_SIZE, x2)
    y2 = min(IMG_SIZE, y2)

    width = x2 - x1
    height = y2 - y1
    cropped = warped[y1:y2, x1:x2]

    # Mirror left-side pieces (chesscog convention for symmetry)
    if col < 4:
        cropped = cv2.flip(cropped, 1)

    # Pad to fixed output size
    result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped.dtype)
    h, w = cropped.shape[:2]
    result[OUT_HEIGHT - h:, :w] = cropped
    return result


def chess_pos_to_rowcol(pos: str):
    """Convert 'e4' to (row, col) where row=0 is rank 8 (top)."""
    file_idx = ord(pos[0]) - ord('a')  # col
    rank_idx = int(pos[1]) - 1  # 0-7
    row = 7 - rank_idx  # rank 8 = row 0
    return row, file_idx


def main():
    annotations_path = BASE_DIR / "annotations.json"
    if not annotations_path.exists():
        # Check parent
        annotations_path = BASE_DIR.parent.parent / "annotations.json"
    if not annotations_path.exists():
        print("annotations.json not found!")
        sys.exit(1)

    images_root = BASE_DIR / "chessred2k"
    if not images_root.exists():
        print(f"Images not found at {images_root}")
        sys.exit(1)

    print("Loading annotations...")
    with open(annotations_path) as f:
        data = json.load(f)

    # Build lookups
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}
    corner_map = {c["image_id"]: c["corners"] for c in data["annotations"]["corners"]}

    # Build per-image piece map: image_id -> {position: class_name}
    piece_map = {}
    for piece in data["annotations"]["pieces"]:
        img_id = piece["image_id"]
        if img_id not in piece_map:
            piece_map[img_id] = {}
        pos = piece["chessboard_position"]
        cat_name = cat_map[piece["category_id"]]
        piece_map[img_id][pos] = CHESSRED_TO_CLASS.get(cat_name, "empty")

    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in PIECE_CLASSES:
            (SQUARES_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ["train", "val", "test"]:
        split_ids = data["splits"]["chessred2k"][split]["image_ids"]
        print(f"\n{split}: {len(split_ids)} images")

        counts = {cls: 0 for cls in PIECE_CLASSES}
        errors = 0

        for idx, img_id in enumerate(split_ids):
            if img_id not in images or img_id not in corner_map:
                errors += 1
                continue

            img_info = images[img_id]
            img_path = images_root / img_info["path"]
            if not img_path.exists():
                errors += 1
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    errors += 1
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get corners and determine orientation
                corners_dict = corner_map[img_id]
                # ChessReD corner format: {"top_left": [x,y], ...}
                # We need to figure out which corner label corresponds to
                # which board corner based on the "bottom_left" being near white's a1
                corners = np.array([
                    corners_dict["top_left"],
                    corners_dict["top_right"],
                    corners_dict["bottom_right"],
                    corners_dict["bottom_left"],
                ], dtype=np.float32)

                warped = warp_board(img, corners)

                # Get pieces for this image
                pieces = piece_map.get(img_id, {})

                # Extract all 64 squares
                for row in range(8):
                    for col in range(8):
                        file_letter = chr(ord('a') + col)
                        rank_num = 8 - row
                        pos = f"{file_letter}{rank_num}"

                        cls = pieces.get(pos, "empty")
                        square_img = crop_square(warped, row, col)

                        fname = f"{img_info['file_name'].replace('.jpg', '')}_{pos}.jpg"
                        out_path = SQUARES_DIR / split / cls / fname
                        square_pil = Image.fromarray(square_img)
                        square_pil.save(out_path, quality=90)
                        counts[cls] += 1

            except Exception as e:
                print(f"  Error processing {img_id}: {e}")
                errors += 1
                continue

            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(split_ids)}")

        print(f"  Errors: {errors}")
        print(f"  Class distribution:")
        for cls in PIECE_CLASSES:
            print(f"    {cls}: {counts[cls]}")


if __name__ == "__main__":
    main()
