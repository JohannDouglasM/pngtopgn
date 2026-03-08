#!/usr/bin/env python3
"""
Add user's chess board images to training data with manual corner annotations.

For each image, specify corners (TL, TR, BR, BL) as pixel coordinates,
plus the FEN position. This generates:
1. Corner annotations for the corner detector
2. Per-square crops for the square classifier

Usage:
    python3 training/add_user_images.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

BASE_DIR = Path(__file__).parent
SQUARES_DIR = BASE_DIR / "data" / "squares"

# Same constants as prepare_squares.py
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE * 2
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2
MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = 0.25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)

FEN_TO_CLASS = {
    'P': 'white_pawn', 'N': 'white_knight', 'B': 'white_bishop',
    'R': 'white_rook', 'Q': 'white_queen', 'K': 'white_king',
    'p': 'black_pawn', 'n': 'black_knight', 'b': 'black_bishop',
    'r': 'black_rook', 'q': 'black_queen', 'k': 'black_king',
}


def sort_corner_points(points):
    points = points[points[:, 1].argsort()]
    points[:2] = points[:2][points[:2, 0].argsort()]
    points[2:] = points[2:][points[2:, 0].argsort()[::-1]]
    return points


def warp_board(img, corners):
    src = sort_corner_points(corners)
    dst = np.array([
        [MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN],
        [MARGIN, BOARD_SIZE + MARGIN],
    ], dtype=np.float32)
    M, _ = cv2.findHomography(src, dst)
    return cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE))


def crop_square(warped, row, col):
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
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(IMG_SIZE, x2), min(IMG_SIZE, y2)
    cropped = warped[y1:y2, x1:x2]
    if col < 4:
        cropped = cv2.flip(cropped, 1)
    result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped.dtype)
    h, w = cropped.shape[:2]
    result[OUT_HEIGHT - h:, :w] = cropped
    return result


def fen_to_board(fen):
    """Convert FEN to dict of {(row, col): class_name}."""
    board = {}
    ranks = fen.split("/")
    for row, rank_str in enumerate(ranks):
        col = 0
        for ch in rank_str:
            if ch.isdigit():
                col += int(ch)
            else:
                cls = FEN_TO_CLASS.get(ch, 'empty')
                board[(row, col)] = cls
                col += 1
    return board


def process_image(image_path, corners, fen, image_id, split="train"):
    """Process a single user image and add squares to training data."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Error reading {image_path}")
        return 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    corners_np = np.array(corners, dtype=np.float32)
    warped = warp_board(img, corners_np)

    board = fen_to_board(fen)
    count = 0

    for row in range(8):
        for col in range(8):
            cls = board.get((row, col), 'empty')
            square_img = crop_square(warped, row, col)

            file_letter = chr(ord('a') + col)
            rank_num = 8 - row
            pos = f"{file_letter}{rank_num}"

            fname = f"user_{image_id}_{pos}.jpg"
            out_dir = SQUARES_DIR / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / fname

            square_pil = Image.fromarray(square_img)
            square_pil.save(out_path, quality=90)
            count += 1

    return count


def main():
    # User's images with manual corner annotations and FEN
    # Corners are [TL, TR, BR, BL] in pixel coordinates
    # Determined by visual inspection of each image

    user_images = [
        {
            "path": "assets/rnbqkbnr:pppppppp:8:8:8:8:PPPPPPPP:RNBQKBNR.jpeg",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            "corners": [[155, 345], [1380, 165], [1465, 1370], [210, 1445]],
        },
        {
            "path": "assets/r1b1kb1r:pp2pppp:1nnq4:8:2BP4:2N5:PP3PPP:R1BQK1NR.jpeg",
            "fen": "r1b1kb1r/pp2pppp/1nnq4/8/2BP4/2N5/PP3PPP/R1BQK1NR",
            "corners": [[100, 220], [1380, 100], [1460, 1440], [120, 1460]],
        },
        {
            "path": "assets/r1b2rk1:pp2pp1p:6p1:3q4:3b4:8:PPP1BPPP:R2Q1RK1.jpeg",
            "fen": "r1b2rk1/pp2pp1p/6p1/3q4/3b4/8/PPP1BPPP/R2Q1RK1",
            "corners": [[115, 245], [1400, 125], [1470, 1425], [120, 1460]],
        },
        {
            "path": "assets/r1bq1rk1:2p1ppbp:2np1np1:1B6:3PP3:2N1BP2:PP1Q2PP:2KR2NR.jpeg",
            "fen": "r1bq1rk1/2p1ppbp/2np1np1/1B6/3PP3/2N1BP2/PP1Q2PP/2KR2NR",
            "corners": [[105, 280], [1405, 130], [1470, 1440], [115, 1485]],
        },
        {
            "path": "assets/r1bq1rk1:2p1ppbp:2np1np1:1B6:3PP3:2N1BP2:PP1Q2PP:2KR2NR(2).jpeg",
            "fen": "r1bq1rk1/2p1ppbp/2np1np1/1B6/3PP3/2N1BP2/PP1Q2PP/2KR2NR",
            "corners": [[105, 280], [1405, 130], [1470, 1440], [115, 1485]],
        },
        {
            "path": "assets/test_board.jpeg",
            "fen": "r1bq1rk1/pp1pppbp/5np1/8/2PQ4/2N3P1/PP2PPBP/R1B2RK1",
            "corners": [[115, 385], [1375, 200], [1435, 1400], [180, 1490]],
        },
    ]

    # Also save corner annotations for the corner detector
    corner_annotations = []

    total = 0
    for i, entry in enumerate(user_images):
        path = Path(entry["path"])
        if not path.exists():
            print(f"Skipping {path} (not found)")
            continue

        img = Image.open(path)
        w, h = img.size

        print(f"\n[{i+1}] {path.name} ({w}x{h})")
        print(f"  FEN: {entry['fen']}")
        print(f"  Corners: {entry['corners']}")

        count = process_image(path, entry["corners"], entry["fen"], f"img{i}", split="train")
        total += count
        print(f"  Generated {count} square crops")

        # Save corner annotation (normalized)
        corners = entry["corners"]
        corner_annotations.append({
            "image_path": str(path),
            "width": w,
            "height": h,
            "corners_normalized": {
                "top_left": [corners[0][0] / w, corners[0][1] / h],
                "top_right": [corners[1][0] / w, corners[1][1] / h],
                "bottom_right": [corners[2][0] / w, corners[2][1] / h],
                "bottom_left": [corners[3][0] / w, corners[3][1] / h],
            },
            "fen": entry["fen"],
        })

    # Save corner annotations
    ann_path = BASE_DIR / "data" / "user_corners.json"
    with open(ann_path, "w") as f:
        json.dump(corner_annotations, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Total squares generated: {total}")
    print(f"Corner annotations saved to: {ann_path}")
    print(f"\nNow retrain with:")
    print(f"  python3 training/train_squares.py --epochs 30 --batch-size 32 --lr 0.001")


if __name__ == "__main__":
    main()
