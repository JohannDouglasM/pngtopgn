#!/usr/bin/env python3
"""
Prepare training data from all sources:
1. ChessReD2K (already prepared as squares + corner annotations)
2. Kaggle synthetic chess board images (corners + per-square pieces)
3. User's own board images (manual corners + FEN)

Generates:
- Square crops in training/data/squares/{train,val}/{class_name}/ for the square classifier
- Corner annotations integrated into annotations.json for the corner detector

Usage:
    python3 training/prepare_all_data.py
"""

import json
import sys
import os
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

BASE_DIR = Path(__file__).parent
SQUARES_DIR = BASE_DIR / "data" / "squares"
KAGGLE_DIR = BASE_DIR / "data" / "kaggle" / "data"

# Warping constants (must match prepare_squares.py / test_pipeline.py)
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE  # 400
IMG_SIZE = BOARD_SIZE * 2      # 800
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2  # 200

MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = 0.25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)   # 100
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)  # 200

# Kaggle piece name → our class name
KAGGLE_TO_CLASS = {
    'bishop_b': 'black_bishop', 'king_b': 'black_king', 'knight_b': 'black_knight',
    'pawn_b': 'black_pawn', 'queen_b': 'black_queen', 'rook_b': 'black_rook',
    'bishop_w': 'white_bishop', 'king_w': 'white_king', 'knight_w': 'white_knight',
    'pawn_w': 'white_pawn', 'queen_w': 'white_queen', 'rook_w': 'white_rook',
}

# FEN char → our class name
FEN_TO_CLASS = {
    'P': 'white_pawn', 'N': 'white_knight', 'B': 'white_bishop',
    'R': 'white_rook', 'Q': 'white_queen', 'K': 'white_king',
    'p': 'black_pawn', 'n': 'black_knight', 'b': 'black_bishop',
    'r': 'black_rook', 'q': 'black_queen', 'k': 'black_king',
}


def crop_square(warped, row, col):
    """Crop a single square from warped board image (chesscog style)."""
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


def warp_board(img_rgb, corners_px):
    """Warp board using 4 corner points [TL, TR, BR, BL] in pixel coords."""
    src = np.array(corners_px, dtype=np.float32)
    dst = np.array([
        [MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN],
        [MARGIN, BOARD_SIZE + MARGIN],
    ], dtype=np.float32)
    M, _ = cv2.findHomography(src, dst)
    return cv2.warpPerspective(img_rgb, M, (IMG_SIZE, IMG_SIZE))


def save_square(square_img, cls_name, split, prefix, pos):
    """Save a square crop to the appropriate class directory."""
    out_dir = SQUARES_DIR / split / cls_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{pos}.jpg"
    Image.fromarray(square_img).save(out_dir / fname, quality=90)


def process_kaggle():
    """Process Kaggle synthetic chess board images."""
    if not KAGGLE_DIR.exists():
        print("Kaggle data not found, skipping")
        return 0, []

    json_files = sorted(KAGGLE_DIR.glob("*.json"))
    json_files = [f for f in json_files if f.name != "config.json"]
    print(f"\nProcessing {len(json_files)} Kaggle images...")

    # 90/10 train/val split
    random.seed(42)
    indices = list(range(len(json_files)))
    random.shuffle(indices)
    val_count = max(1, len(indices) // 10)
    val_set = set(indices[:val_count])

    total_squares = 0
    corner_annotations = []

    for i, jf in enumerate(json_files):
        img_id = jf.stem
        img_path = KAGGLE_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        with open(jf) as f:
            ann = json.load(f)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Kaggle corners are [A1, H1, A8, H8] normalized
        # We need [TL, TR, BR, BL] for our standard orientation:
        #   TL=A8 corner, TR=H8 corner, BR=H1 corner, BL=A1 corner
        # This puts rank 8 at top, rank 1 at bottom, file A left, file H right
        raw_corners = ann['corners']  # [A1, H1, A8, H8]
        # Convert to pixel coords
        a1 = [raw_corners[0][0] * w, raw_corners[0][1] * h]
        h1 = [raw_corners[1][0] * w, raw_corners[1][1] * h]
        a8 = [raw_corners[2][0] * w, raw_corners[2][1] * h]
        h8 = [raw_corners[3][0] * w, raw_corners[3][1] * h]

        # Standard orientation: TL=A8, TR=H8, BR=H1, BL=A1
        corners_px = [a8, h8, h1, a1]

        split = "val" if i in val_set else "train"

        # Warp and extract squares
        warped = warp_board(img, corners_px)

        # Build piece map: (row, col) → class_name
        # row 0 = rank 8, col 0 = file A
        piece_map = {}
        for pos_str, piece_name in ann['config'].items():
            file_letter = pos_str[0]  # A-H
            rank_num = int(pos_str[1])  # 1-8
            col = ord(file_letter) - ord('A')
            row = 8 - rank_num
            cls_name = KAGGLE_TO_CLASS.get(piece_name, 'empty')
            piece_map[(row, col)] = cls_name

        for row in range(8):
            for col in range(8):
                cls_name = piece_map.get((row, col), 'empty')
                square_img = crop_square(warped, row, col)
                file_letter = chr(ord('a') + col)
                rank_num = 8 - row
                pos = f"{file_letter}{rank_num}"
                save_square(square_img, cls_name, split, f"kaggle_{img_id}", pos)
                total_squares += 1

        # Save corner annotation (normalized, in our TL/TR/BR/BL order)
        corner_annotations.append({
            "source": "kaggle",
            "image_path": str(img_path),
            "width": w, "height": h,
            "split": split,
            "corners_pixel": {
                "top_left": a8, "top_right": h8,
                "bottom_right": h1, "bottom_left": a1,
            },
        })

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(json_files)} images processed...")

    print(f"  Kaggle: {total_squares} squares from {len(corner_annotations)} images")
    return total_squares, corner_annotations


def process_user_images():
    """Process user's own chess board images with manual annotations."""
    project_root = BASE_DIR.parent

    user_images = [
        {
            "path": project_root / "assets/rnbqkbnr:pppppppp:8:8:8:8:PPPPPPPP:RNBQKBNR.jpeg",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            "corners": [[155, 345], [1380, 165], [1465, 1370], [210, 1445]],
        },
        {
            "path": project_root / "assets/r1b1kb1r:pp2pppp:1nnq4:8:2BP4:2N5:PP3PPP:R1BQK1NR.jpeg",
            "fen": "r1b1kb1r/pp2pppp/1nnq4/8/2BP4/2N5/PP3PPP/R1BQK1NR",
            "corners": [[100, 220], [1380, 100], [1460, 1440], [120, 1460]],
        },
        {
            "path": project_root / "assets/r1b2rk1:pp2pp1p:6p1:3q4:3b4:8:PPP1BPPP:R2Q1RK1.jpeg",
            "fen": "r1b2rk1/pp2pp1p/6p1/3q4/3b4/8/PPP1BPPP/R2Q1RK1",
            "corners": [[115, 245], [1400, 125], [1470, 1425], [120, 1460]],
        },
        {
            "path": project_root / "assets/r1bq1rk1:2p1ppbp:2np1np1:1B6:3PP3:2N1BP2:PP1Q2PP:2KR2NR.jpeg",
            "fen": "r1bq1rk1/2p1ppbp/2np1np1/1B6/3PP3/2N1BP2/PP1Q2PP/2KR2NR",
            "corners": [[105, 280], [1405, 130], [1470, 1440], [115, 1485]],
        },
        {
            "path": project_root / "assets/r1bq1rk1:2p1ppbp:2np1np1:1B6:3PP3:2N1BP2:PP1Q2PP:2KR2NR(2).jpeg",
            "fen": "r1bq1rk1/2p1ppbp/2np1np1/1B6/3PP3/2N1BP2/PP1Q2PP/2KR2NR",
            "corners": [[105, 280], [1405, 130], [1470, 1440], [115, 1485]],
        },
        {
            "path": project_root / "assets/test_board.jpeg",
            "fen": "r1bq1rk1/pp1pppbp/5np1/8/2PQ4/2N3P1/PP2PPBP/R1B2RK1",
            "corners": [[115, 385], [1375, 200], [1435, 1400], [180, 1490]],
        },
    ]

    total_squares = 0
    corner_annotations = []

    print(f"\nProcessing {len(user_images)} user images...")

    for i, entry in enumerate(user_images):
        path = Path(entry["path"])
        if not path.exists():
            print(f"  Skipping {path.name} (not found)")
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Corners are [TL, TR, BR, BL] already
        corners_px = entry["corners"]
        warped = warp_board(img, corners_px)

        # Parse FEN to piece map
        piece_map = {}
        ranks = entry["fen"].split("/")
        for row, rank_str in enumerate(ranks):
            col = 0
            for ch in rank_str:
                if ch.isdigit():
                    col += int(ch)
                else:
                    piece_map[(row, col)] = FEN_TO_CLASS.get(ch, 'empty')
                    col += 1

        # Add to both train and val (small dataset, put in train with 1 copy in val)
        for split in ["train", "val"]:
            for row in range(8):
                for col in range(8):
                    cls_name = piece_map.get((row, col), 'empty')
                    square_img = crop_square(warped, row, col)
                    file_letter = chr(ord('a') + col)
                    rank_num = 8 - row
                    pos = f"{file_letter}{rank_num}"
                    save_square(square_img, cls_name, split, f"user_{i}", pos)
                    total_squares += 1

        corner_annotations.append({
            "source": "user",
            "image_path": str(path),
            "width": w, "height": h,
            "split": "train",
            "corners_pixel": {
                "top_left": corners_px[0],
                "top_right": corners_px[1],
                "bottom_right": corners_px[2],
                "bottom_left": corners_px[3],
            },
        })
        print(f"  [{i+1}] {path.name} ({w}x{h})")

    print(f"  User: {total_squares} squares from {len(corner_annotations)} images")
    return total_squares, corner_annotations


def update_corner_annotations(kaggle_corners, user_corners):
    """Add Kaggle and user corner annotations to the main annotations.json."""
    ann_path = BASE_DIR / "data" / "annotations.json"
    if not ann_path.exists():
        print(f"\n{ann_path} not found — skipping corner annotation update")
        return

    with open(ann_path) as f:
        data = json.load(f)

    # Find next available image ID and annotation ID
    max_img_id = max(img["id"] for img in data["images"]) if data["images"] else 0
    max_ann_id = max(a["image_id"] for a in data["annotations"]["corners"]) if data["annotations"]["corners"] else 0
    next_id = max(max_img_id, max_ann_id) + 1

    # Remove any previously added kaggle/user entries
    existing_paths = {img["path"] for img in data["images"]}
    kaggle_ids_to_add = []
    user_ids_to_add = []

    all_corners = kaggle_corners + user_corners
    new_images = []
    new_corner_anns = []

    for entry in all_corners:
        # Use relative path for matching
        rel_path = entry["image_path"]
        # Skip if already in annotations
        if rel_path in existing_paths:
            continue

        img_id = next_id
        next_id += 1

        new_images.append({
            "id": img_id,
            "path": rel_path,
            "width": entry["width"],
            "height": entry["height"],
        })
        new_corner_anns.append({
            "image_id": img_id,
            "corners": {
                "top_left": [int(entry["corners_pixel"]["top_left"][0]),
                             int(entry["corners_pixel"]["top_left"][1])],
                "top_right": [int(entry["corners_pixel"]["top_right"][0]),
                              int(entry["corners_pixel"]["top_right"][1])],
                "bottom_right": [int(entry["corners_pixel"]["bottom_right"][0]),
                                 int(entry["corners_pixel"]["bottom_right"][1])],
                "bottom_left": [int(entry["corners_pixel"]["bottom_left"][0]),
                                int(entry["corners_pixel"]["bottom_left"][1])],
            }
        })

        if entry["source"] == "kaggle":
            kaggle_ids_to_add.append((img_id, entry["split"]))
        else:
            user_ids_to_add.append((img_id, entry["split"]))

    if not new_images:
        print("\nNo new corner annotations to add")
        return

    data["images"].extend(new_images)
    data["annotations"]["corners"].extend(new_corner_anns)

    # Add to splits
    if "kaggle" not in data["splits"]:
        data["splits"]["kaggle"] = {"train": {"image_ids": []}, "val": {"image_ids": []}}
    if "user" not in data["splits"]:
        data["splits"]["user"] = {"train": {"image_ids": []}, "val": {"image_ids": []}}

    for img_id, split in kaggle_ids_to_add:
        data["splits"]["kaggle"][split]["image_ids"].append(img_id)
    for img_id, split in user_ids_to_add:
        data["splits"]["user"][split]["image_ids"].append(img_id)
        # Also add to chessred2k split so corner trainer picks them up
        data["splits"]["chessred2k"]["train"]["image_ids"].append(img_id)

    # Add kaggle to chessred2k split too so corner trainer uses them
    for img_id, split in kaggle_ids_to_add:
        data["splits"]["chessred2k"][split]["image_ids"].append(img_id)

    with open(ann_path, 'w') as f:
        json.dump(data, f)

    print(f"\nAdded {len(new_images)} images to annotations.json")
    print(f"  Kaggle: {len(kaggle_ids_to_add)} (train: {sum(1 for _,s in kaggle_ids_to_add if s=='train')}, val: {sum(1 for _,s in kaggle_ids_to_add if s=='val')})")
    print(f"  User: {len(user_ids_to_add)}")


def main():
    print("=" * 60)
    print("Preparing training data from all sources")
    print("=" * 60)

    kaggle_squares, kaggle_corners = process_kaggle()
    user_squares, user_corners = process_user_images()

    # Update corner annotations
    update_corner_annotations(kaggle_corners, user_corners)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Total new squares: {kaggle_squares + user_squares}")
    print(f"  Kaggle: {kaggle_squares}")
    print(f"  User: {user_squares}")

    # Count total squares per class
    print(f"\nSquare distribution:")
    for split in ["train", "val"]:
        split_dir = SQUARES_DIR / split
        if not split_dir.exists():
            continue
        total = 0
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = len(list(cls_dir.glob("*.jpg")))
                total += n
        print(f"  {split}: {total} total squares")

    print(f"\nNext steps:")
    print(f"  python3 training/train_corners.py --epochs 40 --batch-size 16 --lr 0.001")
    print(f"  python3 training/train_squares.py --epochs 30 --batch-size 32 --lr 0.001")


if __name__ == "__main__":
    main()
