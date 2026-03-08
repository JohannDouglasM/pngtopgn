#!/usr/bin/env python3
"""
Prepare square training data from the FULL ChessReD dataset (~10.9K images).

For images WITH corner annotations: use ground-truth corners.
For images WITHOUT corner annotations: use the trained corner detector to predict corners.

This gives ~5x more training data than ChessReD2K alone.

Usage:
    python3 training/prepare_squares_full.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

BASE_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
SQUARES_DIR = BASE_DIR / "squares"

SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE * 2
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2

CORNER_SIZE = 384

MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = 0.25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)

PIECE_CLASSES = [
    "empty",
    "white_pawn", "white_knight", "white_bishop",
    "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop",
    "black_rook", "black_queen", "black_king",
]

CHESSRED_TO_CLASS = {
    "white-pawn": "white_pawn", "white-knight": "white_knight",
    "white-bishop": "white_bishop", "white-rook": "white_rook",
    "white-queen": "white_queen", "white-king": "white_king",
    "black-pawn": "black_pawn", "black-knight": "black_knight",
    "black-bishop": "black_bishop", "black-rook": "black_rook",
    "black-queen": "black_queen", "black-king": "black_king",
    "empty": "empty",
}


def build_corner_model():
    model = models.resnet18(weights=None)
    n = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(n, 8), nn.Sigmoid())
    return model


def load_corner_model():
    ckpt = torch.load(MODELS_DIR / "best_corner_detector.pt", map_location="cpu", weights_only=True)
    model = build_corner_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_corners(model, img_path, device="cpu"):
    """Use the corner detector to predict corners for an image."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    transform = transforms.Compose([
        transforms.Resize((CORNER_SIZE, CORNER_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()

    corners = np.array([
        [pred[0] * orig_w, pred[1] * orig_h],
        [pred[2] * orig_w, pred[3] * orig_h],
        [pred[4] * orig_w, pred[5] * orig_h],
        [pred[6] * orig_w, pred[7] * orig_h],
    ], dtype=np.float32)

    return corners


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

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(IMG_SIZE, x2)
    y2 = min(IMG_SIZE, y2)

    cropped = warped[y1:y2, x1:x2]

    if col < 4:
        cropped = cv2.flip(cropped, 1)

    result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped.dtype)
    h, w = cropped.shape[:2]
    result[OUT_HEIGHT - h:, :w] = cropped
    return result


def main():
    annotations_path = BASE_DIR / "annotations.json"
    with open(annotations_path) as f:
        data = json.load(f)

    # Build lookups
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}
    corner_map = {c["image_id"]: c["corners"] for c in data["annotations"]["corners"]}

    # Build per-image piece map
    piece_map = {}
    for piece in data["annotations"]["pieces"]:
        img_id = piece["image_id"]
        if img_id not in piece_map:
            piece_map[img_id] = {}
        pos = piece["chessboard_position"]
        cat_name = cat_map[piece["category_id"]]
        piece_map[img_id][pos] = CHESSRED_TO_CLASS.get(cat_name, "empty")

    # Determine which images need predicted corners
    has_corners = set(corner_map.keys())
    has_pieces = set(piece_map.keys())
    needs_prediction = has_pieces - has_corners
    print(f"Images with ground-truth corners: {len(has_corners & has_pieces)}", flush=True)
    print(f"Images needing corner prediction: {len(needs_prediction)}", flush=True)

    # Image roots - try both chessred and chessred2k
    # Note: path field already includes "images/", so use parent directory
    chessred_root = BASE_DIR / "chessred"
    chessred2k_root = BASE_DIR / "chessred2k"

    # Load corner model if needed
    corner_model = None
    device = "cpu"
    if needs_prediction:
        print("Loading corner detector for prediction...", flush=True)
        if torch.backends.mps.is_available():
            device = "mps"
        corner_model = load_corner_model().to(device)
        print(f"Using {device}", flush=True)

    # Create output directories
    for split in ["train", "val"]:
        for cls in PIECE_CLASSES:
            (SQUARES_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # Process train and val splits
    for split in ["train", "val"]:
        split_ids = data["splits"]["train"]["image_ids"] if split == "train" else data["splits"]["val"]["image_ids"]
        print(f"\n{split}: {len(split_ids)} images", flush=True)

        counts = {cls: 0 for cls in PIECE_CLASSES}
        errors = 0
        predicted = 0
        gt_used = 0

        for idx, img_id in enumerate(split_ids):
            if img_id not in images or img_id not in piece_map:
                continue

            img_info = images[img_id]
            img_path = None

            # Try chessred2k first, then full chessred
            path_rel = img_info.get("path", img_info["file_name"])
            for root in [chessred2k_root, chessred_root]:
                candidate = root / path_rel
                if candidate.exists():
                    img_path = candidate
                    break
                # Also try with just filename
                candidate = root / img_info["file_name"]
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                errors += 1
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    errors += 1
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get corners
                if img_id in corner_map:
                    corners_dict = corner_map[img_id]
                    corners = np.array([
                        corners_dict["top_left"],
                        corners_dict["top_right"],
                        corners_dict["bottom_right"],
                        corners_dict["bottom_left"],
                    ], dtype=np.float32)
                    gt_used += 1
                else:
                    corners = predict_corners(corner_model, img_path, device)
                    predicted += 1

                warped = warp_board(img, corners)
                pieces = piece_map.get(img_id, {})

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
                errors += 1
                continue

            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(split_ids)} (gt={gt_used}, predicted={predicted}, errors={errors})", flush=True)

        total_squares = sum(counts.values())
        print(f"  Done: {total_squares} squares (gt_corners={gt_used}, predicted_corners={predicted}, errors={errors})", flush=True)
        print(f"  Class distribution:", flush=True)
        for cls in sorted(counts.keys()):
            print(f"    {cls}: {counts[cls]}", flush=True)


if __name__ == "__main__":
    main()
