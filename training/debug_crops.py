#!/usr/bin/env python3
"""Save warped board and cropped squares for visual debugging."""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

CORNER_SIZE = 384
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE * 2
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2

MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = 0.25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)


def build_corner_model():
    model = models.resnet18(weights=None)
    n = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(n, 8), nn.Sigmoid())
    return model


def detect_corners(model, image_path):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    transform = transforms.Compose([
        transforms.Resize((CORNER_SIZE, CORNER_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).squeeze(0).numpy()
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


def warp_board(image_path, corners):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    src = sort_corner_points(corners)
    dst = np.array([
        [MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN],
        [MARGIN, BOARD_SIZE + MARGIN],
    ], dtype=np.float32)
    M, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE))
    return warped


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
    if len(sys.argv) < 2:
        print("Usage: python3 training/debug_crops.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    out_dir = Path("training/debug_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load corner model
    ckpt = torch.load(MODELS_DIR / "best_corner_detector.pt", map_location="cpu", weights_only=True)
    model = build_corner_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Detect corners
    corners = detect_corners(model, image_path)
    print(f"Corners: {corners.tolist()}")

    # Draw corners on original
    img_orig = cv2.imread(str(image_path))
    for i, (x, y) in enumerate(corners):
        cv2.circle(img_orig, (int(x), int(y)), 15, (0, 0, 255), 3)
        cv2.putText(img_orig, f"{i}", (int(x)+20, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / "corners.jpg"), img_orig)

    # Warp
    warped = warp_board(image_path, corners)

    # Draw grid on warped
    warped_grid = warped.copy()
    for i in range(9):
        y = int(MARGIN + i * SQUARE_SIZE)
        x = int(MARGIN + i * SQUARE_SIZE)
        cv2.line(warped_grid, (int(MARGIN), y), (int(MARGIN + BOARD_SIZE), y), (255, 0, 0), 1)
        cv2.line(warped_grid, (x, int(MARGIN)), (x, int(MARGIN + BOARD_SIZE)), (255, 0, 0), 1)
    cv2.imwrite(str(out_dir / "warped_grid.jpg"), cv2.cvtColor(warped_grid, cv2.COLOR_RGB2BGR))

    # Save select squares
    for row, col, label in [(0, 0, "a8"), (0, 4, "e8"), (7, 0, "a1"), (7, 4, "e1"),
                             (1, 0, "a7"), (6, 0, "a2"), (0, 7, "h8"), (7, 7, "h1")]:
        sq = crop_square(warped, row, col)
        cv2.imwrite(str(out_dir / f"sq_{label}.jpg"), cv2.cvtColor(sq, cv2.COLOR_RGB2BGR))

    # Save ALL 64 squares in a grid (8x8, each 100x200)
    grid = np.zeros((8 * OUT_HEIGHT, 8 * OUT_WIDTH, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            sq = crop_square(warped, row, col)
            grid[row*OUT_HEIGHT:(row+1)*OUT_HEIGHT, col*OUT_WIDTH:(col+1)*OUT_WIDTH] = sq
    cv2.imwrite(str(out_dir / "all_squares.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
