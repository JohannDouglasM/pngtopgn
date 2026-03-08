#!/usr/bin/env python3
"""Test pipeline with manually specified corners to find accuracy ceiling."""

import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE * 2
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2
MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = 0.25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)

CLASS_NAMES = [
    'black_bishop', 'black_king', 'black_knight', 'black_pawn',
    'black_queen', 'black_rook', 'empty', 'white_bishop',
    'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
]
CLASS_TO_FEN = {
    'black_bishop': 'b', 'black_king': 'k', 'black_knight': 'n',
    'black_pawn': 'p', 'black_queen': 'q', 'black_rook': 'r',
    'empty': None,
    'white_bishop': 'B', 'white_king': 'K', 'white_knight': 'N',
    'white_pawn': 'P', 'white_queen': 'Q', 'white_rook': 'R',
}
PIECE_UNICODE = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656", "B": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f",
}


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
        [MARGIN, MARGIN], [BOARD_SIZE + MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN], [MARGIN, BOARD_SIZE + MARGIN],
    ], dtype=np.float32)
    M, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE))
    # Save debug
    debug = warped.copy()
    sq = SQUARE_SIZE
    for i in range(9):
        y = int(MARGIN + i * sq)
        cv2.line(debug, (int(MARGIN), y), (int(MARGIN + BOARD_SIZE), y), (255, 0, 0), 1)
        x = int(MARGIN + i * sq)
        cv2.line(debug, (x, int(MARGIN)), (x, int(MARGIN + BOARD_SIZE)), (255, 0, 0), 1)
    cv2.imwrite("training/debug_output/warped_manual.jpg", cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
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
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(IMG_SIZE, x2), min(IMG_SIZE, y2)
    cropped = warped[y1:y2, x1:x2]
    if col < 4:
        cropped = cv2.flip(cropped, 1)
    result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped.dtype)
    h, w = cropped.shape[:2]
    result[OUT_HEIGHT - h:, :w] = cropped
    return result


def main():
    if len(sys.argv) < 10:
        print("Usage: python3 test_manual_corners.py <image> <tl_x> <tl_y> <tr_x> <tr_y> <br_x> <br_y> <bl_x> <bl_y> [--expected-fen <fen>]")
        sys.exit(1)

    image_path = sys.argv[1]
    corners = np.array([
        [float(sys.argv[2]), float(sys.argv[3])],  # TL
        [float(sys.argv[4]), float(sys.argv[5])],  # TR
        [float(sys.argv[6]), float(sys.argv[7])],  # BR
        [float(sys.argv[8]), float(sys.argv[9])],  # BL
    ], dtype=np.float32)

    expected_fen = None
    if "--expected-fen" in sys.argv:
        idx = sys.argv.index("--expected-fen")
        expected_fen = sys.argv[idx + 1]

    # Load classifier
    square_ckpt = torch.load(MODELS_DIR / "best_square_classifier.pt", map_location="cpu", weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(square_ckpt["classes"]))
    model.load_state_dict(square_ckpt["model_state_dict"])
    model.eval()

    warped = warp_board(image_path, corners)

    transform = transforms.Compose([
        transforms.Resize((OUT_HEIGHT, OUT_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    board = {}
    for row in range(8):
        for col in range(8):
            crop = crop_square(warped, row, col)
            x = transform(Image.fromarray(crop)).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1).item()
                conf = probs[0, pred_idx].item()
            cls_name = CLASS_NAMES[pred_idx]
            fen_char = CLASS_TO_FEN[cls_name]
            pos = f"{chr(ord('a') + col)}{8 - row}"
            board[pos] = (fen_char, conf, cls_name)

    ranks = []
    for rank in range(8, 0, -1):
        rank_str = ""
        empty = 0
        for file_idx in range(8):
            pos = f"{chr(ord('a') + file_idx)}{rank}"
            piece, _, _ = board[pos]
            if piece is None:
                empty += 1
            else:
                if empty > 0:
                    rank_str += str(empty)
                    empty = 0
                rank_str += piece
        if empty > 0:
            rank_str += str(empty)
        ranks.append(rank_str)
    fen = "/".join(ranks)

    print(f"FEN: {fen}")
    print("  a b c d e f g h")
    for i, rank in enumerate(fen.split("/")):
        line = f"{8 - i} "
        for ch in rank:
            if ch.isdigit():
                line += "\u00b7 " * int(ch)
            else:
                line += PIECE_UNICODE.get(ch, ch) + " "
        print(line)

    if expected_fen:
        def expand(f):
            result = []
            for r in f.split("/"):
                for ch in r:
                    if ch.isdigit():
                        result.extend([None] * int(ch))
                    else:
                        result.append(ch)
            return result
        p = expand(fen)
        e = expand(expected_fen)
        correct = sum(1 for a, b in zip(p, e) if a == b)
        print(f"\nAccuracy: {correct}/64 ({correct/64*100:.1f}%)")
        if correct < 64:
            for i, (a, b) in enumerate(zip(p, e)):
                if a != b:
                    f = chr(ord('a') + (i % 8))
                    r = 8 - (i // 8)
                    print(f"  {f}{r}: predicted={a or '.'} expected={b or '.'}")


if __name__ == "__main__":
    main()
