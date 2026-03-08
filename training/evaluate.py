#!/usr/bin/env python3
"""
Evaluate the trained model on a single board image.

Usage:
    python3 training/evaluate.py <image_path> [--expected-fen <fen>]
    python3 training/evaluate.py assets/test_board.jpeg --expected-fen "r1bq1rk1/pp1pppbp/5np1/8/2PQ4/2N3P1/PP2PPBP/R1B2RK1"
"""

import argparse
import sys
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import ChessRecognitionModel, target_to_fen, CLASS_TO_FEN

MODELS_DIR = Path(__file__).parent / "models"

PIECE_UNICODE = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656", "B": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f",
}


def print_board(fen):
    ranks = fen.split("/")
    print("  a b c d e f g h")
    for i, rank in enumerate(ranks):
        line = f"{8 - i} "
        for ch in rank:
            if ch.isdigit():
                line += "\u00b7 " * int(ch)
            else:
                line += PIECE_UNICODE.get(ch, ch) + " "
        print(line)


def compare_fens(predicted, expected):
    def expand(fen):
        result = []
        for rank in fen.split("/"):
            for ch in rank:
                if ch.isdigit():
                    result.extend([None] * int(ch))
                else:
                    result.append(ch)
        return result

    p = expand(predicted)
    e = expand(expected)
    correct = sum(1 for a, b in zip(p, e) if a == b)
    total = 64

    if correct < total:
        print("Mismatches:")
        for i, (a, b) in enumerate(zip(p, e)):
            if a != b:
                file_letter = chr(ord('a') + (i % 8))
                rank_num = 8 - (i // 8)
                print(f"  {file_letter}{rank_num}: predicted={a or '.'} expected={b or '.'}")

    print(f"\nSquare accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to board image")
    parser.add_argument("--expected-fen", type=str)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else MODELS_DIR / "best_model.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first: python3 training/train.py")
        sys.exit(1)

    # Load model
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    backbone = ckpt.get("backbone", args.backbone)
    model = ChessRecognitionModel(backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model ({backbone}) from {model_path}")

    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        logits = model(x)  # (1, 64, 13)
        probs = torch.softmax(logits, dim=2)
        preds = logits.argmax(dim=2).squeeze(0)  # (64,)
        confs = probs.max(dim=2).values.squeeze(0)  # (64,)

    fen = target_to_fen(preds)
    print(f"\nPredicted FEN: {fen}")
    print()
    print_board(fen)

    # Show low-confidence squares
    low_conf = []
    for sq in range(64):
        if confs[sq] < 0.8:
            file_letter = chr(ord('a') + (sq % 8))
            rank_num = (sq // 8) + 1
            cls_name = CLASS_TO_FEN.get(preds[sq].item(), "?")
            low_conf.append(f"  {file_letter}{rank_num}: {cls_name or '.'} ({confs[sq]:.2f})")

    if low_conf:
        print(f"\nLow confidence ({len(low_conf)} squares):")
        for s in low_conf:
            print(s)

    if args.expected_fen:
        print(f"\nExpected FEN: {args.expected_fen}")
        compare_fens(fen, args.expected_fen)

    full_fen = fen + " w KQkq - 0 1"
    print(f"\nLichess: https://lichess.org/analysis/{full_fen.replace(' ', '_')}")


if __name__ == "__main__":
    main()
