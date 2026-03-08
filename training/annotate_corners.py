#!/usr/bin/env python3
"""
Simple corner annotation tool for chess board images.

Click the 4 corners of the board in order: top-left, top-right, bottom-right, bottom-left.
Then enter the FEN for that position.

Usage:
    python3 training/annotate_corners.py <image_or_directory>

Saves annotations to training/data/manual/annotations.json
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

MANUAL_DIR = Path(__file__).parent / "data" / "manual"
ANNOTATIONS_FILE = MANUAL_DIR / "annotations.json"

corners = []
current_img = None
display_img = None
window_name = "Click 4 corners: TL -> TR -> BR -> BL"


def mouse_callback(event, x, y, flags, param):
    global corners, display_img
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append([x, y])
        # Draw the point
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        if len(corners) > 1:
            cv2.line(display_img, tuple(corners[-2]), tuple(corners[-1]), (0, 255, 0), 2)
        if len(corners) == 4:
            cv2.line(display_img, tuple(corners[3]), tuple(corners[0]), (0, 255, 0), 2)
            cv2.putText(display_img, "Press ENTER to confirm, R to reset",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        labels = ["TL", "TR", "BR", "BL"]
        for i, (cx, cy) in enumerate(corners):
            cv2.putText(display_img, labels[i], (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, display_img)


def annotate_image(img_path: Path) -> dict | None:
    global corners, current_img, display_img

    corners = []
    current_img = cv2.imread(str(img_path))
    if current_img is None:
        print(f"Could not read: {img_path}")
        return None

    # Resize for display if too large
    h, w = current_img.shape[:2]
    scale = 1.0
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        current_img = cv2.resize(current_img, (int(w * scale), int(h * scale)))

    display_img = current_img.copy()

    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\nAnnotating: {img_path.name}")
    print("Click 4 corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    print("Press R to reset, Q to skip, ESC to quit")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return "QUIT"

        if key == ord("q"):
            return None

        if key == ord("r"):
            corners = []
            display_img = current_img.copy()
            cv2.imshow(window_name, display_img)
            continue

        if key == 13 and len(corners) == 4:  # Enter
            break

    # Scale corners back to original image size
    original_corners = [[int(x / scale), int(y / scale)] for x, y in corners]

    # Get FEN from terminal
    cv2.destroyAllWindows()
    fen = input(f"Enter FEN for {img_path.name} (piece placement only): ").strip()
    if not fen:
        return None

    # Validate FEN
    ranks = fen.split("/")
    if len(ranks) != 8:
        print(f"Invalid FEN: expected 8 ranks, got {len(ranks)}")
        return None

    return {
        "image": img_path.name,
        "fen": fen,
        "corners": original_corners,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 training/annotate_corners.py <image_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_file():
        images = [target]
    elif target.is_dir():
        images = sorted(
            list(target.glob("*.jpg")) +
            list(target.glob("*.jpeg")) +
            list(target.glob("*.png"))
        )
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    # Load existing annotations
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    annotations = []
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations")

    already_annotated = {a["image"] for a in annotations}

    # Copy images to manual dir and annotate
    skipped = 0
    for img_path in images:
        if img_path.name in already_annotated:
            skipped += 1
            continue

        # Copy to manual dir if not already there
        dest = MANUAL_DIR / img_path.name
        if not dest.exists() and img_path.parent != MANUAL_DIR:
            import shutil
            shutil.copy2(img_path, dest)

        result = annotate_image(img_path)
        if result == "QUIT":
            break
        if result is not None:
            annotations.append(result)
            # Save after each annotation
            with open(ANNOTATIONS_FILE, "w") as f:
                json.dump(annotations, f, indent=2)
            print(f"Saved! Total annotations: {len(annotations)}")

    if skipped:
        print(f"Skipped {skipped} already-annotated images")

    cv2.destroyAllWindows()
    print(f"\nDone! {len(annotations)} total annotations in {ANNOTATIONS_FILE}")


if __name__ == "__main__":
    main()
