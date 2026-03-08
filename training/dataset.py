"""
Dataset loaders for chess board recognition.

Supports:
- ChessReD / ChessReD2K (real photos with FEN + optional corners)
- Manual annotations (your own photos)
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from model import FEN_TO_CLASS, NUM_CLASSES

IMG_SIZE = 400  # Match Fenify-3D input size


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class ChessReD2KDataset(Dataset):
    """
    ChessReD2K dataset: real photos with piece annotations + corner annotations.

    Each sample: full board image → 64-class target tensor.
    We reconstruct FEN from per-square piece annotations.
    """

    def __init__(self, annotations_path: str, images_root: str,
                 split: str = "train", transform=None):
        """
        Args:
            annotations_path: Path to annotations.json
            images_root: Root directory containing images/ folder
            split: "train", "val", or "test"
            transform: torchvision transforms
        """
        with open(annotations_path) as f:
            data = json.load(f)

        self.images_root = Path(images_root)
        self.transform = transform

        # Build category map
        self.cat_map = {}
        for cat in data["categories"]:
            self.cat_map[cat["id"]] = cat["name"]

        # Map category names to our class indices
        self.cat_to_class = {
            "empty": 0,
            "white-pawn": 1, "white-knight": 2, "white-bishop": 3,
            "white-rook": 4, "white-queen": 5, "white-king": 6,
            "black-pawn": 7, "black-knight": 8, "black-bishop": 9,
            "black-rook": 10, "black-queen": 11, "black-king": 12,
        }

        # Get image IDs for this split
        # ChessReD2K subset has corner annotations
        split_ids = set(data["splits"]["chessred2k"][split]["image_ids"])

        # Build image info lookup
        self.images = {}
        for img in data["images"]:
            if img["id"] in split_ids:
                self.images[img["id"]] = img

        # Build per-image piece annotations
        self.piece_annotations = {}
        for piece in data["annotations"]["pieces"]:
            img_id = piece["image_id"]
            if img_id in self.images:
                if img_id not in self.piece_annotations:
                    self.piece_annotations[img_id] = []
                self.piece_annotations[img_id].append(piece)

        # Build corner annotations lookup
        self.corner_annotations = {}
        for corner in data["annotations"]["corners"]:
            img_id = corner["image_id"]
            if img_id in self.images:
                self.corner_annotations[img_id] = corner["corners"]

        # Final sample list: only images that have both pieces and corners
        self.sample_ids = [
            img_id for img_id in self.images
            if img_id in self.piece_annotations and img_id in self.corner_annotations
        ]
        self.sample_ids.sort()

        print(f"ChessReD2K {split}: {len(self.sample_ids)} samples "
              f"(from {len(split_ids)} in split)")

    def _chess_pos_to_square(self, pos: str) -> int:
        """Convert algebraic notation (e.g. 'e4') to square index (0-63).
        a1=0, b1=1, ..., h8=63 (matching python-chess)."""
        file_idx = ord(pos[0]) - ord('a')  # 0-7
        rank_idx = int(pos[1]) - 1  # 0-7
        return rank_idx * 8 + file_idx

    def _build_target(self, img_id: int) -> torch.Tensor:
        """Build a (64,) target tensor from piece annotations."""
        target = torch.zeros(64, dtype=torch.long)  # all empty by default
        for piece in self.piece_annotations.get(img_id, []):
            cat_name = self.cat_map[piece["category_id"]]
            cls_idx = self.cat_to_class.get(cat_name, 0)
            sq = self._chess_pos_to_square(piece["chessboard_position"])
            target[sq] = cls_idx
        return target

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        img_id = self.sample_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.images_root / img_info["path"]
        image = Image.open(img_path).convert("RGB")

        # Get corners for potential cropping
        corners = self.corner_annotations[img_id]

        # Crop to board region using corners (bounding box of corners)
        all_x = [corners[k][0] for k in corners]
        all_y = [corners[k][1] for k in corners]
        margin = 50  # small margin around board
        left = max(0, min(all_x) - margin)
        top = max(0, min(all_y) - margin)
        right = min(image.width, max(all_x) + margin)
        bottom = min(image.height, max(all_y) + margin)
        image = image.crop((left, top, right, bottom))

        if self.transform:
            image = self.transform(image)

        target = self._build_target(img_id)
        return image, target


class ChessReDFullDataset(Dataset):
    """
    Full ChessReD dataset (10,800 images) — no corner annotations needed.
    Uses the full image and reconstructs FEN from piece annotations.
    For use with end-to-end models that don't need corner detection.
    """

    def __init__(self, annotations_path: str, images_root: str,
                 split: str = "train", transform=None):
        with open(annotations_path) as f:
            data = json.load(f)

        self.images_root = Path(images_root)
        self.transform = transform

        self.cat_map = {}
        for cat in data["categories"]:
            self.cat_map[cat["id"]] = cat["name"]

        self.cat_to_class = {
            "empty": 0,
            "white-pawn": 1, "white-knight": 2, "white-bishop": 3,
            "white-rook": 4, "white-queen": 5, "white-king": 6,
            "black-pawn": 7, "black-knight": 8, "black-bishop": 9,
            "black-rook": 10, "black-queen": 11, "black-king": 12,
        }

        # Get image IDs for this split (full chessred)
        split_ids = set(data["splits"]["chessred"][split]["image_ids"])

        self.images = {}
        for img in data["images"]:
            if img["id"] in split_ids:
                self.images[img["id"]] = img

        self.piece_annotations = {}
        for piece in data["annotations"]["pieces"]:
            img_id = piece["image_id"]
            if img_id in self.images:
                if img_id not in self.piece_annotations:
                    self.piece_annotations[img_id] = []
                self.piece_annotations[img_id].append(piece)

        self.sample_ids = sorted(self.images.keys())
        print(f"ChessReD full {split}: {len(self.sample_ids)} samples")

    def _chess_pos_to_square(self, pos: str) -> int:
        file_idx = ord(pos[0]) - ord('a')
        rank_idx = int(pos[1]) - 1
        return rank_idx * 8 + file_idx

    def _build_target(self, img_id: int) -> torch.Tensor:
        target = torch.zeros(64, dtype=torch.long)
        for piece in self.piece_annotations.get(img_id, []):
            cat_name = self.cat_map[piece["category_id"]]
            cls_idx = self.cat_to_class.get(cat_name, 0)
            sq = self._chess_pos_to_square(piece["chessboard_position"])
            target[sq] = cls_idx
        return target

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        img_id = self.sample_ids[idx]
        img_info = self.images[img_id]

        img_path = self.images_root / img_info["path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        target = self._build_target(img_id)
        return image, target
