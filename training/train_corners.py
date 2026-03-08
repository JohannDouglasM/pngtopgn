#!/usr/bin/env python3
"""
Train a corner detection model for chess board images.

Input: RGB image (resized to 256x256)
Output: 8 floats — normalized (x,y) coordinates for 4 corners
        [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]

Architecture: ResNet-18 backbone → Linear(512, 8)

Usage:
    python3 training/train_corners.py [--epochs 30] [--batch-size 16]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
IMG_SIZE = 384


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


class CornerDataset(Dataset):
    """Dataset of chess board images with corner annotations."""

    def __init__(self, annotations_path, images_root, split="train", transform=None):
        with open(annotations_path) as f:
            data = json.load(f)

        self.images_root = Path(images_root)
        self.transform = transform

        # Build lookups
        self.images = {img["id"]: img for img in data["images"]}
        self.corner_map = {c["image_id"]: c["corners"] for c in data["annotations"]["corners"]}

        # Get split IDs that have corners
        split_ids = data["splits"]["chessred2k"][split]["image_ids"]
        self.sample_ids = [i for i in split_ids if i in self.corner_map and i in self.images]
        self.sample_ids.sort()
        print(f"Corner dataset {split}: {len(self.sample_ids)} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        img_id = self.sample_ids[idx]
        img_info = self.images[img_id]
        corners = self.corner_map[img_id]

        # Load image
        img_path = self.images_root / img_info["path"]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Normalize corner coordinates to [0, 1] relative to original image
        target = torch.tensor([
            corners["top_left"][0] / orig_w,
            corners["top_left"][1] / orig_h,
            corners["top_right"][0] / orig_w,
            corners["top_right"][1] / orig_h,
            corners["bottom_right"][0] / orig_w,
            corners["bottom_right"][1] / orig_h,
            corners["bottom_left"][0] / orig_w,
            corners["bottom_left"][1] / orig_h,
        ], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


def build_model():
    """ResNet-18 with regression head for 4 corner points (8 values)."""
    import os
    model = models.resnet18(weights=None)
    cached = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    if os.path.exists(cached):
        state = torch.load(cached, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    n = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n, 8),
        nn.Sigmoid(),  # Output in [0, 1] range
    )
    return model


def corner_distance(pred, target, img_size=1.0):
    """Mean Euclidean distance between predicted and target corners (in normalized coords)."""
    # pred, target: (B, 8) — 4 corners × 2 coords
    pred = pred.view(-1, 4, 2)
    target = target.view(-1, 4, 2)
    dists = torch.sqrt(((pred - target) ** 2).sum(dim=2))  # (B, 4)
    return dists.mean()


def train_epoch(model, loader, criterion, optimizer, device, epoch_str=""):
    model.train()
    total_loss = 0
    total_dist = 0
    total = 0
    n_batches = len(loader)
    log_every = max(1, n_batches // 4)

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_dist += corner_distance(outputs, targets).item() * images.size(0)
        total += images.size(0)

        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            pct = (i + 1) / n_batches * 100
            print(f"  {epoch_str} train {pct:5.1f}% | batch {i+1}/{n_batches} | "
                  f"loss={total_loss/total:.4f} dist={total_dist/total:.4f}",
                  flush=True)

    return total_loss / total, total_dist / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dist = 0
    total = 0
    all_dists = []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)

            # Per-corner distances
            pred = outputs.view(-1, 4, 2)
            targ = targets.view(-1, 4, 2)
            dists = torch.sqrt(((pred - targ) ** 2).sum(dim=2))  # (B, 4)
            total_dist += dists.mean().item() * images.size(0)
            all_dists.append(dists.cpu())
            total += images.size(0)

    all_dists = torch.cat(all_dists, dim=0)  # (N, 4)
    mean_dist = all_dists.mean().item()
    max_dist = all_dists.max().item()
    per_corner = all_dists.mean(dim=0)  # (4,)

    return total_loss / total, mean_dist, max_dist, per_corner


def main():
    parser = argparse.ArgumentParser(description="Train corner detection model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Find data
    annotations_path = None
    for p in [BASE_DIR / "data" / "annotations.json", BASE_DIR.parent / "annotations.json"]:
        if p.exists():
            annotations_path = str(p)
            break
    if not annotations_path:
        print("annotations.json not found!")
        sys.exit(1)

    images_root = None
    for p in [BASE_DIR / "data" / "chessred2k", BASE_DIR / "data"]:
        if (p / "images").exists():
            images_root = str(p)
            break
    if not images_root:
        print("images/ directory not found!")
        sys.exit(1)

    print(f"Annotations: {annotations_path}")
    print(f"Images root: {images_root}")

    device = get_device()

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15),
        transforms.RandomGrayscale(p=0.15),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CornerDataset(annotations_path, images_root, "train", train_transform)
    val_dataset = CornerDataset(annotations_path, images_root, "val", val_transform)

    num_workers = 0 if device.type == "mps" else 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=False)

    model = build_model()
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)

    # Use smooth L1 loss (Huber) — more robust than MSE for regression
    criterion = nn.SmoothL1Loss()

    # Phase 1: head only
    head_epochs = min(3, args.epochs) if not args.resume else 0

    if head_epochs > 0:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_dist = float("inf")
    no_improve = 0

    print(f"\nTraining corner detector for {args.epochs} epochs on {device}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Input size: {IMG_SIZE}x{IMG_SIZE}")
    print("=" * 60, flush=True)

    if head_epochs > 0:
        print(f"\nPhase 1: Head only for {head_epochs} epochs")
        for epoch in range(head_epochs):
            start = time.time()
            epoch_str = f"[{epoch+1}/{args.epochs}]"
            train_loss, train_dist = train_epoch(model, train_loader, criterion, optimizer, device, epoch_str)
            val_loss, val_dist, val_max, per_corner = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start

            print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) | "
                  f"Train: loss={train_loss:.4f} dist={train_dist:.4f} | "
                  f"Val: loss={val_loss:.4f} dist={val_dist:.4f} max={val_max:.4f}",
                  flush=True)

            if val_dist < best_val_dist:
                best_val_dist = val_dist
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dist": val_dist,
                    "img_size": IMG_SIZE,
                }, MODELS_DIR / "best_corner_detector.pt")
                print(f"  -> New best! dist={val_dist:.4f}", flush=True)

    # Phase 2: full fine-tune
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - head_epochs)

    remaining = args.epochs - head_epochs
    print(f"\nPhase 2: Fine-tuning all layers for {remaining} epochs (early stop after 7 no-improve)")
    print("=" * 60, flush=True)

    for epoch in range(remaining):
        start = time.time()
        epoch_str = f"[{head_epochs+epoch+1}/{args.epochs}]"
        train_loss, train_dist = train_epoch(model, train_loader, criterion, optimizer, device, epoch_str)
        val_loss, val_dist, val_max, per_corner = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        corner_names = ["TL", "TR", "BR", "BL"]
        per_corner_str = " ".join(f"{n}={d:.4f}" for n, d in zip(corner_names, per_corner))

        print(f"Epoch {head_epochs+epoch+1}/{args.epochs} ({elapsed:.0f}s) | "
              f"Train: loss={train_loss:.4f} dist={train_dist:.4f} | "
              f"Val: loss={val_loss:.4f} dist={val_dist:.4f} max={val_max:.4f} | "
              f"{per_corner_str}",
              flush=True)

        if val_dist < best_val_dist:
            best_val_dist = val_dist
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": head_epochs + epoch,
                "val_dist": val_dist,
                "img_size": IMG_SIZE,
            }, MODELS_DIR / "best_corner_detector.pt")
            print(f"  -> New best! dist={val_dist:.4f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 7:
                print("  Early stopping: no improvement for 7 epochs", flush=True)
                break

    print(f"\n{'='*60}")
    print(f"Done! Best mean corner distance: {best_val_dist:.4f}")
    # Distance is in normalized [0,1] coords. On a 3072px image, 0.01 = ~30px
    print(f"  (On a 3072px image, {best_val_dist:.4f} ≈ {best_val_dist*3072:.0f}px per corner)")

    # Export to ONNX
    print("\nExporting to ONNX...")
    best_ckpt = torch.load(MODELS_DIR / "best_corner_detector.pt",
                           map_location="cpu", weights_only=True)
    export_model = build_model()
    export_model.load_state_dict(best_ckpt["model_state_dict"])
    export_model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = MODELS_DIR / "corner_detector.onnx"
    torch.onnx.export(
        export_model, dummy, str(onnx_path),
        input_names=["image"], output_names=["corners"],
        dynamic_axes={"image": {0: "batch"}, "corners": {0: "batch"}},
        opset_version=13,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported to {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
