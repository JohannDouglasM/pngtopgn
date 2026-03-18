#!/usr/bin/env python3
"""
Train a hybrid corner detection model for chess board images.

Input: 3-channel image (384x384):
  Ch0: Grayscale (normalized)
  Ch1: Canny edge map
  Ch2: Square center heatmap (Gaussian blobs at detected empty square centers)

Output: 8 floats — normalized (x,y) coordinates for 4 corners
        [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]

Architecture: ResNet-18 (modified first conv for 3 custom channels) → Linear(512, 8)

Usage:
    python3 training/train_corners_hybrid.py [--epochs 30] [--batch-size 16]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import numpy as np
import cv2

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
IMG_SIZE = 384

# Sigma for Gaussian blobs in heatmap (relative to IMG_SIZE)
HEATMAP_SIGMA = 8


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


def find_empty_squares_for_image(image_bgr):
    """
    Detect empty square centers on a BGR image.
    Simplified version of _find_empty_squares from detect_board_v5.py.
    Returns list of (cx, cy) in pixel coordinates, or empty list.
    """
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    min_side = min(h, w) * 0.015
    max_side = min(h, w) * 0.12

    best_pts = []
    best_score = 0

    for thresh in [70, 80, 90, 100, 110, 120]:
        for flag in [cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY]:
            _, binary = cv2.threshold(l_ch, thresh, 255, flag)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            pts = []
            areas = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                side = np.sqrt(area)
                if side < min_side or side > max_side:
                    continue
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if max(rw, rh) < 1:
                    continue
                if min(rw, rh) / max(rw, rh) < 0.45:
                    continue
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                if solidity < 0.85:
                    continue
                bx, by, bw_r, bh_r = cv2.boundingRect(cnt)
                roi = l_ch[by:by + bh_r, bx:bx + bw_r]
                if roi.size == 0:
                    continue
                roi_mask = np.zeros((bh_r, bw_r), dtype=np.uint8)
                cnt_shifted = cnt.copy()
                cnt_shifted[:, :, 0] -= bx
                cnt_shifted[:, :, 1] -= by
                cv2.drawContours(roi_mask, [cnt_shifted], 0, 255, -1)
                pixels = roi[roi_mask > 0]
                if len(pixels) == 0 or np.std(pixels) > 18:
                    continue
                pts.append((rect[0][0], rect[0][1]))
                areas.append(area)

            if len(pts) < 4:
                continue

            areas_arr = np.array(areas)
            med_area = np.median(areas_arr)
            keep = (areas_arr > med_area * 0.4) & (areas_arr < med_area * 2.5)
            pts_filt = [p for p, k in zip(pts, keep) if k]

            if len(pts_filt) < 4:
                continue

            areas_filt = areas_arr[keep]
            size_cv = np.std(areas_filt) / np.mean(areas_filt)
            consistency = max(0, 1.0 - size_cv)
            score = len(pts_filt) * (0.5 + 0.5 * consistency)

            if score > best_score:
                best_score = score
                best_pts = pts_filt

    return best_pts


def make_heatmap(square_centers, orig_w, orig_h, out_size=IMG_SIZE):
    """
    Create a heatmap image with Gaussian blobs at detected square centers.
    Centers are in original image coordinates; output is out_size x out_size.
    Returns float32 array in [0, 1].
    """
    heatmap = np.zeros((out_size, out_size), dtype=np.float32)
    if not square_centers:
        return heatmap

    sx = out_size / orig_w
    sy = out_size / orig_h
    sigma = HEATMAP_SIGMA

    for (cx, cy) in square_centers:
        # Scale to output size
        x = cx * sx
        y = cy * sy

        # Draw Gaussian blob
        x0 = max(0, int(x - 3 * sigma))
        x1 = min(out_size, int(x + 3 * sigma) + 1)
        y0 = max(0, int(y - 3 * sigma))
        y1 = min(out_size, int(y + 3 * sigma) + 1)

        for iy in range(y0, y1):
            for ix in range(x0, x1):
                d2 = (ix - x) ** 2 + (iy - y) ** 2
                heatmap[iy, ix] = max(heatmap[iy, ix],
                                      np.exp(-d2 / (2 * sigma ** 2)))

    return heatmap


def make_3ch_input(image_bgr, square_centers, out_size=IMG_SIZE):
    """
    Build 3-channel input tensor from a BGR image:
      Ch0: Grayscale (0-1)
      Ch1: Canny edges (0-1)
      Ch2: Square center heatmap (0-1)
    Returns float32 numpy array (3, out_size, out_size).
    """
    h, w = image_bgr.shape[:2]

    # Resize to output size
    resized = cv2.resize(image_bgr, (out_size, out_size))

    # Channel 0: Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Channel 1: Canny edges
    gray_uint8 = (gray * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_uint8, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 50, 150).astype(np.float32) / 255.0

    # Channel 2: Heatmap
    heatmap = make_heatmap(square_centers, w, h, out_size)

    return np.stack([gray, edges, heatmap], axis=0)


class HybridCornerDataset(Dataset):
    """Dataset that produces 3-channel hybrid inputs + corner targets."""

    def __init__(self, annotations_path, images_root, split="train",
                 augment=False, cache_squares=True):
        with open(annotations_path) as f:
            data = json.load(f)

        self.images_root = Path(images_root)
        self.augment = augment

        # Build lookups
        self.images = {img["id"]: img for img in data["images"]}
        self.corner_map = {c["image_id"]: c["corners"]
                           for c in data["annotations"]["corners"]}

        # Get split IDs
        split_ids = data["splits"]["chessred2k"][split]["image_ids"]
        self.sample_ids = [i for i in split_ids
                           if i in self.corner_map and i in self.images]
        self.sample_ids.sort()

        # Cache detected squares per image (expensive to compute)
        self.square_cache = {}
        self.cache_squares = cache_squares

        print(f"Hybrid corner dataset {split}: {len(self.sample_ids)} samples")

    def _get_squares(self, img_id, image_bgr):
        if img_id in self.square_cache:
            return self.square_cache[img_id]
        centers = find_empty_squares_for_image(image_bgr)
        if self.cache_squares:
            self.square_cache[img_id] = centers
        return centers

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        img_id = self.sample_ids[idx]
        img_info = self.images[img_id]
        corners = self.corner_map[img_id]

        # Load image as BGR for CV operations
        img_path = self.images_root / img_info["path"]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            # Missing file — return zeros and dummy target
            dummy_input = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
            dummy_target = torch.tensor([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8],
                                        dtype=torch.float32)
            return dummy_input, dummy_target
        orig_h, orig_w = image_bgr.shape[:2]

        # Detect squares
        square_centers = self._get_squares(img_id, image_bgr)

        # Augmentation: brightness/contrast jitter on the BGR image
        if self.augment:
            # Random brightness
            beta = np.random.uniform(-40, 40)
            # Random contrast
            alpha = np.random.uniform(0.6, 1.4)
            image_bgr = np.clip(alpha * image_bgr.astype(np.float32) + beta,
                                0, 255).astype(np.uint8)

        # Build 3-channel input
        input_3ch = make_3ch_input(image_bgr, square_centers, IMG_SIZE)

        # Normalize corner coordinates to [0, 1]
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

        return torch.from_numpy(input_3ch), target


def build_model():
    """ResNet-18 with 3-channel input and regression head for 4 corners."""
    model = models.resnet18(weights=None)

    # Load pretrained weights where possible
    import os
    cached = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    if os.path.exists(cached):
        state = torch.load(cached, map_location="cpu", weights_only=True)
        # The first conv expects 3 channels — our input is also 3 channels,
        # but they're not RGB. We still load the pretrained weights since
        # the learned edge/texture filters are useful starting points.
        model.load_state_dict(state)

    n = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(n, 8),
        nn.Sigmoid(),
    )
    return model


def corner_distance(pred, target):
    """Mean Euclidean distance between predicted and target corners."""
    pred = pred.view(-1, 4, 2)
    target = target.view(-1, 4, 2)
    dists = torch.sqrt(((pred - target) ** 2).sum(dim=2))
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

            pred = outputs.view(-1, 4, 2)
            targ = targets.view(-1, 4, 2)
            dists = torch.sqrt(((pred - targ) ** 2).sum(dim=2))
            total_dist += dists.mean().item() * images.size(0)
            all_dists.append(dists.cpu())
            total += images.size(0)

    all_dists = torch.cat(all_dists, dim=0)
    mean_dist = all_dists.mean().item()
    max_dist = all_dists.max().item()
    per_corner = all_dists.mean(dim=0)

    return total_loss / total, mean_dist, max_dist, per_corner


def main():
    parser = argparse.ArgumentParser(description="Train hybrid corner detection model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
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

    # Pre-compute square detection for all images (slow but only once)
    print("\nBuilding datasets (square detection will be cached on first epoch)...")
    train_dataset = HybridCornerDataset(annotations_path, images_root, "train",
                                         augment=True, cache_squares=True)
    val_dataset = HybridCornerDataset(annotations_path, images_root, "val",
                                       augment=False, cache_squares=True)

    num_workers = 0  # Must be 0 for MPS and for our caching to work
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

    print(f"\nTraining HYBRID corner detector for {args.epochs} epochs on {device}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Input: {IMG_SIZE}x{IMG_SIZE} x 3ch (gray + canny + heatmap)")
    print("=" * 60, flush=True)

    if head_epochs > 0:
        print(f"\nPhase 1: Head only for {head_epochs} epochs")
        for epoch in range(head_epochs):
            start = time.time()
            epoch_str = f"[{epoch+1}/{args.epochs}]"
            train_loss, train_dist = train_epoch(model, train_loader, criterion,
                                                  optimizer, device, epoch_str)
            val_loss, val_dist, val_max, per_corner = validate(model, val_loader,
                                                                criterion, device)
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
                    "channels": "gray+canny+heatmap",
                }, MODELS_DIR / "best_corner_hybrid.pt")
                print(f"  -> New best! dist={val_dist:.4f}", flush=True)

    # Phase 2: full fine-tune
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=args.epochs - head_epochs)

    remaining = args.epochs - head_epochs
    print(f"\nPhase 2: Fine-tuning all layers for {remaining} epochs "
          f"(early stop after 7 no-improve)")
    print("=" * 60, flush=True)

    for epoch in range(remaining):
        start = time.time()
        epoch_str = f"[{head_epochs+epoch+1}/{args.epochs}]"
        train_loss, train_dist = train_epoch(model, train_loader, criterion,
                                              optimizer, device, epoch_str)
        val_loss, val_dist, val_max, per_corner = validate(model, val_loader,
                                                            criterion, device)
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
                "channels": "gray+canny+heatmap",
            }, MODELS_DIR / "best_corner_hybrid.pt")
            print(f"  -> New best! dist={val_dist:.4f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 7:
                print("  Early stopping: no improvement for 7 epochs", flush=True)
                break

    print(f"\n{'='*60}")
    print(f"Done! Best mean corner distance: {best_val_dist:.4f}")
    print(f"  (On a 3072px image, {best_val_dist:.4f} ~ {best_val_dist*3072:.0f}px per corner)")

    return  # autoresearch: skip ONNX export

    return  # autoresearch: skip ONNX export

    # Export to ONNX
    print("\nExporting to ONNX...")
    best_ckpt = torch.load(MODELS_DIR / "best_corner_hybrid.pt",
                           map_location="cpu", weights_only=True)
    export_model = build_model()
    export_model.load_state_dict(best_ckpt["model_state_dict"])
    export_model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = MODELS_DIR / "corner_hybrid.onnx"
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
