#!/usr/bin/env python3
"""
Train a per-square piece classifier on warped+cropped square images.

Architecture: ResNet-18 fine-tuned for 13 classes.
Input: individual square crops (100x200) → Output: piece class.

Usage:
    python3 training/train_squares.py [--epochs 20] [--batch-size 64]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "squares"
MODELS_DIR = BASE_DIR / "models"

# Match chesscog square size: 100x200 (width x height) for piece classifier
IMG_WIDTH = 100
IMG_HEIGHT = 200


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


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_model(num_classes: int = 13):
    """ResNet-18 with custom head for piece classification."""
    import os
    model = models.resnet18(weights=None)
    cached = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    if os.path.exists(cached):
        state = torch.load(cached, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        # Download without hash check
        from torch.hub import download_url_to_file
        url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        download_url_to_file(url, cached, hash_prefix=None, progress=True)
        state = torch.load(cached, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    n = model.fc.in_features
    model.fc = nn.Linear(n, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device, epoch_str=""):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    n_batches = len(loader)
    log_every = max(1, n_batches // 4)  # Log at 25%, 50%, 75%, 100%

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            pct = (i + 1) / n_batches * 100
            acc_so_far = correct / total
            print(f"  {epoch_str} train {pct:5.1f}% | batch {i+1}/{n_batches} | "
                  f"loss={total_loss/total:.4f} acc={acc_so_far:.3f}",
                  flush=True)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    n_batches = len(loader)
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

            for pred, label in zip(predicted, labels):
                cls = label.item()
                class_total[cls] = class_total.get(cls, 0) + 1
                if pred == label:
                    class_correct[cls] = class_correct.get(cls, 0) + 1

    return total_loss / total, correct / total, class_correct, class_total


def main():
    parser = argparse.ArgumentParser(description="Train square piece classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    if not train_dir.exists():
        print(f"Training data not found at {train_dir}")
        print("Run: python3 training/prepare_squares.py")
        sys.exit(1)

    device = get_device()
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} squares")
    print(f"Val: {len(val_dataset)} squares")

    # Print class distribution
    print("\nClass distribution (train):")
    for cls_name in train_dataset.classes:
        d = train_dir / cls_name
        n = len(list(d.glob("*")))
        print(f"  {cls_name}: {n}")

    num_workers = 0 if device.type == "mps" else 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=False)

    num_classes = len(train_dataset.classes)
    model = build_model(num_classes=num_classes)

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)

    # Compute class weights: sqrt of inverse frequency (mild rebalancing)
    class_counts = []
    for cls_name in train_dataset.classes:
        d = train_dir / cls_name
        class_counts.append(len(list(d.glob("*"))))
    total_samples = sum(class_counts)
    # sqrt(median_freq / class_freq) — gentle rebalancing
    import math
    median_count = sorted(class_counts)[len(class_counts) // 2]
    weights = [math.sqrt(median_count / max(c, 1)) for c in class_counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"\nClass weights (sqrt-inv-freq):")
    for name, w in zip(train_dataset.classes, weights):
        print(f"  {name}: {w:.3f}")
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # When resuming, skip Phase 1 (head-only) — go straight to full fine-tune
    if args.resume:
        head_epochs = 0
    else:
        head_epochs = min(3, args.epochs)

    if head_epochs > 0:
        # Phase 1: freeze backbone
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0

    if head_epochs > 0:
        print(f"\nPhase 1: Training head only for {head_epochs} epochs")
        print("=" * 60)

        for epoch in range(head_epochs):
            start = time.time()
            epoch_str = f"[{epoch+1}/{args.epochs}]"
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch_str)
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start

            print(f"Epoch {epoch+1}/{head_epochs} ({elapsed:.0f}s) | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.3f}",
                  flush=True)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "classes": train_dataset.classes,
                    "class_to_idx": train_dataset.class_to_idx,
                }, MODELS_DIR / "best_square_classifier.pt")
                print(f"  -> New best! {val_acc:.3f}")

    # Phase 2: fine-tune all layers
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - head_epochs)

    remaining = args.epochs - head_epochs
    print(f"\nPhase 2: Fine-tuning all layers for {remaining} epochs (early stop after 5 no-improve)")
    print("=" * 60)
    no_improve = 0

    for epoch in range(remaining):
        start = time.time()
        epoch_str = f"[{head_epochs + epoch + 1}/{args.epochs}]"
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch_str)
        val_loss, val_acc, class_correct, class_total = validate(
            model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        print(f"Epoch {head_epochs + epoch + 1}/{args.epochs} ({elapsed:.0f}s) | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f}",
              flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": head_epochs + epoch,
                "val_acc": val_acc,
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
            }, MODELS_DIR / "best_square_classifier.pt")
            print(f"  -> New best! {val_acc:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 5:
                print(f"  Early stopping: no improvement for 5 epochs", flush=True)
                break

    print(f"\n{'=' * 60}")
    print(f"Done! Best val accuracy: {best_val_acc:.3f}")

    # Per-class accuracy
    print("\nPer-class accuracy (final epoch):")
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    for cls_idx in sorted(class_total.keys()):
        cls_name = idx_to_class.get(cls_idx, f"class_{cls_idx}")
        acc = class_correct.get(cls_idx, 0) / class_total[cls_idx]
        print(f"  {cls_name}: {acc:.3f} ({class_correct.get(cls_idx, 0)}/{class_total[cls_idx]})")

    # Export to ONNX
    print("\nExporting to ONNX...")
    best_ckpt = torch.load(MODELS_DIR / "best_square_classifier.pt",
                           map_location="cpu", weights_only=True)
    export_model = build_model(num_classes=num_classes)
    export_model.load_state_dict(best_ckpt["model_state_dict"])
    export_model.eval()
    dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
    onnx_path = MODELS_DIR / "square_classifier.onnx"
    torch.onnx.export(
        export_model, dummy, str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported to {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
