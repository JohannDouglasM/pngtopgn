#!/usr/bin/env python3
"""
Train a per-square piece classifier v2 — EfficientNet-V2-S with stronger augmentation.

Key changes from v1:
- EfficientNet-V2-S backbone (better features, more capacity)
- RandAugment + strong color jitter for domain randomization
- Label smoothing to prevent overconfidence
- Cosine annealing with warm restarts

Usage:
    python3 training/train_squares_v2.py [--epochs 30] [--batch-size 32]
"""

import argparse
import math
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
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
        transforms.RandomGrayscale(p=0.15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
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
    """EfficientNet-V2-S with custom head for piece classification."""
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    n = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(n, num_classes),
    )
    return model


def train_epoch(model, loader, criterion, optimizer, device, epoch_str=""):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    n_batches = len(loader)
    log_every = max(1, n_batches // 4)

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

    with torch.no_grad():
        for images, labels in loader:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    if not train_dir.exists():
        print(f"Training data not found at {train_dir}")
        sys.exit(1)

    device = get_device()
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} squares")
    print(f"Val: {len(val_dataset)} squares")

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
    model = build_model(num_classes=num_classes).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EfficientNet-V2-S ({total_params/1e6:.1f}M params, {trainable_params/1e6:.1f}M trainable)")

    # Class weights
    class_counts = []
    for cls_name in train_dataset.classes:
        d = train_dir / cls_name
        class_counts.append(len(list(d.glob("*"))))
    median_count = sorted(class_counts)[len(class_counts) // 2]
    weights = [math.sqrt(median_count / max(c, 1)) for c in class_counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"\nClass weights (sqrt-inv-freq):")
    for name, w in zip(train_dataset.classes, weights):
        print(f"  {name}: {w:.3f}")

    # Label smoothing to prevent overconfidence on training domain
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

    # Phase 1: freeze backbone, train head
    head_epochs = 3
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0

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
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f}", flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "model_type": "efficientnet_v2_s",
            }, MODELS_DIR / "best_square_classifier_v2.pt")
            print(f"  -> New best! {val_acc:.3f}")

    # Phase 2: fine-tune all layers
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    remaining = args.epochs - head_epochs
    print(f"\nPhase 2: Fine-tuning all layers for {remaining} epochs (early stop after 7 no-improve)")
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
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": head_epochs + epoch,
                "val_acc": val_acc,
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "model_type": "efficientnet_v2_s",
            }, MODELS_DIR / "best_square_classifier_v2.pt")
            print(f"  -> New best! {val_acc:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 7:
                print(f"  Early stopping: no improvement for 7 epochs", flush=True)
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
    best_ckpt = torch.load(MODELS_DIR / "best_square_classifier_v2.pt",
                           map_location="cpu", weights_only=True)
    export_model = build_model(num_classes=num_classes)
    export_model.load_state_dict(best_ckpt["model_state_dict"])
    export_model.eval()
    dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
    onnx_path = MODELS_DIR / "square_classifier_v2.onnx"
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
