#!/usr/bin/env python3
"""
Train end-to-end chess board recognition model.

Architecture: EfficientNetV2-S → Linear(1280, 64*13)
Input: 400x400 full board photo → Output: 64 squares × 13 piece classes

Usage:
    python3 training/train.py --dataset chessred2k --epochs 30
    python3 training/train.py --dataset chessred2k --resume training/models/best_model.pt
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

sys.path.insert(0, str(Path(__file__).parent))
from model import (ChessRecognitionModel, NUM_CLASSES,
                   target_to_fen, load_fenify_weights)
from dataset import ChessReD2KDataset, ChessReDFullDataset, get_transforms

MODELS_DIR = Path(__file__).parent / "models"


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


def train_epoch(model, loader, criterion, optimizer, device, epoch_str=""):
    model.train()
    total_loss = 0
    correct_squares = 0
    total_squares = 0
    correct_boards = 0
    total_boards = 0
    n_batches = len(loader)
    log_every = max(1, n_batches // 4)

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)  # (B, 64, 13)
        loss = criterion(logits.view(-1, NUM_CLASSES), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=2)
        correct_squares += (preds == targets).sum().item()
        total_squares += targets.numel()
        board_correct = (preds == targets).all(dim=1)
        correct_boards += board_correct.sum().item()
        total_boards += images.size(0)

        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            pct = (i + 1) / n_batches * 100
            sq_acc = correct_squares / total_squares
            print(f"  {epoch_str} train {pct:5.1f}% | batch {i+1}/{n_batches} | "
                  f"loss={total_loss/total_boards:.4f} sq_acc={sq_acc:.3f}",
                  flush=True)

    return (total_loss / total_boards,
            correct_squares / total_squares,
            correct_boards / total_boards)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_squares = 0
    total_squares = 0
    correct_boards = 0
    total_boards = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits.view(-1, NUM_CLASSES), targets.view(-1))

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=2)
            correct_squares += (preds == targets).sum().item()
            total_squares += targets.numel()
            board_correct = (preds == targets).all(dim=1)
            correct_boards += board_correct.sum().item()
            total_boards += images.size(0)

            for p, t in zip(preds.view(-1), targets.view(-1)):
                cls = t.item()
                class_total[cls] = class_total.get(cls, 0) + 1
                if p.item() == cls:
                    class_correct[cls] = class_correct.get(cls, 0) + 1

    return (total_loss / total_boards,
            correct_squares / total_squares,
            correct_boards / total_boards,
            class_correct, class_total)


def main():
    parser = argparse.ArgumentParser(description="Train chess board recognition model")
    parser.add_argument("--dataset", choices=["chessred2k", "chessred"], default="chessred2k")
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--images-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s",
                        choices=["efficientnet_v2_s", "resnet50", "resnet18"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    training_dir = Path(__file__).parent
    if args.annotations:
        annotations_path = args.annotations
    else:
        for c in [training_dir / "data" / "annotations.json",
                  training_dir.parent / "annotations.json"]:
            if c.exists():
                annotations_path = str(c)
                break
        else:
            print("Could not find annotations.json")
            sys.exit(1)

    if args.images_root:
        images_root = args.images_root
    else:
        for c in [training_dir / "data" / "chessred2k",
                  training_dir / "data" / "chessred",
                  training_dir / "data",
                  training_dir.parent]:
            if (c / "images").exists():
                images_root = str(c)
                break
        else:
            print("Could not find images/ directory")
            sys.exit(1)

    print(f"Annotations: {annotations_path}")
    print(f"Images root: {images_root}")
    print(f"Dataset: {args.dataset}")

    device = get_device()

    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    DatasetClass = ChessReD2KDataset if args.dataset == "chessred2k" else ChessReDFullDataset
    train_dataset = DatasetClass(annotations_path, images_root,
                                 split="train", transform=train_transform)
    val_dataset = DatasetClass(annotations_path, images_root,
                               split="val", transform=val_transform)

    if len(train_dataset) == 0:
        print("No training samples found!")
        sys.exit(1)

    num_workers = 0 if device.type == "mps" else 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=False)

    print(f"\nTrain: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")

    model = ChessRecognitionModel(backbone=args.backbone, pretrained=True)

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)

    # Compute class weights from training data
    # Count class frequencies across all training targets
    print("\nCounting class frequencies...")
    class_counts = [0] * NUM_CLASSES
    for i in range(len(train_dataset)):
        _, target = train_dataset[i]
        for sq in range(64):
            class_counts[target[sq].item()] += 1
    total_samples = sum(class_counts)
    median_count = sorted(class_counts)[len(class_counts) // 2]
    weights = [math.sqrt(median_count / max(c, 1)) for c in class_counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    CLASS_NAMES = ['empty', 'white_pawn', 'white_knight', 'white_bishop',
                   'white_rook', 'white_queen', 'white_king',
                   'black_pawn', 'black_knight', 'black_bishop',
                   'black_rook', 'black_queen', 'black_king']
    print("Class weights:")
    for name, c, w in zip(CLASS_NAMES, class_counts, weights):
        print(f"  {name:20s}: count={c:6d} weight={w:.3f}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Differential LR: lower for backbone, higher for head
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    if args.resume:
        # Full fine-tune from start when resuming
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=0.01)
        head_epochs = 0
    else:
        # Phase 1: freeze backbone, train head only
        head_epochs = min(3, args.epochs)
        for param in backbone_params:
            param.requires_grad = False
        optimizer = optim.Adam(head_params, lr=args.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - head_epochs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_sq_acc = 0

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nTraining {args.backbone} for {args.epochs} epochs on {device}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Parameters: {param_count:,}")
    print("=" * 70, flush=True)

    no_improve = 0

    # Phase 1: head only
    if head_epochs > 0:
        print(f"\nPhase 1: Training head only for {head_epochs} epochs")
        for epoch in range(head_epochs):
            start = time.time()
            epoch_str = f"[{epoch+1}/{args.epochs}]"
            train_loss, train_sq, train_bd = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch_str)
            val_loss, val_sq, val_bd, cc, ct = validate(
                model, val_loader, criterion, device)
            elapsed = time.time() - start

            print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) | "
                  f"Train: loss={train_loss:.3f} sq={train_sq:.3f} | "
                  f"Val: loss={val_loss:.3f} sq={val_sq:.3f} bd={val_bd:.3f}",
                  flush=True)

            if val_sq > best_val_sq_acc:
                best_val_sq_acc = val_sq
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_sq_acc": val_sq,
                    "val_bd_acc": val_bd,
                    "backbone": args.backbone,
                }, MODELS_DIR / "best_model.pt")
                print(f"  -> New best! sq={val_sq:.3f}", flush=True)

    # Phase 2: unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    if head_epochs > 0:
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - head_epochs)

    remaining = args.epochs - head_epochs
    print(f"\nPhase 2: Fine-tuning all layers for {remaining} epochs (early stop after 7 no-improve)")
    print("=" * 70, flush=True)

    for epoch in range(remaining):
        start = time.time()
        epoch_str = f"[{head_epochs + epoch + 1}/{args.epochs}]"
        train_loss, train_sq, train_bd = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch_str)
        val_loss, val_sq, val_bd, class_correct, class_total = validate(
            model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        print(f"Epoch {head_epochs+epoch+1}/{args.epochs} ({elapsed:.0f}s) | "
              f"Train: loss={train_loss:.3f} sq={train_sq:.3f} bd={train_bd:.3f} | "
              f"Val: loss={val_loss:.3f} sq={val_sq:.3f} bd={val_bd:.3f}",
              flush=True)

        if val_sq > best_val_sq_acc:
            best_val_sq_acc = val_sq
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": head_epochs + epoch,
                "val_sq_acc": val_sq,
                "val_bd_acc": val_bd,
                "backbone": args.backbone,
            }, MODELS_DIR / "best_model.pt")
            print(f"  -> New best! sq={val_sq:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 7:
                print(f"  Early stopping: no improvement for 7 epochs", flush=True)
                break

    print(f"\n{'='*70}")
    print(f"Done! Best square accuracy: {best_val_sq_acc:.3f}")

    # Per-class accuracy
    print("\nPer-class accuracy (final epoch):")
    for cls_idx in sorted(class_total.keys()):
        name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
        correct = class_correct.get(cls_idx, 0)
        total = class_total[cls_idx]
        print(f"  {name:20s}: {correct:5d}/{total:5d} = {correct/total:.3f}")

    # Export to ONNX
    print("\nExporting best model to ONNX...")
    best_ckpt = torch.load(MODELS_DIR / "best_model.pt", map_location="cpu", weights_only=True)
    export_model = ChessRecognitionModel(backbone=args.backbone, pretrained=False)
    export_model.load_state_dict(best_ckpt["model_state_dict"])
    export_model.eval()

    dummy = torch.randn(1, 3, 400, 400)
    onnx_path = MODELS_DIR / "chess_recognizer.onnx"
    torch.onnx.export(
        export_model, dummy, str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported to {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
