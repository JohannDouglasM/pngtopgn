#!/usr/bin/env python3
"""Quick evaluation of the best square classifier checkpoint on val set."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "squares"
MODELS_DIR = BASE_DIR / "models"

IMG_WIDTH = 100
IMG_HEIGHT = 200

def build_model(num_classes=13):
    import os
    model = models.resnet18(weights=None)
    cached = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    if os.path.exists(cached):
        state = torch.load(cached, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    n = model.fc.in_features
    model.fc = nn.Linear(n, num_classes)
    return model

def main():
    val_dir = DATA_DIR / "val"
    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    ckpt = torch.load(MODELS_DIR / "best_square_classifier.pt", map_location="cpu", weights_only=True)
    print(f"Checkpoint: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    print(f"Classes: {ckpt['classes']}")

    num_classes = len(ckpt['classes'])
    model = build_model(num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Per-class stats
    class_correct = {}
    class_total = {}
    # Confusion: (true, pred) -> count
    confusion = {}

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            for pred, label in zip(predicted, labels):
                t, p = label.item(), pred.item()
                class_total[t] = class_total.get(t, 0) + 1
                if t == p:
                    class_correct[t] = class_correct.get(t, 0) + 1
                else:
                    key = (t, p)
                    confusion[key] = confusion.get(key, 0) + 1

    idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    print(f"\nOverall: {total_correct}/{total_samples} = {total_correct/total_samples:.3f}")

    print("\nPer-class accuracy:")
    for cls_idx in sorted(class_total.keys()):
        name = idx_to_class[cls_idx]
        correct = class_correct.get(cls_idx, 0)
        total = class_total[cls_idx]
        print(f"  {name:20s}: {correct:5d}/{total:5d} = {correct/total:.3f}")

    print("\nTop confusions (true -> predicted: count):")
    sorted_conf = sorted(confusion.items(), key=lambda x: -x[1])[:20]
    for (t, p), count in sorted_conf:
        t_name = idx_to_class[t]
        p_name = idx_to_class[p]
        pct = count / class_total[t] * 100
        print(f"  {t_name:20s} -> {p_name:20s}: {count:4d} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
