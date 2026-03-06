#!/usr/bin/env python3
"""
Convert chesscog PyTorch model checkpoints to ONNX format.

The .pt files contain full model objects (not just state_dicts),
so we need to create chesscog-compatible stub classes and load
with weights_only=False.

Models:
  - Occupancy: ResNet-18, 100x100 input, 2 classes (empty/occupied)
  - Pieces: InceptionV3, 299x299 input, 12 classes
"""

import sys
import os
import types

import torch
import torch.nn as nn
import torchvision.models as models

MODELS_DIR = "/tmp/chesscog_models"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "models")


def create_chesscog_stubs():
    """Create stub modules so torch.load can unpickle the chesscog model objects."""

    # Create the chesscog module hierarchy
    chesscog = types.ModuleType("chesscog")
    chesscog.occupancy_classifier = types.ModuleType("chesscog.occupancy_classifier")
    chesscog.occupancy_classifier.models = types.ModuleType("chesscog.occupancy_classifier.models")
    chesscog.piece_classifier = types.ModuleType("chesscog.piece_classifier")
    chesscog.piece_classifier.models = types.ModuleType("chesscog.piece_classifier.models")

    # Occupancy classifier: ResNet-18 with 2-class output
    class ResNet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            base = models.resnet18(weights=None)
            base.fc = nn.Linear(base.fc.in_features, 2)
            self.model = base

        def forward(self, x):
            return self.model(x)

    # Piece classifier: InceptionV3 with 12-class output
    class InceptionV3(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            base = models.inception_v3(weights=None, aux_logits=True, init_weights=False)
            base.fc = nn.Linear(base.fc.in_features, 12)
            base.AuxLogits.fc = nn.Linear(base.AuxLogits.fc.in_features, 12)
            self.model = base

        def forward(self, x):
            return self.model(x)

    chesscog.occupancy_classifier.models.ResNet = ResNet
    chesscog.piece_classifier.models.InceptionV3 = InceptionV3

    # Register in sys.modules
    sys.modules["chesscog"] = chesscog
    sys.modules["chesscog.occupancy_classifier"] = chesscog.occupancy_classifier
    sys.modules["chesscog.occupancy_classifier.models"] = chesscog.occupancy_classifier.models
    sys.modules["chesscog.piece_classifier"] = chesscog.piece_classifier
    sys.modules["chesscog.piece_classifier.models"] = chesscog.piece_classifier.models


def convert_occupancy():
    print("Converting occupancy model (ResNet-18, 100x100, 2 classes)...")
    weights_path = os.path.join(MODELS_DIR, "occupancy", "ResNet.pt")
    model = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.eval()

    # The loaded model might be wrapped - get the actual module
    if hasattr(model, "model"):
        export_model = model.model
    else:
        export_model = model

    dummy = torch.randn(1, 3, 100, 100)
    out_path = os.path.join(OUTPUT_DIR, "occupancy.onnx")

    torch.onnx.export(
        export_model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")


def convert_pieces():
    print("Converting piece model (InceptionV3, 299x299, 12 classes)...")
    weights_path = os.path.join(MODELS_DIR, "pieces", "InceptionV3.pt")
    model = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.eval()

    if hasattr(model, "model"):
        export_model = model.model
    else:
        export_model = model

    dummy = torch.randn(1, 3, 299, 299)
    out_path = os.path.join(OUTPUT_DIR, "pieces.onnx")

    torch.onnx.export(
        export_model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_chesscog_stubs()

    try:
        convert_occupancy()
    except Exception as e:
        print(f"  Error converting occupancy: {e}")
        import traceback; traceback.print_exc()

    try:
        convert_pieces()
    except Exception as e:
        print(f"  Error converting pieces: {e}")
        import traceback; traceback.print_exc()

    print("\nDone!")
