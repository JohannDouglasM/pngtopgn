#!/usr/bin/env python3
"""
Convert chesscog PyTorch models to TFLite format for mobile inference.

Usage:
  1. Clone chesscog: git clone https://github.com/georg-wolflein/chesscog
  2. Download pre-trained weights (see chesscog README)
  3. Run this script: python convert_model.py --chesscog-dir ./chesscog --output-dir ../assets/models

Requirements:
  pip install torch torchvision onnx onnx-tf tensorflow
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def convert_occupancy_model(chesscog_dir: Path, output_dir: Path):
    """Convert the occupancy classifier (empty vs occupied)."""
    import torch
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # Load chesscog occupancy model
    sys.path.insert(0, str(chesscog_dir))
    from chesscog.occupancy_classifier.models import build_occupancy_classifier

    model = build_occupancy_classifier()
    weights_path = chesscog_dir / "runs" / "occupancy_classifier" / "model.pt"

    if not weights_path.exists():
        print(f"Warning: Weights not found at {weights_path}")
        print("Creating a dummy model for testing. Replace with real weights.")
        # Create dummy model for structure testing
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.eval()

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 100, 100)
    onnx_path = output_dir / "occupancy.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )
    print(f"Exported ONNX: {onnx_path}")

    # Convert ONNX → TF → TFLite
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_path = output_dir / "occupancy_tf"
    tf_rep.export_graph(str(tf_path))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_path = output_dir / "occupancy.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"Exported TFLite: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")

    # Cleanup intermediate files
    onnx_path.unlink()
    import shutil
    shutil.rmtree(tf_path, ignore_errors=True)


def convert_piece_model(chesscog_dir: Path, output_dir: Path):
    """Convert the piece classifier (12 classes: 6 pieces × 2 colors)."""
    import torch
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    sys.path.insert(0, str(chesscog_dir))
    from chesscog.piece_classifier.models import build_piece_classifier

    model = build_piece_classifier()
    weights_path = chesscog_dir / "runs" / "piece_classifier" / "model.pt"

    if not weights_path.exists():
        print(f"Warning: Weights not found at {weights_path}")
        print("Creating a dummy model for testing. Replace with real weights.")
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.eval()

    dummy_input = torch.randn(1, 3, 100, 100)
    onnx_path = output_dir / "pieces.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )
    print(f"Exported ONNX: {onnx_path}")

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_path = output_dir / "pieces_tf"
    tf_rep.export_graph(str(tf_path))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_path = output_dir / "pieces.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"Exported TFLite: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")

    onnx_path.unlink()
    import shutil
    shutil.rmtree(tf_path, ignore_errors=True)


def create_tfjs_model(output_dir: Path):
    """
    Alternative: Create a TF.js compatible model (JSON + weights.bin)
    that can be loaded directly with tf.loadGraphModel() in React Native.
    """
    import tensorflow as tf
    import tensorflowjs as tfjs

    # Simple CNN for occupancy (binary classifier)
    occupancy_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(100, 100, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    occupancy_model.compile(optimizer="adam", loss="binary_crossentropy")

    occ_dir = output_dir / "occupancy"
    occ_dir.mkdir(exist_ok=True)
    tfjs.converters.save_keras_model(occupancy_model, str(occ_dir))
    print(f"Exported TF.js occupancy model: {occ_dir}")

    # CNN for piece classification (12 classes)
    piece_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(100, 100, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(12, activation="softmax"),
    ])
    piece_model.compile(optimizer="adam", loss="categorical_crossentropy")

    piece_dir = output_dir / "pieces"
    piece_dir.mkdir(exist_ok=True)
    tfjs.converters.save_keras_model(piece_model, str(piece_dir))
    print(f"Exported TF.js piece model: {piece_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert chess models for mobile")
    parser.add_argument("--chesscog-dir", type=Path, help="Path to chesscog repo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "assets" / "models",
    )
    parser.add_argument(
        "--format",
        choices=["tflite", "tfjs"],
        default="tfjs",
        help="Output format (tfjs recommended for React Native)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "tflite":
        if not args.chesscog_dir:
            print("Error: --chesscog-dir required for TFLite conversion")
            sys.exit(1)
        convert_occupancy_model(args.chesscog_dir, args.output_dir)
        convert_piece_model(args.chesscog_dir, args.output_dir)
    else:
        create_tfjs_model(args.output_dir)

    print("\nDone! Models saved to:", args.output_dir)


if __name__ == "__main__":
    main()
