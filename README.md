# pngtopgn

React Native Expo app that recognizes chess positions from top-down photos and converts them to FEN/PGN. Everything runs on-device (no API calls).

## Pipeline

Photo → **Board Detection** (find 4 corners) → **Perspective Warp** → **Square Classification** (13 classes per square) → FEN → PGN/Lichess

## Project Structure

```
app/                     # Expo Router screens
  _layout.tsx            # Root layout
  index.tsx              # Camera/photo picker screen
  result.tsx             # Result display + FEN editing

src/
  ml/inference.ts        # On-device inference (ONNX Runtime)
                         #   - Corner detection model → 4 corner coords
                         #   - JS homography → warp + crop 64 squares
                         #   - Square classifier → piece per square
  chess/fenBuilder.ts    # Board array → FEN string
  chess/pgnExporter.ts   # FEN → PGN/Lichess URL
  components/
    ChessBoard.tsx       # Visual board display
    PieceEditor.tsx      # Manual piece correction UI

training/                # Python training & evaluation scripts
  detect_board_v5.py     # Board corner detection (CV-based, no ML)
                         #   - Multi-method binary image generation
                         #     (Canny, adaptive, OTSU, CLAHE, bilateral,
                         #      morphological gradient, HSV/LAB color channels,
                         #      brightness segmentation)
                         #   - Quadrilateral contour finding + minAreaRect fallback
                         #   - Checkerboard validation scoring (picks best candidate)
                         #   - Inner border refinement (variance + contour based)
                         #   - 100% detection on 600 images across 3 datasets
  eval_board_detection.py    # Evaluate detection across datasets
  inspect_failures.py        # Visual grid of failures/weak detections
  test_v5_pipeline.py        # End-to-end test: detection → classification → FEN
  train_corners.py           # Train ResNet-18 corner regression model
  train_squares.py           # Train ResNet-18 square classifier (13 classes)
  train_squares_v2.py        # V2 with sqrt class weights + strong augmentation
  prepare_squares.py         # Extract per-square crops from annotated images
                             #   (chesscog-style: variable height/width increase,
                             #    col<4 horizontal flip, bottom-aligned padding)
  prepare_squares_full.py    # Full dataset square preparation
  prepare_all_data.py        # Download + prepare all datasets
  download_datasets.py       # Dataset download helpers
  dataset.py                 # PyTorch dataset classes
  model.py                   # Model architecture definitions
  eval_checkpoint.py         # Evaluate a saved checkpoint
  evaluate.py                # General evaluation utilities
  debug_crops.py             # Debug square cropping visually
  annotate_corners.py        # Manual corner annotation tool
  add_user_images.py         # Add user photos to training data
  test_manual_corners.py     # Test with manually annotated corners
  requirements.txt           # Python dependencies (torch, opencv, etc.)

assets/                  # App assets (images, model files)
```

## Current State

### Board Detection (detect_board_v5.py) — STRONG
- **100% detection rate** on 600 images (Kaggle, ChessReD2K, ChessReD datasets)
- **97.2% good warps**, 0% bad warps
- Works on: tournament boards, wooden boards, vinyl mats, rotated boards, low contrast, busy backgrounds
- Algorithm: generate many binary images from different preprocessing methods → find all quad contours → score each by checkerboard pattern → pick best → refine to inner playing area
- No ML model needed — pure OpenCV

### Square Classifier — NEEDS WORK
- ResNet-18 trained on ChessReD2K tournament board squares
- 13 classes: 6 white pieces + 6 black pieces + empty
- ~92% accuracy on ChessReD2K validation set
- ~65% accuracy on other chess sets (domain gap problem)
- Crop style matters: must use exact chesscog-style cropping from prepare_squares.py

### On-Device Inference (src/ml/inference.ts)
- Uses onnxruntime-react-native for model execution
- jpeg-js for pure-JS pixel extraction
- JS homography implementation (no native OpenCV dependency)
- Requires Expo dev client build (not Expo Go)

## Models

Both exported as ONNX (~43MB each), stored in `assets/models/` (gitignored):
- **Corner model**: ResNet-18 → 8 floats (4 normalized corner coordinates). Input: 256x256
- **Square classifier**: ResNet-18 → 13 classes. Input: 100x200 (WxH)

## Training Data

All in `training/data/` (gitignored):
- **ChessReD2K**: 1442 train / 330 val / 306 test images with corner + piece annotations
- **ChessReD**: Similar format, different images
- **Kaggle**: Synthetic chess board images with JSON sidecar annotations
- Squares dataset: ~92k train / ~21k val crops extracted by prepare_squares.py

## Key Technical Decisions

1. **Two-stage pipeline** (detect corners → classify squares) instead of end-to-end model — more interpretable, each stage independently improvable
2. **Checkerboard validation** as candidate selection criterion — warps each candidate rectangle and checks for alternating light/dark pattern. This is what made detection robust
3. **No ML for board detection** — CV-based approach with many preprocessing methods is more generalizable than a trained corner regression model
4. **Chesscog-style square cropping** — variable height/width margins based on row/col position, horizontal flip for left-side columns, bottom-aligned padding. Must match exactly between training and inference
5. **On-device only** — no cloud APIs, all inference via ONNX Runtime

## Next Steps

1. **Improve square classifier generalization** — the main bottleneck. Options:
   - Train on more diverse datasets (different board/piece styles)
   - Domain adaptation / data augmentation
   - Fine-tune on user's specific board
2. Orientation detection (which way is the board facing)
3. Move suggestion / game analysis integration
