/**
 * On-device chess board recognition using ONNX Runtime.
 *
 * Two-step pipeline:
 * 1. Corner detection: ResNet-18 predicts 4 board corners
 * 2. Square classification: ResNet-18 classifies each of 64 squares
 *
 * Perspective warp and square splitting done in JS.
 */

import { InferenceSession, Tensor } from "onnxruntime-react-native";
import { Asset } from "expo-asset";
import * as ImageManipulator from "expo-image-manipulator";
import * as FileSystem from "expo-file-system";
import { decode as decodeJpeg } from "jpeg-js";
import { SquareResult, PieceType } from "../chess/fenBuilder";

// Per-square model class ordering (alphabetical from ImageFolder)
const CLASS_TO_PIECE: (PieceType | null)[] = [
  "b", // 0: black_bishop
  "k", // 1: black_king
  "n", // 2: black_knight
  "p", // 3: black_pawn
  "q", // 4: black_queen
  "r", // 5: black_rook
  null, // 6: empty
  "B", // 7: white_bishop
  "K", // 8: white_king
  "N", // 9: white_knight
  "P", // 10: white_pawn
  "Q", // 11: white_queen
  "R", // 12: white_rook
];

// ImageNet normalization
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// Model input sizes
const CORNER_SIZE = 384; // Corner detector input
const SQUARE_H = 200; // Square classifier height
const SQUARE_W = 100; // Square classifier width

// Square crop parameters (must match prepare_squares.py exactly)
const SQUARE_SIZE = 50;
const BOARD_SIZE = 8 * SQUARE_SIZE; // 400
const IMG_SIZE = BOARD_SIZE * 2; // 800
const MARGIN = (IMG_SIZE - BOARD_SIZE) / 2; // 200
const MIN_HEIGHT_INCREASE = 1;
const MAX_HEIGHT_INCREASE = 3;
const MIN_WIDTH_INCREASE = 0.25;
const MAX_WIDTH_INCREASE = 1;
const OUT_WIDTH = Math.floor((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE); // 100
const OUT_HEIGHT = Math.floor((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE); // 200

let cornerSession: InferenceSession | null = null;
let squareSession: InferenceSession | null = null;

async function loadCornerModel(): Promise<InferenceSession> {
  if (cornerSession) return cornerSession;
  const [asset] = await Asset.loadAsync(
    require("../../assets/models/corner_detector.onnx")
  );
  if (!asset.localUri) throw new Error("Failed to load corner model");
  cornerSession = await InferenceSession.create(asset.localUri);
  return cornerSession;
}

async function loadSquareModel(): Promise<InferenceSession> {
  if (squareSession) return squareSession;
  const [asset] = await Asset.loadAsync(
    require("../../assets/models/square_classifier.onnx")
  );
  if (!asset.localUri) throw new Error("Failed to load square model");
  squareSession = await InferenceSession.create(asset.localUri);
  return squareSession;
}

/**
 * Convert a JPEG base64 string to a normalized CHW Float32Array tensor.
 */
function jpegBase64ToTensor(
  base64: string,
  targetW: number,
  targetH: number
): Float32Array {
  // Decode base64 to Uint8Array
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  // Decode JPEG to raw RGBA pixels
  const decoded = decodeJpeg(bytes, { useTArray: true, formatAsRGBA: true });
  const pixels = decoded.data; // Uint8Array of RGBA

  // The decoded dimensions should match our target (we pre-resized)
  const w = decoded.width;
  const h = decoded.height;

  // Convert to CHW format with ImageNet normalization
  const tensor = new Float32Array(3 * targetH * targetW);
  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      // Handle size mismatch with nearest-neighbor sampling
      const srcX = Math.min(Math.floor((x / targetW) * w), w - 1);
      const srcY = Math.min(Math.floor((y / targetH) * h), h - 1);
      const pixelIdx = (srcY * w + srcX) * 4;
      const tensorIdx = y * targetW + x;

      tensor[0 * targetH * targetW + tensorIdx] =
        (pixels[pixelIdx] / 255 - MEAN[0]) / STD[0];
      tensor[1 * targetH * targetW + tensorIdx] =
        (pixels[pixelIdx + 1] / 255 - MEAN[1]) / STD[1];
      tensor[2 * targetH * targetW + tensorIdx] =
        (pixels[pixelIdx + 2] / 255 - MEAN[2]) / STD[2];
    }
  }

  return tensor;
}

/**
 * Resize an image and get its base64 JPEG data.
 */
async function resizeToBase64(
  uri: string,
  width: number,
  height: number
): Promise<string> {
  const result = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width, height } }],
    { format: ImageManipulator.SaveFormat.JPEG, base64: true }
  );
  return result.base64!;
}

/**
 * Crop a region from an image and get its base64 JPEG data.
 */
async function cropAndResizeToBase64(
  uri: string,
  originX: number,
  originY: number,
  cropW: number,
  cropH: number,
  resizeW: number,
  resizeH: number
): Promise<string> {
  const result = await ImageManipulator.manipulateAsync(
    uri,
    [
      { crop: { originX, originY, width: cropW, height: cropH } },
      { resize: { width: resizeW, height: resizeH } },
    ],
    { format: ImageManipulator.SaveFormat.JPEG, base64: true }
  );
  return result.base64!;
}

/**
 * Apply a 3x3 perspective transform matrix to a point.
 */
function applyPerspective(
  H: number[],
  x: number,
  y: number
): [number, number] {
  const w = H[6] * x + H[7] * y + H[8];
  const px = (H[0] * x + H[1] * y + H[2]) / w;
  const py = (H[3] * x + H[4] * y + H[5]) / w;
  return [px, py];
}

/**
 * Compute a 3x3 perspective transform matrix from 4 source points to 4 dest points.
 * Uses the DLT (Direct Linear Transform) algorithm.
 */
function computeHomography(
  src: [number, number][],
  dst: [number, number][]
): number[] {
  // Build 8x8 system of equations
  const A: number[][] = [];
  const b: number[] = [];

  for (let i = 0; i < 4; i++) {
    const [sx, sy] = src[i];
    const [dx, dy] = dst[i];

    A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy]);
    b.push(dx);
    A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy]);
    b.push(dy);
  }

  // Solve using Gaussian elimination
  const n = 8;
  const aug: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col]);
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    // Eliminate
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  const h = new Array(9);
  for (let i = 0; i < 8; i++) {
    h[i] = aug[i][n] / aug[i][i];
  }
  h[8] = 1;

  return h;
}

/**
 * Step 1: Detect board corners using the corner detection model.
 * Returns 4 corners in pixel coordinates: [tl, tr, br, bl].
 */
async function detectCorners(
  imageUri: string,
  imgWidth: number,
  imgHeight: number,
  onProgress?: (done: number, total: number) => void
): Promise<[number, number][]> {
  const model = await loadCornerModel();
  onProgress?.(1, 68);

  // Resize to model input size and get pixels
  const base64 = await resizeToBase64(imageUri, CORNER_SIZE, CORNER_SIZE);
  const tensor = jpegBase64ToTensor(base64, CORNER_SIZE, CORNER_SIZE);

  const inputTensor = new Tensor("float32", tensor, [
    1,
    3,
    CORNER_SIZE,
    CORNER_SIZE,
  ]);
  const output = await model.run({ image: inputTensor });
  const corners = output.corners.data as Float32Array;

  // Corners are normalized [0,1] — scale to original image coordinates
  return [
    [corners[0] * imgWidth, corners[1] * imgHeight], // top-left
    [corners[2] * imgWidth, corners[3] * imgHeight], // top-right
    [corners[4] * imgWidth, corners[5] * imgHeight], // bottom-right
    [corners[6] * imgWidth, corners[7] * imgHeight], // bottom-left
  ];
}

/**
 * Step 2: Warp the board to a regular grid, then crop and classify each square.
 *
 * We compute a homography from the detected corners to a regular grid (matching
 * prepare_squares.py), then for each square, compute the inverse mapping to find
 * the source region in the original image. This matches the training crop logic
 * exactly: variable height_increase, width_increase, col<4 mirroring, and padding.
 */
async function classifySquares(
  imageUri: string,
  corners: [number, number][],
  imgWidth: number,
  imgHeight: number,
  onProgress?: (done: number, total: number) => void
): Promise<SquareResult[]> {
  const model = await loadSquareModel();
  onProgress?.(3, 68);

  // Sort corners: TL, TR, BR, BL (same as sort_corner_points in test_pipeline.py)
  const sorted = sortCornerPoints(corners);

  // Compute homography: original image corners → warped regular grid with margin
  // Destination: board occupies [MARGIN, MARGIN+BOARD_SIZE] in an IMG_SIZE x IMG_SIZE image
  const dst: [number, number][] = [
    [MARGIN, MARGIN], // TL
    [BOARD_SIZE + MARGIN, MARGIN], // TR
    [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN], // BR
    [MARGIN, BOARD_SIZE + MARGIN], // BL
  ];
  const H = computeHomography(sorted, dst);

  const results: SquareResult[] = [];

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const squareIdx = row * 8 + col;

      // Compute crop region in warped space (matching prepare_squares.py exactly)
      const heightIncrease =
        MIN_HEIGHT_INCREASE +
        (MAX_HEIGHT_INCREASE - MIN_HEIGHT_INCREASE) * ((7 - row) / 7);
      const leftIncrease =
        col >= 4
          ? 0
          : MIN_WIDTH_INCREASE +
            (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((3 - col) / 3);
      const rightIncrease =
        col < 4
          ? 0
          : MIN_WIDTH_INCREASE +
            (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((col - 4) / 3);

      const x1w = Math.max(
        0,
        Math.floor(MARGIN + SQUARE_SIZE * (col - leftIncrease))
      );
      const x2w = Math.min(
        IMG_SIZE,
        Math.floor(MARGIN + SQUARE_SIZE * (col + 1 + rightIncrease))
      );
      const y1w = Math.max(
        0,
        Math.floor(MARGIN + SQUARE_SIZE * (row - heightIncrease))
      );
      const y2w = Math.min(
        IMG_SIZE,
        Math.floor(MARGIN + SQUARE_SIZE * (row + 1))
      );

      // Map the 4 corners of the warped crop back to original image space
      // using inverse homography
      const Hinv = computeHomography(dst, sorted);
      const c1 = applyPerspective(Hinv, x1w, y1w);
      const c2 = applyPerspective(Hinv, x2w, y1w);
      const c3 = applyPerspective(Hinv, x2w, y2w);
      const c4 = applyPerspective(Hinv, x1w, y2w);

      const xs = [c1[0], c2[0], c3[0], c4[0]];
      const ys = [c1[1], c2[1], c3[1], c4[1]];

      let cropX = Math.max(0, Math.floor(Math.min(...xs)));
      let cropY = Math.max(0, Math.floor(Math.min(...ys)));
      let cropRight = Math.min(imgWidth, Math.ceil(Math.max(...xs)));
      let cropBottom = Math.min(imgHeight, Math.ceil(Math.max(...ys)));
      let cropW = Math.max(1, cropRight - cropX);
      let cropH = Math.max(1, cropBottom - cropY);

      try {
        const base64 = await cropAndResizeToBase64(
          imageUri,
          cropX,
          cropY,
          cropW,
          cropH,
          OUT_WIDTH,
          OUT_HEIGHT
        );

        // Decode to pixels for potential mirroring and padding
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const decoded = decodeJpeg(bytes, {
          useTArray: true,
          formatAsRGBA: true,
        });
        const pixels = decoded.data;
        const decW = decoded.width;
        const decH = decoded.height;

        // Build CHW tensor with mirroring for col<4 and bottom-aligned padding
        const tensor = new Float32Array(3 * SQUARE_H * SQUARE_W);
        const padTop = SQUARE_H - decH;

        for (let y = 0; y < SQUARE_H; y++) {
          for (let x = 0; x < SQUARE_W; x++) {
            const tensorIdx = y * SQUARE_W + x;
            const srcY = y - padTop;

            if (srcY < 0 || srcY >= decH || x >= decW) {
              // Black padding (after normalization)
              tensor[0 * SQUARE_H * SQUARE_W + tensorIdx] =
                (0 - MEAN[0]) / STD[0];
              tensor[1 * SQUARE_H * SQUARE_W + tensorIdx] =
                (0 - MEAN[1]) / STD[1];
              tensor[2 * SQUARE_H * SQUARE_W + tensorIdx] =
                (0 - MEAN[2]) / STD[2];
            } else {
              // Mirror horizontally for col < 4
              const srcX = col < 4 ? decW - 1 - x : x;
              const pixelIdx = (srcY * decW + srcX) * 4;

              tensor[0 * SQUARE_H * SQUARE_W + tensorIdx] =
                (pixels[pixelIdx] / 255 - MEAN[0]) / STD[0];
              tensor[1 * SQUARE_H * SQUARE_W + tensorIdx] =
                (pixels[pixelIdx + 1] / 255 - MEAN[1]) / STD[1];
              tensor[2 * SQUARE_H * SQUARE_W + tensorIdx] =
                (pixels[pixelIdx + 2] / 255 - MEAN[2]) / STD[2];
            }
          }
        }

        const inputTensor = new Tensor("float32", tensor, [
          1,
          3,
          SQUARE_H,
          SQUARE_W,
        ]);
        const output = await model.run({ image: inputTensor });
        const logits = output.logits.data as Float32Array;

        // Find max class
        let maxIdx = 0;
        let maxVal = logits[0];
        for (let i = 1; i < logits.length; i++) {
          if (logits[i] > maxVal) {
            maxVal = logits[i];
            maxIdx = i;
          }
        }

        results.push({ row, col, piece: CLASS_TO_PIECE[maxIdx] });
      } catch (e) {
        // If crop fails (e.g. out of bounds), mark as empty
        results.push({ row, col, piece: null });
      }

      onProgress?.(4 + squareIdx, 68);
    }
  }

  return results;
}

/**
 * Sort corner points: top-left, top-right, bottom-right, bottom-left.
 * Matches sort_corner_points in test_pipeline.py.
 */
function sortCornerPoints(
  corners: [number, number][]
): [number, number][] {
  // Sort by Y coordinate
  const sorted = [...corners].sort((a, b) => a[1] - b[1]);
  // Top two: sort by X
  const top = sorted.slice(0, 2).sort((a, b) => a[0] - b[0]);
  // Bottom two: sort by X descending (BR before BL)
  const bottom = sorted.slice(2).sort((a, b) => b[0] - a[0]);
  return [top[0], top[1], bottom[0], bottom[1]];
}

/**
 * Run the full 2-step inference pipeline.
 *
 * 1. Detect board corners
 * 2. Crop and classify each square
 */
export async function runFullPipeline(
  imageUri: string,
  onProgress?: (done: number, total: number) => void
): Promise<SquareResult[]> {
  onProgress?.(0, 68);

  // Get image dimensions
  const imageInfo = await ImageManipulator.manipulateAsync(imageUri, [], {
    format: ImageManipulator.SaveFormat.JPEG,
    base64: false,
  });
  const imgWidth = imageInfo.width;
  const imgHeight = imageInfo.height;

  // Step 1: Detect corners
  const corners = await detectCorners(
    imageUri,
    imgWidth,
    imgHeight,
    onProgress
  );

  onProgress?.(2, 68);

  // Step 2: Classify squares
  const results = await classifySquares(
    imageUri,
    corners,
    imgWidth,
    imgHeight,
    onProgress
  );

  onProgress?.(68, 68);
  return results;
}
