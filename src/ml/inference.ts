/**
 * Chess piece inference pipeline.
 *
 * Full pipeline:
 * 1. Load image as tensor
 * 2. Use perspective transform to locate 64 squares
 * 3. Extract each square as a 50x50 crop
 * 4. Run occupancy model → is the square empty?
 * 5. Run piece model on occupied squares → which piece?
 * 6. Assemble into SquareResult[]
 */

import * as tf from "@tensorflow/tfjs";
import { decodeJpeg } from "@tensorflow/tfjs-react-native";
import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import { File } from "expo-file-system";
import { SquareResult, PieceType } from "../chess/fenBuilder";
import { Point, computeHomography, applyHomography } from "../vision/perspective";
import { getOccupancyModel, getPieceModel, SQUARE_SIZE } from "./modelLoader";

export const PIECE_LABELS: PieceType[] = [
  "P", "N", "B", "R", "Q", "K",
  "p", "n", "b", "r", "q", "k",
];

/**
 * Run the full inference pipeline on an image with selected corners.
 *
 * imageUri: local file URI of the chess board photo
 * corners: [a1, a8, h8, h1] in normalized (0-1) image coordinates
 */
export async function runFullPipeline(
  imageUri: string,
  corners: Point[]
): Promise<SquareResult[]> {
  const occupancyModel = getOccupancyModel();
  const pieceModel = getPieceModel();

  if (!occupancyModel || !pieceModel) {
    throw new Error("Models not loaded. Call initModels() first.");
  }

  // 1. Load image as tensor
  const imageTensor = await loadImageAsTensor(imageUri);
  const [imgHeight, imgWidth] = imageTensor.shape;

  // 2. Compute homography: maps unit-square coords → image pixel coords
  const dst: Point[] = [
    { x: 0, y: 1 }, // a1 → bottom-left of unit square
    { x: 0, y: 0 }, // a8 → top-left
    { x: 1, y: 0 }, // h8 → top-right
    { x: 1, y: 1 }, // h1 → bottom-right
  ];
  const srcPixels = corners.map((c) => ({
    x: c.x * imgWidth,
    y: c.y * imgHeight,
  }));
  const H = computeHomography(dst, srcPixels);

  // 3. Extract 64 squares and batch into tensors
  const squareTensors: tf.Tensor4D[] = [];
  const squareCoords: { row: number; col: number }[] = [];

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const crop = extractSquareTensor(imageTensor, H, row, col, imgWidth, imgHeight);
      squareTensors.push(crop);
      squareCoords.push({ row, col });
    }
  }

  // 4. Batch all squares: [64, SQUARE_SIZE, SQUARE_SIZE, 3]
  const batch = tf.concat(squareTensors, 0);

  // 5. Run occupancy model
  const occPred = occupancyModel.predict(batch) as tf.Tensor;
  const occValues = await occPred.data();
  occPred.dispose();

  // 6. Run piece model on all squares (filter later)
  const piecePred = pieceModel.predict(batch) as tf.Tensor;
  const pieceValues = await piecePred.array() as number[][];
  piecePred.dispose();

  // 7. Assemble results
  const results: SquareResult[] = [];
  const OCCUPANCY_THRESHOLD = 0.5;

  for (let i = 0; i < 64; i++) {
    const { row, col } = squareCoords[i];
    const occupied = occValues[i] > OCCUPANCY_THRESHOLD;

    let piece: PieceType | null = null;
    if (occupied) {
      const probs = pieceValues[i];
      const maxIdx = probs.indexOf(Math.max(...probs));
      piece = PIECE_LABELS[maxIdx];
    }

    results.push({ row, col, piece });
  }

  // Cleanup
  imageTensor.dispose();
  batch.dispose();
  squareTensors.forEach((t) => t.dispose());

  return results;
}

/**
 * Load a local image file as a tf.Tensor3D [height, width, 3].
 */
async function loadImageAsTensor(uri: string): Promise<tf.Tensor3D> {
  // Convert to JPEG if needed
  const manipulated = await manipulateAsync(uri, [], {
    format: SaveFormat.JPEG,
    compress: 0.9,
  });

  // Read file bytes using the new expo-file-system File API
  const file = new File(manipulated.uri);
  const buffer = await file.arrayBuffer();
  const rawBytes = new Uint8Array(buffer);

  const imageTensor = decodeJpeg(rawBytes, 3);
  return imageTensor as tf.Tensor3D;
}

/**
 * Extract a single square from the image tensor using the homography.
 * Samples pixels at the mapped locations using nearest-neighbor.
 *
 * Returns a [1, SQUARE_SIZE, SQUARE_SIZE, 3] tensor normalized to [0, 1].
 */
function extractSquareTensor(
  imageTensor: tf.Tensor3D,
  H: number[],
  row: number,
  col: number,
  imgWidth: number,
  imgHeight: number
): tf.Tensor4D {
  return tf.tidy(() => {
    const pixels: number[][][] = [];

    for (let py = 0; py < SQUARE_SIZE; py++) {
      const rowPixels: number[][] = [];
      for (let px = 0; px < SQUARE_SIZE; px++) {
        // Map from square-local coords to unit-square coords
        const u = (col + px / SQUARE_SIZE) / 8;
        const v = (row + py / SQUARE_SIZE) / 8;

        // Apply homography to get image pixel coords
        const imgPt = applyHomography(H, { x: u, y: v });
        const ix = Math.min(Math.max(Math.round(imgPt.x), 0), imgWidth - 1);
        const iy = Math.min(Math.max(Math.round(imgPt.y), 0), imgHeight - 1);

        // We'll gather these coordinates below
        rowPixels.push([iy, ix]);
      }
      pixels.push(rowPixels);
    }

    // Flatten indices for gather
    const flatIndices: number[][] = [];
    for (let py = 0; py < SQUARE_SIZE; py++) {
      for (let px = 0; px < SQUARE_SIZE; px++) {
        flatIndices.push(pixels[py][px]);
      }
    }

    const indices = tf.tensor2d(flatIndices, [SQUARE_SIZE * SQUARE_SIZE, 2], "int32");
    const gathered = tf.gatherND(imageTensor, indices); // [SQUARE_SIZE*SQUARE_SIZE, 3]
    indices.dispose();

    const reshaped = gathered.reshape([1, SQUARE_SIZE, SQUARE_SIZE, 3]);
    const normalized = reshaped.div(255.0) as tf.Tensor4D;
    return normalized;
  });
}
