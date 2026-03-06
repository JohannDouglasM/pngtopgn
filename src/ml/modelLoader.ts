/**
 * Load ONNX models for chess piece recognition.
 *
 * Uses onnxruntime-react-native for on-device inference with
 * chesscog's pre-trained weights (ResNet-18 + InceptionV3).
 */

import { InferenceSession } from "onnxruntime-react-native";
import * as FileSystem from "expo-file-system";
import { Asset } from "expo-asset";

let isReady = false;
let occupancySession: InferenceSession | null = null;
let pieceSession: InferenceSession | null = null;

// Model input sizes (from chesscog configs)
export const OCCUPANCY_SIZE = 100; // ResNet-18: 100x100
export const PIECE_SIZE = 299; // InceptionV3: 299x299

// ImageNet normalization (used by chesscog)
export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

// Piece class labels (from chesscog's InceptionV3.yaml, alphabetical order)
export const PIECE_CLASSES = [
  "b", // black_bishop
  "k", // black_king
  "n", // black_knight
  "p", // black_pawn
  "q", // black_queen
  "r", // black_rook
  "B", // white_bishop
  "K", // white_king
  "N", // white_knight
  "P", // white_pawn
  "Q", // white_queen
  "R", // white_rook
] as const;

/**
 * Copy a bundled asset to a local file path that ONNX Runtime can read.
 */
async function getModelPath(assetModule: number): Promise<string> {
  const [asset] = await Asset.loadAsync(assetModule);
  if (asset.localUri) return asset.localUri;

  // If no localUri, download to cache
  const dest = `${FileSystem.Paths.cache.uri}/${asset.name}.${asset.type}`;
  const file = new FileSystem.File(dest);
  if (!file.exists) {
    await FileSystem.File.downloadFileAsync(asset.uri!, new FileSystem.Directory(FileSystem.Paths.cache.uri));
  }
  return dest;
}

/**
 * Initialize ONNX Runtime and load both models.
 */
export async function initModels(): Promise<void> {
  if (isReady) return;

  console.log("Loading ONNX models...");

  try {
    // Load occupancy model (ResNet-18, ~43MB)
    const occPath = await getModelPath(require("../../assets/models/occupancy.onnx"));
    occupancySession = await InferenceSession.create(occPath);
    console.log("Occupancy model loaded");

    // Load piece model (InceptionV3, ~83MB)
    const piecePath = await getModelPath(require("../../assets/models/pieces.onnx"));
    pieceSession = await InferenceSession.create(piecePath);
    console.log("Piece model loaded");

    isReady = true;
  } catch (err) {
    console.error("Failed to load ONNX models:", err);
    throw err;
  }
}

export function getOccupancySession(): InferenceSession | null {
  return occupancySession;
}

export function getPieceSession(): InferenceSession | null {
  return pieceSession;
}

export function isModelReady(): boolean {
  return isReady;
}
