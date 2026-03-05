/**
 * Load and manage TF.js models for chess piece recognition.
 *
 * Creates functional CNN models on-device. When real chesscog weights
 * are converted (via scripts/convert_model.py), swap in the bundled
 * model files instead of the in-memory models below.
 */

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";

let isReady = false;
let occupancyModel: tf.LayersModel | null = null;
let pieceModel: tf.LayersModel | null = null;

const SQUARE_SIZE = 50; // Input size for models (50x50 px)

/**
 * Initialize TensorFlow.js backend and build/load models.
 */
export async function initModels(): Promise<void> {
  if (isReady) return;

  await tf.ready();
  console.log("TF.js backend:", tf.getBackend());

  // Build in-memory models (untrained — replace with real weights later)
  occupancyModel = buildOccupancyModel();
  pieceModel = buildPieceModel();

  isReady = true;
}

/**
 * Binary classifier: is a square occupied?
 * Input: [batch, 50, 50, 3]  Output: [batch, 1] (sigmoid)
 */
function buildOccupancyModel(): tf.LayersModel {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [SQUARE_SIZE, SQUARE_SIZE, 3],
    filters: 16,
    kernelSize: 3,
    activation: "relu",
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
  model.add(tf.layers.globalAveragePooling2d({}));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({ optimizer: "adam", loss: "binaryCrossentropy" });
  return model;
}

/**
 * 12-class classifier: which piece (6 types × 2 colors)?
 * Input: [batch, 50, 50, 3]  Output: [batch, 12] (softmax)
 *
 * Classes: P, N, B, R, Q, K, p, n, b, r, q, k
 */
function buildPieceModel(): tf.LayersModel {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [SQUARE_SIZE, SQUARE_SIZE, 3],
    filters: 32,
    kernelSize: 3,
    activation: "relu",
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" }));
  model.add(tf.layers.globalAveragePooling2d({}));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 12, activation: "softmax" }));
  model.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });
  return model;
}

export function getOccupancyModel(): tf.LayersModel | null {
  return occupancyModel;
}

export function getPieceModel(): tf.LayersModel | null {
  return pieceModel;
}

export function isModelReady(): boolean {
  return isReady;
}

export { SQUARE_SIZE };
