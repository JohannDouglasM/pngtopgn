/**
 * Split a chessboard image into 64 individual square crops.
 *
 * Since we're in React Native without OpenCV, we use a sampling approach:
 * for each square, we extract a region around its center point (computed
 * via the perspective transform) from the original image tensor.
 */

import { Point, computeHomography, applyHomography } from "./perspective";

export type SquareCrop = {
  row: number; // 0=rank8, 7=rank1
  col: number; // 0=file-a, 7=file-h
  // The 4 corners of this square in original image pixel coordinates
  topLeft: Point;
  topRight: Point;
  bottomLeft: Point;
  bottomRight: Point;
  center: Point;
};

/**
 * Compute the image-space bounding regions for all 64 squares.
 *
 * corners: [a1, a8, h8, h1] in normalized (0-1) image coordinates
 */
export function computeSquareCrops(
  corners: Point[],
  imageWidth: number,
  imageHeight: number
): SquareCrop[] {
  // Map corners to pixel space
  const srcPixels = corners.map((c) => ({
    x: c.x * imageWidth,
    y: c.y * imageHeight,
  }));

  // Unit square destinations matching our corner order
  const dst: Point[] = [
    { x: 0, y: 1 }, // a1 → bottom-left
    { x: 0, y: 0 }, // a8 → top-left
    { x: 1, y: 0 }, // h8 → top-right
    { x: 1, y: 1 }, // h1 → bottom-right
  ];

  // Homography from unit square to image pixels
  const H = computeHomography(dst, srcPixels);

  const crops: SquareCrop[] = [];

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const unitTL: Point = { x: col / 8, y: row / 8 };
      const unitTR: Point = { x: (col + 1) / 8, y: row / 8 };
      const unitBL: Point = { x: col / 8, y: (row + 1) / 8 };
      const unitBR: Point = { x: (col + 1) / 8, y: (row + 1) / 8 };
      const unitCenter: Point = { x: (col + 0.5) / 8, y: (row + 0.5) / 8 };

      crops.push({
        row,
        col,
        topLeft: applyHomography(H, unitTL),
        topRight: applyHomography(H, unitTR),
        bottomLeft: applyHomography(H, unitBL),
        bottomRight: applyHomography(H, unitBR),
        center: applyHomography(H, unitCenter),
      });
    }
  }

  return crops;
}

/**
 * Extract pixel data for a square from an image data array.
 * Uses bilinear sampling around the square's mapped region.
 *
 * imageData: Uint8Array of RGBA pixels (width * height * 4)
 * crop: the square region to extract
 * outputSize: the target size (e.g., 100 for 100x100)
 *
 * Returns a Float32Array of RGB values normalized to [0, 1], shape [outputSize, outputSize, 3]
 */
export function extractSquarePixels(
  imageData: Uint8Array,
  imageWidth: number,
  imageHeight: number,
  crop: SquareCrop,
  outputSize: number
): Float32Array {
  const output = new Float32Array(outputSize * outputSize * 3);

  for (let outY = 0; outY < outputSize; outY++) {
    for (let outX = 0; outX < outputSize; outX++) {
      // Bilinear interpolation within the quad
      const u = outX / (outputSize - 1);
      const v = outY / (outputSize - 1);

      // Bilinear interpolation of the 4 corners
      const px =
        crop.topLeft.x * (1 - u) * (1 - v) +
        crop.topRight.x * u * (1 - v) +
        crop.bottomLeft.x * (1 - u) * v +
        crop.bottomRight.x * u * v;
      const py =
        crop.topLeft.y * (1 - u) * (1 - v) +
        crop.topRight.y * u * (1 - v) +
        crop.bottomLeft.y * (1 - u) * v +
        crop.bottomRight.y * u * v;

      // Clamp to image bounds
      const ix = Math.min(Math.max(Math.round(px), 0), imageWidth - 1);
      const iy = Math.min(Math.max(Math.round(py), 0), imageHeight - 1);

      const srcIdx = (iy * imageWidth + ix) * 4;
      const dstIdx = (outY * outputSize + outX) * 3;

      output[dstIdx] = (imageData[srcIdx] ?? 0) / 255;
      output[dstIdx + 1] = (imageData[srcIdx + 1] ?? 0) / 255;
      output[dstIdx + 2] = (imageData[srcIdx + 2] ?? 0) / 255;
    }
  }

  return output;
}
