/**
 * Perspective transform utilities.
 *
 * Given 4 corner points of a chessboard in an image, compute a homography
 * matrix and warp coordinates so the board becomes a flat square.
 *
 * We use a standard 4-point perspective transform (DLT algorithm).
 */

export type Point = { x: number; y: number };

/**
 * Solve a 3x3 homography from 4 source → 4 destination point correspondences
 * using the Direct Linear Transform (DLT) method.
 *
 * source: the 4 corners the user tapped (normalized 0-1 in image space)
 * dest:   the 4 target corners of a unit square [(0,0),(0,1),(1,1),(1,0)]
 */
export function computeHomography(src: Point[], dst: Point[]): number[] {
  // Build the 8x9 matrix for DLT
  const A: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const { x: sx, y: sy } = src[i];
    const { x: dx, y: dy } = dst[i];
    A.push([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx]);
    A.push([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy]);
  }

  // Solve Ah = 0 using SVD approximation — for a 8x9 system we find
  // the null space. We use a simplified approach: solve the 8x8 system
  // by setting h9 = 1.
  const M: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 8; i++) {
    M.push(A[i].slice(0, 8));
    b.push(-A[i][8]);
  }

  const h = solveLinearSystem(M, b);
  return [...h, 1];
}

/**
 * Apply a 3x3 homography to a point.
 */
export function applyHomography(H: number[], p: Point): Point {
  const w = H[6] * p.x + H[7] * p.y + H[8];
  return {
    x: (H[0] * p.x + H[1] * p.y + H[2]) / w,
    y: (H[3] * p.x + H[4] * p.y + H[5]) / w,
  };
}

/**
 * Get the pixel coordinates of each of the 64 squares after warping.
 * Returns an 8x8 array of { row, col, srcQuad } where srcQuad gives
 * the 4 corners of that square in original image coordinates.
 *
 * corners: [a1, a8, h8, h1] — the 4 user-selected corners (normalized 0-1)
 * The board is mapped so rank 1 is at the bottom.
 */
export function getSquareRegions(
  corners: Point[],
  imageWidth: number,
  imageHeight: number
): { row: number; col: number; centerX: number; centerY: number }[] {
  // corners order: [a1(bottom-left), a8(top-left), h8(top-right), h1(bottom-right)]
  // Map to unit square: a8→(0,0), h8→(1,0), h1→(1,1), a1→(0,1)
  const dst: Point[] = [
    { x: 0, y: 1 }, // a1 → bottom-left of unit square
    { x: 0, y: 0 }, // a8 → top-left
    { x: 1, y: 0 }, // h8 → top-right
    { x: 1, y: 1 }, // h1 → bottom-right
  ];

  // Scale corners to pixel coordinates
  const srcPixels = corners.map((c) => ({
    x: c.x * imageWidth,
    y: c.y * imageHeight,
  }));

  const H = computeHomography(dst, srcPixels);

  const squares: { row: number; col: number; centerX: number; centerY: number }[] = [];

  // row 0 = rank 8 (top), row 7 = rank 1 (bottom)
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      // Center of this square in unit-square space
      const unitCenter: Point = {
        x: (col + 0.5) / 8,
        y: (row + 0.5) / 8,
      };
      const imgPoint = applyHomography(H, unitCenter);
      squares.push({
        row,
        col,
        centerX: imgPoint.x,
        centerY: imgPoint.y,
      });
    }
  }

  return squares;
}

// --- Linear algebra helpers ---

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  // Augmented matrix
  const aug = A.map((row, i) => [...row, b[i]]);

  // Gaussian elimination with partial pivoting
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

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-10) continue;

    for (let j = col; j <= n; j++) aug[col][j] /= pivot;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = col; j <= n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  return aug.map((row) => row[n]);
}
