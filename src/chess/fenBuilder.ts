/**
 * Assemble per-square classification results into a FEN string.
 */

export type PieceType = "P" | "N" | "B" | "R" | "Q" | "K" | "p" | "n" | "b" | "r" | "q" | "k";

export type SquareResult = {
  row: number; // 0 = rank 8, 7 = rank 1
  col: number; // 0 = file a, 7 = file h
  piece: PieceType | null; // null = empty
};

/**
 * Build FEN position string from 64 square results.
 * Only the piece placement portion — no side-to-move, castling, etc.
 */
export function buildFenPosition(squares: SquareResult[]): string {
  // Sort by row then col
  const sorted = [...squares].sort((a, b) => a.row - b.row || a.col - b.col);

  const ranks: string[] = [];

  for (let row = 0; row < 8; row++) {
    let rank = "";
    let emptyCount = 0;
    for (let col = 0; col < 8; col++) {
      const sq = sorted.find((s) => s.row === row && s.col === col);
      if (!sq || sq.piece === null) {
        emptyCount++;
      } else {
        if (emptyCount > 0) {
          rank += emptyCount;
          emptyCount = 0;
        }
        rank += sq.piece;
      }
    }
    if (emptyCount > 0) rank += emptyCount;
    ranks.push(rank);
  }

  return ranks.join("/");
}

/**
 * Build a complete FEN string with default metadata.
 * Assumes it's white to move, all castling available, no en passant.
 */
export function buildFullFen(
  squares: SquareResult[],
  options?: {
    activeColor?: "w" | "b";
    castling?: string;
    enPassant?: string;
    halfmove?: number;
    fullmove?: number;
  }
): string {
  const position = buildFenPosition(squares);
  const color = options?.activeColor ?? "w";
  const castling = options?.castling ?? inferCastling(squares);
  const ep = options?.enPassant ?? "-";
  const half = options?.halfmove ?? 0;
  const full = options?.fullmove ?? 1;

  return `${position} ${color} ${castling} ${ep} ${half} ${full}`;
}

/**
 * Infer castling rights from piece positions.
 * If king and rook are on their starting squares, assume castling is available.
 */
function inferCastling(squares: SquareResult[]): string {
  const get = (row: number, col: number) =>
    squares.find((s) => s.row === row && s.col === col)?.piece;

  let castling = "";

  // White: king on e1 (row=7, col=4)
  if (get(7, 4) === "K") {
    if (get(7, 7) === "R") castling += "K";
    if (get(7, 0) === "R") castling += "Q";
  }
  // Black: king on e8 (row=0, col=4)
  if (get(0, 4) === "k") {
    if (get(0, 7) === "r") castling += "k";
    if (get(0, 0) === "r") castling += "q";
  }

  return castling || "-";
}

/**
 * Validate a FEN position for basic sanity.
 */
export function validateFen(squares: SquareResult[]): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  const pieces = squares.filter((s) => s.piece !== null);
  const whiteKings = pieces.filter((s) => s.piece === "K").length;
  const blackKings = pieces.filter((s) => s.piece === "k").length;
  const whitePieces = pieces.filter((s) => s.piece && s.piece === s.piece.toUpperCase()).length;
  const blackPieces = pieces.filter((s) => s.piece && s.piece === s.piece.toLowerCase()).length;

  if (whiteKings !== 1) errors.push(`Expected 1 white king, found ${whiteKings}`);
  if (blackKings !== 1) errors.push(`Expected 1 black king, found ${blackKings}`);
  if (whitePieces > 16) errors.push(`Too many white pieces: ${whitePieces}`);
  if (blackPieces > 16) errors.push(`Too many black pieces: ${blackPieces}`);

  // Pawns on first/last rank
  for (const sq of pieces) {
    if ((sq.piece === "P" || sq.piece === "p") && (sq.row === 0 || sq.row === 7)) {
      errors.push(`Pawn on rank ${sq.row === 0 ? 8 : 1} (${fileLabel(sq.col)}${rankLabel(sq.row)})`);
    }
  }

  return { valid: errors.length === 0, errors };
}

function fileLabel(col: number): string {
  return String.fromCharCode(97 + col);
}

function rankLabel(row: number): string {
  return String(8 - row);
}
