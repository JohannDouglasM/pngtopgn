import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { PieceType } from "../chess/fenBuilder";

const PIECE_UNICODE: Record<PieceType, string> = {
  K: "♔",
  Q: "♕",
  R: "♖",
  B: "♗",
  N: "♘",
  P: "♙",
  k: "♚",
  q: "♛",
  r: "♜",
  b: "♝",
  n: "♞",
  p: "♟",
};

const FILE_LABELS = ["a", "b", "c", "d", "e", "f", "g", "h"];

type Props = {
  position: (PieceType | null)[][]; // 8x8, row 0 = rank 8
  onSquarePress?: (row: number, col: number) => void;
  selectedSquare?: { row: number; col: number } | null;
  size?: number;
};

export default function ChessBoard({ position, onSquarePress, selectedSquare, size = 320 }: Props) {
  const squareSize = size / 8;

  return (
    <View style={[styles.board, { width: size, height: size }]}>
      {position.map((rank, row) =>
        rank.map((piece, col) => {
          const isLight = (row + col) % 2 === 0;
          const isSelected = selectedSquare?.row === row && selectedSquare?.col === col;

          return (
            <TouchableOpacity
              key={`${row}-${col}`}
              style={[
                styles.square,
                {
                  width: squareSize,
                  height: squareSize,
                  backgroundColor: isSelected
                    ? "#e94560"
                    : isLight
                      ? "#f0d9b5"
                      : "#b58863",
                },
              ]}
              onPress={() => onSquarePress?.(row, col)}
              activeOpacity={0.7}
            >
              {piece && (
                <Text
                  style={[
                    styles.piece,
                    { fontSize: squareSize * 0.7 },
                  ]}
                >
                  {PIECE_UNICODE[piece]}
                </Text>
              )}
              {row === 7 && (
                <Text style={[styles.fileLabel, { color: isLight ? "#b58863" : "#f0d9b5" }]}>
                  {FILE_LABELS[col]}
                </Text>
              )}
              {col === 0 && (
                <Text style={[styles.rankLabel, { color: isLight ? "#b58863" : "#f0d9b5" }]}>
                  {8 - row}
                </Text>
              )}
            </TouchableOpacity>
          );
        })
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  board: {
    flexDirection: "row",
    flexWrap: "wrap",
    borderWidth: 2,
    borderColor: "#333",
    borderRadius: 4,
    overflow: "hidden",
  },
  square: {
    justifyContent: "center",
    alignItems: "center",
  },
  piece: {
    textAlign: "center",
  },
  fileLabel: {
    position: "absolute",
    bottom: 1,
    right: 3,
    fontSize: 9,
    fontWeight: "bold",
  },
  rankLabel: {
    position: "absolute",
    top: 1,
    left: 3,
    fontSize: 9,
    fontWeight: "bold",
  },
});
