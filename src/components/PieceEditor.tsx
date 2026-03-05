import { View, Text, TouchableOpacity, StyleSheet, Modal } from "react-native";
import { PieceType } from "../chess/fenBuilder";

const PIECE_OPTIONS: { piece: PieceType | null; label: string }[] = [
  { piece: null, label: "Empty" },
  { piece: "K", label: "♔" },
  { piece: "Q", label: "♕" },
  { piece: "R", label: "♖" },
  { piece: "B", label: "♗" },
  { piece: "N", label: "♘" },
  { piece: "P", label: "♙" },
  { piece: "k", label: "♚" },
  { piece: "q", label: "♛" },
  { piece: "r", label: "♜" },
  { piece: "b", label: "♝" },
  { piece: "n", label: "♞" },
  { piece: "p", label: "♟" },
];

type Props = {
  visible: boolean;
  squareLabel: string;
  onSelect: (piece: PieceType | null) => void;
  onClose: () => void;
};

export default function PieceEditor({ visible, squareLabel, onSelect, onClose }: Props) {
  return (
    <Modal visible={visible} transparent animationType="slide">
      <View style={styles.overlay}>
        <View style={styles.sheet}>
          <Text style={styles.title}>Edit {squareLabel}</Text>

          <View style={styles.section}>
            <Text style={styles.sectionLabel}>White</Text>
            <View style={styles.pieceRow}>
              {PIECE_OPTIONS.filter((p) => p.piece && p.piece === p.piece?.toUpperCase()).map((p) => (
                <TouchableOpacity
                  key={p.piece}
                  style={styles.pieceButton}
                  onPress={() => onSelect(p.piece)}
                >
                  <Text style={styles.pieceText}>{p.label}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionLabel}>Black</Text>
            <View style={styles.pieceRow}>
              {PIECE_OPTIONS.filter((p) => p.piece && p.piece === p.piece?.toLowerCase()).map((p) => (
                <TouchableOpacity
                  key={p.piece}
                  style={styles.pieceButton}
                  onPress={() => onSelect(p.piece)}
                >
                  <Text style={styles.pieceText}>{p.label}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          <TouchableOpacity style={styles.emptyButton} onPress={() => onSelect(null)}>
            <Text style={styles.emptyText}>Set Empty</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.cancelButton} onPress={onClose}>
            <Text style={styles.cancelText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "flex-end",
  },
  sheet: {
    backgroundColor: "#1a1a2e",
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 24,
    paddingBottom: 40,
  },
  title: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: 20,
  },
  section: {
    marginBottom: 16,
  },
  sectionLabel: {
    color: "#a8a8b8",
    fontSize: 14,
    marginBottom: 8,
  },
  pieceRow: {
    flexDirection: "row",
    gap: 8,
    justifyContent: "center",
  },
  pieceButton: {
    width: 48,
    height: 48,
    backgroundColor: "#0f3460",
    borderRadius: 10,
    justifyContent: "center",
    alignItems: "center",
  },
  pieceText: {
    fontSize: 28,
  },
  emptyButton: {
    backgroundColor: "#333",
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 8,
  },
  emptyText: {
    color: "#fff",
    fontSize: 16,
  },
  cancelButton: {
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 8,
  },
  cancelText: {
    color: "#e94560",
    fontSize: 16,
  },
});
