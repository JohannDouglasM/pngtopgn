import { useState, useEffect, useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  Dimensions,
  ActivityIndicator,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import * as Clipboard from "expo-clipboard";
import * as Haptics from "expo-haptics";
import * as Linking from "expo-linking";
import ChessBoard from "../src/components/ChessBoard";
import PieceEditor from "../src/components/PieceEditor";
import { PieceType, SquareResult, buildFullFen, validateFen } from "../src/chess/fenBuilder";
import { fenToPgn, lichessAnalysisUrl } from "../src/chess/pgnExporter";
import { runFullPipeline } from "../src/ml/inference";
import { initModels } from "../src/ml/modelLoader";
import { Point } from "../src/vision/perspective";

const FILE_LABELS = ["a", "b", "c", "d", "e", "f", "g", "h"];

export default function ResultScreen() {
  const { imageUri, corners } = useLocalSearchParams<{
    imageUri: string;
    corners: string;
  }>();

  const [position, setPosition] = useState<(PieceType | null)[][]>(
    Array.from({ length: 8 }, () => Array(8).fill(null))
  );
  const [loading, setLoading] = useState(true);
  const [loadingStatus, setLoadingStatus] = useState("Initializing models...");
  const [editSquare, setEditSquare] = useState<{ row: number; col: number } | null>(null);
  const [activeColor, setActiveColor] = useState<"w" | "b">("w");
  const [copied, setCopied] = useState<string | null>(null);

  const screenWidth = Dimensions.get("window").width;
  const boardSize = Math.min(screenWidth - 48, 360);

  useEffect(() => {
    runInference();
  }, []);

  const runInference = async () => {
    setLoading(true);
    try {
      // Parse corners from route params
      const parsedCorners: Point[] = JSON.parse(corners ?? "[]");
      if (parsedCorners.length !== 4 || !imageUri) {
        Alert.alert("Error", "Missing image or corner data.");
        setLoading(false);
        return;
      }

      // Initialize TF.js and models
      setLoadingStatus("Loading ML models...");
      await initModels();

      // Run the full pipeline: image → squares → inference → results
      setLoadingStatus("Analyzing squares (0/64)...");
      const results = await runFullPipeline(imageUri, parsedCorners, (done, total) => {
        setLoadingStatus(`Analyzing squares (${done}/${total})...`);
      });
      applyResults(results);
    } catch (err) {
      console.error("Inference error:", err);
      Alert.alert(
        "Error",
        `Failed to classify the chess position: ${err instanceof Error ? err.message : "Unknown error"}`
      );
    } finally {
      setLoading(false);
    }
  };

  const applyResults = (results: SquareResult[]) => {
    const board: (PieceType | null)[][] = Array.from({ length: 8 }, () => Array(8).fill(null));
    for (const sq of results) {
      board[sq.row][sq.col] = sq.piece;
    }
    setPosition(board);
  };

  const getSquareResults = useCallback((): SquareResult[] => {
    const results: SquareResult[] = [];
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        results.push({ row, col, piece: position[row][col] });
      }
    }
    return results;
  }, [position]);

  const fen = buildFullFen(getSquareResults(), { activeColor });
  const validation = validateFen(getSquareResults());
  const pgn = fenToPgn(fen);

  const copyToClipboard = async (text: string, label: string) => {
    await Clipboard.setStringAsync(text);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setCopied(label);
    setTimeout(() => setCopied(null), 2000);
  };

  const openInLichess = () => {
    const url = lichessAnalysisUrl(fen);
    Linking.openURL(url);
  };

  const handleSquarePress = (row: number, col: number) => {
    setEditSquare({ row, col });
  };

  const handlePieceSelect = (piece: PieceType | null) => {
    if (!editSquare) return;
    setPosition((prev) => {
      const next = prev.map((r) => [...r]);
      next[editSquare.row][editSquare.col] = piece;
      return next;
    });
    setEditSquare(null);
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#e94560" />
        <Text style={styles.loadingText}>{loadingStatus}</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.boardContainer}>
        <ChessBoard
          position={position}
          onSquarePress={handleSquarePress}
          selectedSquare={editSquare}
          size={boardSize}
        />
      </View>

      {!validation.valid && (
        <View style={styles.warningBox}>
          <Text style={styles.warningTitle}>Position warnings:</Text>
          {validation.errors.map((e, i) => (
            <Text key={i} style={styles.warningText}>
              • {e}
            </Text>
          ))}
        </View>
      )}

      <View style={styles.fenContainer}>
        <Text style={styles.label}>FEN</Text>
        <TouchableOpacity
          style={styles.fenBox}
          onPress={() => copyToClipboard(fen, "FEN")}
        >
          <Text style={styles.fenText} numberOfLines={2}>
            {fen}
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.toggleRow}>
        <Text style={styles.label}>To move:</Text>
        <TouchableOpacity
          style={[styles.toggleButton, activeColor === "w" && styles.toggleActive]}
          onPress={() => setActiveColor("w")}
        >
          <Text style={styles.toggleText}>White</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.toggleButton, activeColor === "b" && styles.toggleActive]}
          onPress={() => setActiveColor("b")}
        >
          <Text style={styles.toggleText}>Black</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => copyToClipboard(fen, "FEN")}
        >
          <Text style={styles.actionText}>
            {copied === "FEN" ? "Copied!" : "Copy FEN"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => copyToClipboard(pgn, "PGN")}
        >
          <Text style={styles.actionText}>
            {copied === "PGN" ? "Copied!" : "Copy PGN"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.lichessButton]}
          onPress={openInLichess}
        >
          <Text style={styles.actionText}>Open in Lichess</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.hint}>Tap any square on the board to edit a piece.</Text>

      <PieceEditor
        visible={editSquare !== null}
        squareLabel={
          editSquare
            ? `${FILE_LABELS[editSquare.col]}${8 - editSquare.row}`
            : ""
        }
        onSelect={handlePieceSelect}
        onClose={() => setEditSquare(null)}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#16213e",
  },
  content: {
    padding: 24,
    alignItems: "center",
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: "#16213e",
    justifyContent: "center",
    alignItems: "center",
    gap: 16,
  },
  loadingText: {
    color: "#a8a8b8",
    fontSize: 16,
  },
  boardContainer: {
    marginBottom: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  warningBox: {
    backgroundColor: "#4a2020",
    borderRadius: 10,
    padding: 12,
    width: "100%",
    marginBottom: 16,
  },
  warningTitle: {
    color: "#ff6b6b",
    fontWeight: "bold",
    marginBottom: 4,
  },
  warningText: {
    color: "#ffaaaa",
    fontSize: 13,
  },
  fenContainer: {
    width: "100%",
    marginBottom: 16,
  },
  label: {
    color: "#a8a8b8",
    fontSize: 14,
    marginBottom: 6,
  },
  fenBox: {
    backgroundColor: "#0f3460",
    borderRadius: 10,
    padding: 14,
    borderWidth: 1,
    borderColor: "#1a4a80",
  },
  fenText: {
    color: "#fff",
    fontSize: 13,
    fontFamily: "monospace",
  },
  toggleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 20,
    alignSelf: "flex-start",
  },
  toggleButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: "#0f3460",
  },
  toggleActive: {
    backgroundColor: "#e94560",
  },
  toggleText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
  actions: {
    width: "100%",
    gap: 10,
    marginBottom: 20,
  },
  actionButton: {
    backgroundColor: "#0f3460",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1a4a80",
  },
  lichessButton: {
    backgroundColor: "#4a4a00",
    borderColor: "#6a6a20",
  },
  actionText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  hint: {
    color: "#666",
    fontSize: 13,
    textAlign: "center",
    marginBottom: 40,
  },
});
