import { useState, useRef } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  LayoutChangeEvent,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";

const CORNER_LABELS = ["a1 (bottom-left)", "a8 (top-left)", "h8 (top-right)", "h1 (bottom-right)"];
const CORNER_COLORS = ["#e94560", "#f5a623", "#50c878", "#4a90d9"];

type Point = { x: number; y: number };

export default function CornersScreen() {
  const { imageUri } = useLocalSearchParams<{ imageUri: string }>();
  const router = useRouter();
  const [corners, setCorners] = useState<Point[]>([]);
  const [imageLayout, setImageLayout] = useState({ width: 0, height: 0, x: 0, y: 0 });
  const screenWidth = Dimensions.get("window").width;

  const handleImagePress = (event: any) => {
    if (corners.length >= 4) return;

    const { locationX, locationY } = event.nativeEvent;
    const normalizedX = locationX / imageLayout.width;
    const normalizedY = locationY / imageLayout.height;

    setCorners((prev) => [...prev, { x: normalizedX, y: normalizedY }]);
  };

  const handleImageLayout = (event: LayoutChangeEvent) => {
    const { width, height, x, y } = event.nativeEvent.layout;
    setImageLayout({ width, height, x, y });
  };

  const handleUndo = () => {
    setCorners((prev) => prev.slice(0, -1));
  };

  const handleReset = () => {
    setCorners([]);
  };

  const handleContinue = () => {
    if (corners.length !== 4) return;
    router.push({
      pathname: "/result",
      params: {
        imageUri,
        corners: JSON.stringify(corners),
      },
    });
  };

  return (
    <View style={styles.container}>
      <View style={styles.instructions}>
        {corners.length < 4 ? (
          <Text style={styles.instructionText}>
            Tap corner{" "}
            <Text style={{ color: CORNER_COLORS[corners.length], fontWeight: "bold" }}>
              {CORNER_LABELS[corners.length]}
            </Text>
          </Text>
        ) : (
          <Text style={styles.instructionText}>All corners selected! Tap Continue.</Text>
        )}
      </View>

      <View style={styles.imageContainer}>
        <TouchableOpacity activeOpacity={1} onPress={handleImagePress} onLayout={handleImageLayout}>
          <Image
            source={{ uri: imageUri }}
            style={{ width: screenWidth - 32, height: screenWidth - 32 }}
            resizeMode="contain"
          />
          {corners.map((corner, i) => (
            <View
              key={i}
              style={[
                styles.cornerDot,
                {
                  left: corner.x * imageLayout.width - 12,
                  top: corner.y * imageLayout.height - 12,
                  backgroundColor: CORNER_COLORS[i],
                },
              ]}
            >
              <Text style={styles.cornerLabel}>{i + 1}</Text>
            </View>
          ))}
          {corners.length >= 2 &&
            corners.map((corner, i) => {
              if (i === 0) return null;
              const prev = corners[i - 1];
              return (
                <View
                  key={`line-${i}`}
                  style={[
                    styles.line,
                    getLineStyle(
                      prev.x * imageLayout.width,
                      prev.y * imageLayout.height,
                      corner.x * imageLayout.width,
                      corner.y * imageLayout.height,
                      CORNER_COLORS[i - 1]
                    ),
                  ]}
                />
              );
            })}
          {corners.length === 4 && (
            <View
              style={[
                styles.line,
                getLineStyle(
                  corners[3].x * imageLayout.width,
                  corners[3].y * imageLayout.height,
                  corners[0].x * imageLayout.width,
                  corners[0].y * imageLayout.height,
                  CORNER_COLORS[3]
                ),
              ]}
            />
          )}
        </TouchableOpacity>
      </View>

      <View style={styles.buttonRow}>
        <TouchableOpacity
          style={[styles.button, styles.undoButton]}
          onPress={handleUndo}
          disabled={corners.length === 0}
        >
          <Text style={styles.buttonText}>Undo</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.button, styles.resetButton]}
          onPress={handleReset}
          disabled={corners.length === 0}
        >
          <Text style={styles.buttonText}>Reset</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.button, styles.continueButton, corners.length < 4 && styles.disabledButton]}
          onPress={handleContinue}
          disabled={corners.length < 4}
        >
          <Text style={styles.buttonText}>Continue</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

function getLineStyle(x1: number, y1: number, x2: number, y2: number, color: string) {
  const length = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  const angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);
  return {
    position: "absolute" as const,
    left: x1,
    top: y1,
    width: length,
    height: 2,
    backgroundColor: color,
    transformOrigin: "0 0" as any,
    transform: [{ rotate: `${angle}deg` }],
    opacity: 0.7,
  };
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#16213e",
  },
  instructions: {
    padding: 16,
    alignItems: "center",
  },
  instructionText: {
    color: "#fff",
    fontSize: 16,
    textAlign: "center",
  },
  imageContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 16,
  },
  cornerDot: {
    position: "absolute",
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "#fff",
  },
  cornerLabel: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "bold",
  },
  line: {
    position: "absolute",
  },
  buttonRow: {
    flexDirection: "row",
    padding: 16,
    gap: 12,
  },
  button: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  undoButton: {
    backgroundColor: "#333",
  },
  resetButton: {
    backgroundColor: "#333",
  },
  continueButton: {
    backgroundColor: "#e94560",
  },
  disabledButton: {
    opacity: 0.4,
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
});
