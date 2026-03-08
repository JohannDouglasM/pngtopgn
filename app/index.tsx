import { View, Text, TouchableOpacity, StyleSheet, Alert } from "react-native";
import { useRouter } from "expo-router";
import * as ImagePicker from "expo-image-picker";

export default function HomeScreen() {
  const router = useRouter();

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission needed", "Camera permission is required to take photos of chess boards.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ["images"],
      quality: 1,
    });
    if (!result.canceled && result.assets[0]) {
      router.push({ pathname: "/result", params: { imageUri: result.assets[0].uri } });
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 1,
    });
    if (!result.canceled && result.assets[0]) {
      router.push({ pathname: "/result", params: { imageUri: result.assets[0].uri } });
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.logo}>♔</Text>
        <Text style={styles.title}>pngtopgn</Text>
        <Text style={styles.subtitle}>
          Photo → FEN/PGN{"\n"}Scan a chess board, get the position
        </Text>
      </View>

      <View style={styles.buttons}>
        <TouchableOpacity style={styles.primaryButton} onPress={takePhoto}>
          <Text style={styles.buttonIcon}>📷</Text>
          <Text style={styles.buttonText}>Take Photo</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.secondaryButton} onPress={pickImage}>
          <Text style={styles.buttonIcon}>🖼</Text>
          <Text style={styles.buttonText}>Choose from Gallery</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.hint}>
        For best results, photograph the board straight-on with even lighting.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#16213e",
    paddingHorizontal: 24,
    justifyContent: "center",
  },
  header: {
    alignItems: "center",
    marginBottom: 48,
  },
  logo: {
    fontSize: 64,
    marginBottom: 8,
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#e94560",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: "#a8a8b8",
    textAlign: "center",
    lineHeight: 24,
  },
  buttons: {
    gap: 16,
    marginBottom: 32,
  },
  primaryButton: {
    backgroundColor: "#e94560",
    paddingVertical: 18,
    borderRadius: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  secondaryButton: {
    backgroundColor: "#0f3460",
    paddingVertical: 18,
    borderRadius: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    borderWidth: 1,
    borderColor: "#e94560",
  },
  buttonIcon: {
    fontSize: 20,
  },
  buttonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "600",
  },
  hint: {
    color: "#666",
    fontSize: 13,
    textAlign: "center",
    lineHeight: 20,
  },
});
