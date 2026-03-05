import { useEffect } from "react";
import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { initModels } from "../src/ml/modelLoader";

export default function RootLayout() {
  // Start loading models early so they're ready when the user reaches the result screen
  useEffect(() => {
    initModels().catch((err) =>
      console.warn("Early model init failed (will retry on result screen):", err)
    );
  }, []);

  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: "#1a1a2e" },
          headerTintColor: "#fff",
          headerTitleStyle: { fontWeight: "bold" },
          contentStyle: { backgroundColor: "#16213e" },
        }}
      >
        <Stack.Screen name="index" options={{ title: "pngtopgn" }} />
        <Stack.Screen name="camera" options={{ title: "Capture Board" }} />
        <Stack.Screen name="corners" options={{ title: "Select Corners" }} />
        <Stack.Screen name="result" options={{ title: "Detected Position" }} />
      </Stack>
    </>
  );
}
