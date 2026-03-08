import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";

export default function RootLayout() {
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
        <Stack.Screen name="result" options={{ title: "Detected Position" }} />
      </Stack>
    </>
  );
}
