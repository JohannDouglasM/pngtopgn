const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// Add .onnx to asset extensions so models can be bundled
config.resolver.assetExts.push("onnx");

module.exports = config;
