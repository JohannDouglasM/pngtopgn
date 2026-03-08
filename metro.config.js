const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// Add .onnx as an asset extension so models are bundled
config.resolver.assetExts.push("onnx");

module.exports = config;
