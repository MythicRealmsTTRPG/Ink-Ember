const path = require("path");

module.exports = {
  root: __dirname,
  srcDir: path.join(__dirname, "src"),
  publicDir: path.join(__dirname, "public"),
  outDir: path.join(__dirname, "build"),
  entry: path.join(__dirname, "src", "index.js"),

  port: 3000,
  open: true,

  proxy: {
    "/api": "http://127.0.0.1:8001"
  },

  alias: {
    "@": path.join(__dirname, "src")
  },

  envPrefix: "INK_",
  plugins: []
};