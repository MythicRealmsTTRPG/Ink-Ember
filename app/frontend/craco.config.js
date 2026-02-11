// craco.config.js
const path = require("path");
require("dotenv").config();

// CRA/CRACO: NODE_ENV=development for start, production for build
const isDevServer = process.env.NODE_ENV !== "production";

const flags = {
  enableHealthCheck: process.env.ENABLE_HEALTH_CHECK === "true",
  enableVisualEdits: isDevServer, // dev server only
};

// --- Dev-only Visual Edits ---
let setupDevServer;
let babelMetadataPlugin;

if (flags.enableVisualEdits) {
  setupDevServer = require("./plugins/visual-edits/dev-server-setup");
  babelMetadataPlugin = require("./plugins/visual-edits/babel-metadata-plugin");
}

// --- Optional Health Check ---
let setupHealthEndpoints;
let healthPluginInstance;

if (flags.enableHealthCheck) {
  // Support either default export or named export
  const healthModule = require("./plugins/health-check/webpack-health-plugin");
  const WebpackHealthPlugin =
    healthModule?.WebpackHealthPlugin ?? healthModule;

  setupHealthEndpoints = require("./plugins/health-check/health-endpoints");

  if (typeof WebpackHealthPlugin !== "function") {
    throw new Error(
      "Health check plugin export is not a constructor. Export a class (default) or { WebpackHealthPlugin }."
    );
  }

  healthPluginInstance = new WebpackHealthPlugin();
}

const cracoConfig = {
  eslint: {
    configure: {
      extends: ["plugin:react-hooks/recommended"],
      rules: {
        "react-hooks/rules-of-hooks": "error",
        "react-hooks/exhaustive-deps": "warn",
      },
    },
  },

  webpack: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },

    configure: (cfg) => {
      // Reduce watched directories (better dev perf)
      const ignored = [
        "**/node_modules/**",
        "**/.git/**",
        "**/build/**",
        "**/dist/**",
        "**/coverage/**",
        // "**/public/**", // Uncomment ONLY if you intentionally don't want public/ changes to trigger reload
      ];

      cfg.watchOptions = {
        ...(cfg.watchOptions || {}),
        ignored: Array.isArray(cfg.watchOptions?.ignored)
          ? [...cfg.watchOptions.ignored, ...ignored]
          : ignored,
      };

      // Add health check plugin if enabled
      if (flags.enableHealthCheck && healthPluginInstance) {
        cfg.plugins = cfg.plugins || [];
        cfg.plugins.push(healthPluginInstance);
      }

      return cfg;
    },
  },
};

// Add babel metadata plugin only during dev server
if (flags.enableVisualEdits && babelMetadataPlugin) {
  cracoConfig.babel = {
    plugins: [babelMetadataPlugin],
  };
}

cracoConfig.devServer = (devServerConfig) => {
  let cfg = devServerConfig;

  // Visual edits dev server setup (dev-only)
  if (flags.enableVisualEdits && typeof setupDevServer === "function") {
    cfg = setupDevServer(cfg);
  }

  // Add health endpoints (if enabled)
  if (flags.enableHealthCheck && setupHealthEndpoints && healthPluginInstance) {
    const originalSetupMiddlewares = cfg.setupMiddlewares;

    cfg.setupMiddlewares = (middlewares, devServer) => {
      if (typeof originalSetupMiddlewares === "function") {
        middlewares = originalSetupMiddlewares(middlewares, devServer);
      }

      // Attach health endpoints
      setupHealthEndpoints(devServer, healthPluginInstance);

      return middlewares;
    };
  }

  return cfg;
};

module.exports = cracoConfig;
