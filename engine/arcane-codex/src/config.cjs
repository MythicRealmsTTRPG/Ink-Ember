const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

async function loadConfig(cwd) {
  const candidates = [
    path.join(cwd, 'arcane.config.js'),
    path.join(cwd, 'frontend', 'arcane.config.js')
  ];

  const configPath = candidates.find((p) => fs.existsSync(p));
  if (!configPath) {
    throw new Error('ArcaneCodex: arcane.config.js not found (looked in repo root and /frontend).');
  }

  // Use dynamic import so configs can be ESM.
  const url = pathToFileURL(configPath).toString();
  const mod = await import(url);
  const cfg = mod.default || mod;

  const baseDir = path.dirname(configPath);
  const resolve = (p) => (p ? (path.isAbsolute(p) ? p : path.join(baseDir, p)) : p);

  return {
    root: resolve(cfg.root || '.'),
    srcDir: resolve(cfg.srcDir || 'src'),
    publicDir: resolve(cfg.publicDir || 'public'),
    outDir: resolve(cfg.outDir || 'build'),
    entry: resolve(cfg.entry || path.join('src', 'index.js')),
    port: cfg.port ?? 3000,
    open: cfg.open ?? false,
    proxy: cfg.proxy || {},
    alias: cfg.alias || {},
    envPrefix: cfg.envPrefix || 'REACT_APP_',
    plugins: cfg.plugins || []
  };
}

module.exports = { loadConfig };
