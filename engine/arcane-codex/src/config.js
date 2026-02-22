import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';

export async function loadConfig(cwd) {
  // Prefer frontend/arcane.config.js when run from repo root.
  const candidates = [
    path.join(cwd, 'arcane.config.js'),
    path.join(cwd, 'frontend', 'arcane.config.js')
  ];

  const configPath = candidates.find((p) => fs.existsSync(p));
  if (!configPath) {
    throw new Error('ArcaneCodex: arcane.config.js not found (looked in repo root and /frontend).');
  }

  const url = pathToFileURL(configPath).toString();
  const mod = await import(url);
  const cfg = mod.default || mod;

  // Normalize paths relative to config directory.
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
