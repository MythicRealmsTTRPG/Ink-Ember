import esbuild from 'esbuild';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { copyDir } from './fsUtil.js';
import { aliasPlugin } from './aliasPlugin.js';
import { injectAssets, readHtml, writeHtml } from './htmlInject.js';

function loadEnv(config, mode) {
  const envFiles = [
    path.join(config.root, '.env'),
    path.join(config.root, `.env.${mode}`)
  ];

  for (const p of envFiles) {
    if (fs.existsSync(p)) dotenv.config({ path: p, override: true });
  }
}

export async function build(config) {
  const mode = 'production';
  loadEnv(config, mode);

  fs.rmSync(config.outDir, { recursive: true, force: true });
  fs.mkdirSync(config.outDir, { recursive: true });
  fs.mkdirSync(path.join(config.outDir, 'assets'), { recursive: true });

  copyDir(config.publicDir, config.outDir);

  const define = {
    'process.env.NODE_ENV': JSON.stringify('production')
  };

  // Expose env vars with configured prefix.
  const prefix = config.envPrefix || 'REACT_APP_';
  for (const [k, v] of Object.entries(process.env)) {
    if (k.startsWith(prefix)) define[`process.env.${k}`] = JSON.stringify(String(v));
  }

  await esbuild.build({
    entryPoints: [config.entry],
    bundle: true,
    minify: true,
    sourcemap: false,
    outdir: path.join(config.outDir, 'assets'),
    entryNames: 'app',
    assetNames: '[name]-[hash]',
    platform: 'browser',
    target: ['es2020'],
    plugins: [aliasPlugin(config.alias)],
    loader: {
      '.png': 'file',
      '.jpg': 'file',
      '.jpeg': 'file',
      '.svg': 'file',
      '.woff': 'file',
      '.woff2': 'file',
      '.ttf': 'file',
      '.eot': 'file',
      '.css': 'css'
    },
    define
  });

  const indexPath = path.join(config.outDir, 'index.html');
  if (fs.existsSync(indexPath)) {
    const jsPath = '/assets/app.js';
    const cssOut = path.join(config.outDir, 'assets', 'app.css');
    const cssPath = fs.existsSync(cssOut) ? '/assets/app.css' : null;

    let html = readHtml(indexPath);
    html = injectAssets(html, { jsPath, cssPath });
    writeHtml(indexPath, html);
  }

  console.log(`✅ ArcaneCodex build complete -> ${config.outDir}`);
}
