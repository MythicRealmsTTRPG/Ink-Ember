import express from 'express';
import morgan from 'morgan';
import path from 'path';
import fs from 'fs';
import chokidar from 'chokidar';
import esbuild from 'esbuild';
import dotenv from 'dotenv';
import { aliasPlugin } from './aliasPlugin.js';
import { createProxyMiddleware } from './proxyShim.js';
import { injectAssets, readHtml } from './htmlInject.js';

function loadEnv(config, mode) {
  const envFiles = [
    path.join(config.root, '.env'),
    path.join(config.root, `.env.${mode}`)
  ];

  for (const p of envFiles) {
    if (fs.existsSync(p)) dotenv.config({ path: p, override: true });
  }
}

export async function dev(config) {
  const mode = 'development';
  loadEnv(config, mode);

  const app = express();
  app.use(morgan('dev'));

  // Proxy /api -> backend
  for (const [route, target] of Object.entries(config.proxy || {})) {
    app.use(route, createProxyMiddleware(target));
  }

  // Serve public assets
  app.use(express.static(config.publicDir));

  // Build to a temp folder in frontend/.arcane-dev
  const outDir = path.join(config.root, '.arcane-dev');
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(path.join(outDir, 'assets'), { recursive: true });

  const define = {
    'process.env.NODE_ENV': JSON.stringify('development')
  };

  const prefix = config.envPrefix || 'REACT_APP_';
  for (const [k, v] of Object.entries(process.env)) {
    if (k.startsWith(prefix)) define[`process.env.${k}`] = JSON.stringify(String(v));
  }

  async function rebuild() {
    await esbuild.build({
      entryPoints: [config.entry],
      bundle: true,
      minify: false,
      sourcemap: true,
      outdir: path.join(outDir, 'assets'),
      entryNames: 'app',
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
  }

  await rebuild();

  app.use('/assets', express.static(path.join(outDir, 'assets')));

  app.get('*', (_req, res) => {
    const indexPath = path.join(config.publicDir, 'index.html');
    let html = readHtml(indexPath);

    const cssOut = path.join(outDir, 'assets', 'app.css');
    const cssPath = fs.existsSync(cssOut) ? '/assets/app.css' : null;

    html = injectAssets(html, { jsPath: '/assets/app.js', cssPath });

    res.setHeader('Content-Type', 'text/html');
    res.send(html);
  });

  chokidar.watch(config.srcDir, { ignoreInitial: true }).on('all', async () => {
    try {
      await rebuild();
    } catch (e) {
      console.error(e);
    }
  });

  app.listen(config.port, () => {
    console.log(`🔥 ArcaneCodex dev -> http://127.0.0.1:${config.port}`);
  });
}
