const express = require('express');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');
const chokidar = require('chokidar');
const esbuild = require('esbuild');
const { createProxyMiddleware } = require('./proxyShim.cjs');

async function dev(config) {
  const app = express();
  app.use(morgan('dev'));

  // Proxy /api to backend
  if (config.proxy) {
    for (const [route, target] of Object.entries(config.proxy)) {
      app.use(route, createProxyMiddleware(target));
    }
  }

  // Serve public files first
  app.use(express.static(config.publicDir));

  // Build to a temp dev dir
  const outDir = path.join(config.root, '.arcane-dev');
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });

  async function rebuild() {
    await esbuild.build({
      entryPoints: [config.entry],
      bundle: true,
      sourcemap: true,
      outfile: path.join(outDir, 'assets', 'app.js'),
      platform: 'browser',
      target: ['es2020'],
      define: { 'process.env.NODE_ENV': JSON.stringify('development') }
    });
  }

  await rebuild();

  // Serve compiled bundle
  app.use('/assets', express.static(path.join(outDir, 'assets')));

  // Serve index.html (from public) with injected dev script
  app.get('*', (_req, res) => {
    const indexPath = path.join(config.publicDir, 'index.html');
    let html = fs.readFileSync(indexPath, 'utf-8');
    if (!html.includes('/assets/app.js')) {
      html = html.replace('</body>', `  <script src="/assets/app.js"></script>\n</body>`);
    }
    res.setHeader('Content-Type', 'text/html');
    res.send(html);
  });

  // Watch and rebuild
  chokidar.watch(config.srcDir, { ignoreInitial: true }).on('all', async () => {
    try {
      await rebuild();
    } catch (e) {
      console.error(e);
    }
  });

  app.listen(config.port, () => {
    console.log(`🔥 Arcane dev running at http://127.0.0.1:${config.port}`);
  });
}

module.exports = { dev };
