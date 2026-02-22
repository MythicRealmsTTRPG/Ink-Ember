const esbuild = require('esbuild');
const fs = require('fs');
const path = require('path');

function copyDir(src, dest) {
  if (!fs.existsSync(src)) return;
  fs.mkdirSync(dest, { recursive: true });
  for (const item of fs.readdirSync(src)) {
    const s = path.join(src, item);
    const d = path.join(dest, item);
    const stat = fs.statSync(s);
    if (stat.isDirectory()) copyDir(s, d);
    else fs.copyFileSync(s, d);
  }
}

async function build(config) {
  fs.rmSync(config.outDir, { recursive: true, force: true });
  fs.mkdirSync(config.outDir, { recursive: true });

  // Copy public assets first
  copyDir(config.publicDir, config.outDir);

  // Bundle app entry
  await esbuild.build({
    entryPoints: [config.entry],
    bundle: true,
    minify: true,
    sourcemap: false,
    outfile: path.join(config.outDir, 'assets', 'app.js'),
    platform: 'browser',
    target: ['es2020'],
    loader: {
      '.png': 'file',
      '.jpg': 'file',
      '.jpeg': 'file',
      '.svg': 'file',
      '.woff2': 'file'
    },
    define: {
      'process.env.NODE_ENV': JSON.stringify('production')
    }
  });

  // Ensure index.html includes app bundle
  const indexPath = path.join(config.outDir, 'index.html');
  if (fs.existsSync(indexPath)) {
    let html = fs.readFileSync(indexPath, 'utf-8');
    if (!html.includes('assets/app.js')) {
      html = html.replace('</body>', `  <script src="/assets/app.js"></script>\n</body>`);
      fs.writeFileSync(indexPath, html, 'utf-8');
    }
  }

  console.log(`✅ Arcane build complete -> ${config.outDir}`);
}

module.exports = { build };
