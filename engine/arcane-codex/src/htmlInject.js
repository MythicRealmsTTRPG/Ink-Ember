import fs from 'fs';

export function injectAssets(html, { jsPath, cssPath }) {
  let out = html;

  if (cssPath && !out.includes(cssPath)) {
    // try insert before </head>
    out = out.replace(
      /<\/head>/i,
      `  <link rel="stylesheet" href="${cssPath}"></head>`
    );
  }

  if (jsPath && !out.includes(jsPath)) {
    out = out.replace(
      /<\/body>/i,
      `  <script src="${jsPath}"></script></body>`
    );
  }

  return out;
}

export function readHtml(p) {
  return fs.readFileSync(p, 'utf-8');
}

export function writeHtml(p, html) {
  fs.writeFileSync(p, html, 'utf-8');
}
