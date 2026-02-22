import http from 'http';
import https from 'https';
import { URL } from 'url';

export function createProxyMiddleware(target) {
  const targetUrl = new URL(target);

  return (req, res) => {
    const lib = targetUrl.protocol === 'https:' ? https : http;

    const upstreamReq = lib.request(
      {
        hostname: targetUrl.hostname,
        port: targetUrl.port,
        path: req.originalUrl,
        method: req.method,
        headers: { ...req.headers, host: targetUrl.host }
      },
      (upstreamRes) => {
        res.writeHead(upstreamRes.statusCode || 500, upstreamRes.headers);
        upstreamRes.pipe(res, { end: true });
      }
    );

    req.pipe(upstreamReq, { end: true });

    upstreamReq.on('error', (err) => {
      res.statusCode = 502;
      res.setHeader('Content-Type', 'text/plain');
      res.end(String(err));
    });
  };
}
