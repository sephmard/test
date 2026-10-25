// Zero-dependency Node server for local use or self-hosting (Node 22.13+).
//   node server.js            -> http://localhost:8787, data in ./data/overlap.db
//   PORT=3000 DB_PATH=:memory: node server.js

import { createServer } from 'node:http';
import { mkdirSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import { handleApi } from './src/api.js';
import { createStore, sqliteAdapter } from './src/store.js';

const root = dirname(fileURLToPath(import.meta.url));
const publicDir = join(root, 'public');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.webmanifest': 'application/manifest+json',
};

export function createApp({ dbPath = ':memory:' } = {}) {
  if (dbPath !== ':memory:') mkdirSync(dirname(dbPath), { recursive: true });
  const store = createStore(sqliteAdapter(new DatabaseSync(dbPath)));

  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
      if (url.pathname.startsWith('/api/')) {
        const chunks = [];
        for await (const c of req) chunks.push(c);
        const body = chunks.length ? Buffer.concat(chunks) : undefined;
        const request = new Request(url, { method: req.method, headers: req.headers, body: body?.length ? body : undefined });
        const response = await handleApi(request, store);
        res.writeHead(response.status, Object.fromEntries(response.headers));
        res.end(Buffer.from(await response.arrayBuffer()));
        return;
      }
      await serveStatic(url.pathname, res);
    } catch (e) {
      console.error(e);
      res.writeHead(500).end('Internal error');
    }
  });
}

async function serveStatic(pathname, res) {
  const rel = normalize(decodeURIComponent(pathname)).replace(/^(\.\.[/\\])+/, '');
  let file = join(publicDir, rel);
  if (!file.startsWith(publicDir)) file = join(publicDir, 'index.html');
  let data;
  try {
    data = await readFile(file);
  } catch {
    // Single-page app: unknown paths (e.g. /e/abc123) get index.html.
    file = join(publicDir, 'index.html');
    data = await readFile(file);
  }
  res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' });
  res.end(data);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const port = Number(process.env.PORT || 8787);
  const dbPath = process.env.DB_PATH || join(root, 'data', 'overlap.db');
  createApp({ dbPath }).listen(port, () => console.log(`Overlap running at http://localhost:${port}`));
}
