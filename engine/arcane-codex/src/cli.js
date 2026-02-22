#!/usr/bin/env node
import { loadConfig } from './config.js';
import { dev } from './devServer.js';
import { build } from './builder.js';

const cmd = process.argv[2];

async function main() {
  const config = await loadConfig(process.cwd());

  if (cmd === 'dev') return dev(config);
  if (cmd === 'build') return build(config);

  console.log(`\nArcaneCodex Engine\n\nUsage:\n  arcane dev     Start dev server\n  arcane build   Build production bundle\n`);
  process.exit(cmd ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
