#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';

const arg = process.argv[2];
const defaultReplay = './replays/test-replay.json';

let replayPath;
if (!arg) {
  replayPath = defaultReplay;
} else if (arg.includes('/') || arg.includes('\\')) {
  replayPath = arg;
} else {
  replayPath = `./replays/${arg}`;
}

if (!existsSync(replayPath)) {
  console.error(`[dev-with-replay] Replay file not found: ${replayPath}`);
  process.exit(1);
}

console.log(`[dev-with-replay] Using replay: ${replayPath}`);

const child = spawn('vite', [], {
  env: { ...process.env, VITE_REPLAY_FILE: replayPath },
  stdio: 'inherit',
  shell: true,
});
child.on('exit', (code) => process.exit(code ?? 0));
