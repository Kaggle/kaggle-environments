import fs from 'fs';
import { loadEnvFile } from 'node:process';

try {
  loadEnvFile();
} catch (error) {
  console.log('No .env found');
}

if (!process.env.VITE_REPLAY_FILE) process.exit();

console.log('Copy replay', process.env.VITE_REPLAY_FILE);

const from = process.env.VITE_REPLAY_FILE;
const to = `dist/${process.env.VITE_REPLAY_FILE}`;

fs.cp(from, to, { recursive: true }, (err) => {
  if (err) console.log(err);
  process.exit();
});
