import fs from 'fs';

if (!process.env.VITE_REPLAY_FILE) process.exit();

console.log('Copy replay', process.env.VITE_REPLAY_FILE);

const from = process.env.VITE_REPLAY_FILE;
const to = `dist/${process.env.VITE_REPLAY_FILE}`;

fs.cp(from, to, { recursive: true }, (err) => {
  if (err) console.log(err);
  process.exit();
});
