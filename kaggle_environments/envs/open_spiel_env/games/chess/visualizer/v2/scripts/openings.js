import fs from 'fs';
import path from 'path';
import process from 'node:process';

try {
  process.loadEnvFile();
} catch (error) {
  console.log('No .env found');
}

if (!process.env.REPLAY_DOWNLOAD_FOLDER) process.exit();

const openings = JSON.parse(fs.readFileSync('./src/data/openings.json'));
const statistics = [];

const filenames = fs.readdirSync(process.env.REPLAY_DOWNLOAD_FOLDER);

for (const filename of filenames) {
  if (path.extname(filename) !== '.json') continue;

  const file = path.join(process.env.REPLAY_DOWNLOAD_FOLDER, filename);
  const replay = JSON.parse(fs.readFileSync(file));

  for (const fen of replay.info.stateHistory) {
    const opening = openings.find((opening) => fen.includes(opening.fen));
    if (opening) {
      console.log(replay.info.TeamNames, opening.name);
      const stat = statistics.find((stat) => stat.name === opening.name);
      if (stat) {
        stat.count += 1;
      } else {
        statistics.push({ name: opening.name, count: 1 });
      }
    }
  }
}

for (const stat of statistics) {
  console.log(stat.count.toString().padEnd(5), stat.name);
}
