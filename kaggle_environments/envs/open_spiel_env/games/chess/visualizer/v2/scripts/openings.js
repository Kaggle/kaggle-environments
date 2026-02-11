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

fs.readdir(process.env.REPLAY_DOWNLOAD_FOLDER, (err, files) => {
  files.forEach((filename) => {
    const file = path.join(process.env.REPLAY_DOWNLOAD_FOLDER, filename);
    const replay = JSON.parse(fs.readFileSync(file));

    replay.info.stateHistory.forEach((fen) => {
      const opening = openings.find((opening) => fen.includes(opening.fen));
      if (opening) console.log(replay.info.TeamNames, opening.name);
    });
  });
});
