import fs from 'fs';
import https from 'https';
import { createGunzip } from 'zlib';
import process from 'node:process';
import { Game } from 'tenuki';
import { setTimeout } from 'timers/promises';

try {
  process.loadEnvFile();
} catch (error) {
  console.log('No .env found');
}

if (!process.env.REPLAY_DOWNLOAD_FOLDER) process.exit();

function kaggleApi(path, data) {
  return new Promise((resolve, reject) => {
    const url = `https://www.kaggle.com/api${path}`;
    const options = {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'content-length': Buffer.byteLength(data),
        'cookie': process.env.REPLAY_DOWNLOAD_COOKIE,
        // "accept-encoding": "gzip",
      },
    };

    const req = https.request(url, options, (res) => {
      const isGzip = res.headers['content-encoding'] === 'gzip';
      const stream = isGzip ? res.pipe(createGunzip()) : res;
      let responseData = '';
      stream.on('data', (chunk) => {
        responseData += chunk;
      });
      stream.on('end', () => {
        resolve(JSON.parse(responseData));
      });
    });

    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    let j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

const list = await kaggleApi(
  '/i/competitions.EpisodeService/ListEpisodes',
  JSON.stringify({
    ids: [],
    submissionId: process.env.REPLAY_DOWNLOAD_SUB_ID,
    successfulOnly: true,
  })
);

const teams = [];
const downloads = [];

shuffle(list.episodes);

for (const episode of list.episodes) {
  const replay = await kaggleApi(
    '/i/competitions.EpisodeService/GetEpisodeReplay',
    JSON.stringify({
      episodeId: episode.id,
    })
  );

  if (replay.error) {
    console.log(episode.id, replay.error);
    break;
  }

  // const replay = JSON.parse(fs.readFileSync('replays/ladder-replay.json'));

  console.log(replay.info.TeamNames);

  const boardSize = JSON.parse(replay.info.stateHistory[0]).board_size;

  console.log(boardSize);

  const game = new Game({ boardSize });
  let download = false;

  replay.steps.forEach((step) => {
    step.forEach((player) => {
      if (player.action.actionString) {
        const [, move] = player.action.actionString.split(' ');
        if (move === 'PASS') {
          game.pass();
        } else {
          const cols = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7,
            'j': 8,
            'k': 9,
            'l': 10,
            'm': 11,
            'n': 12,
            'o': 13,
            'p': 14,
            'q': 15,
            'r': 16,
            's': 17,
            't': 18,
          };
          const y = boardSize - parseInt(move.slice(1));
          const x = cols[move.charAt(0)];

          game.playAt(y, x);

          const isLadder = (game) => {
            if (game.currentState().moveNumber < 10) {
              return false;
            }

            let state = game._moves.at(-1);
            let atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            if (atari.length !== 5) {
              return false;
            }

            state = game._moves.at(-3);
            atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            if (atari.length !== 4) {
              return false;
            }

            state = game._moves.at(-5);
            atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            if (atari.length !== 3) {
              return false;
            }

            state = game._moves.at(-7);
            atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            if (atari.length !== 2) {
              return false;
            }

            state = game._moves.at(-9);
            atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            if (atari.length !== 1) {
              return false;
            }

            const dy = game._moves.at(-5).playedPoint.y - game._moves.at(-9).playedPoint.y;
            const dx = game._moves.at(-5).playedPoint.x - game._moves.at(-9).playedPoint.x;

            if (dy === 0 || dx === 0 || Math.abs(dy) !== Math.abs(dx)) {
              return false;
            }

            const py = game._moves.at(-9).playedPoint.y + 2 * dy;
            const px = game._moves.at(-9).playedPoint.x + 2 * dx;

            if (game._moves.at(-1).playedPoint.y !== py || game._moves.at(-1).playedPoint.x !== px) {
              return false;
            }

            // console.log(state.moveNumber);

            return true;
          };

          if (isLadder(game)) {
            download = true;
          }
        }
      }
    });
  });

  if (download) {
    const teamNames = replay.info.TeamNames.join('-').replace(/ /g, '');
    const folderName = process.env.REPLAY_DOWNLOAD_FOLDER;
    const fileName = `${folderName}/${episode.id}-${teamNames}-ladder.json`;

    fs.mkdirSync(folderName, { recursive: true });
    fs.writeFileSync(fileName, JSON.stringify(replay, null, 2));

    console.log(`Downloaded to ${fileName}`);
  }

  await setTimeout(500);
}
