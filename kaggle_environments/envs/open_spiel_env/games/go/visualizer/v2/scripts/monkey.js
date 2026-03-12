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
  // const replay = await kaggleApi(
  //   '/i/competitions.EpisodeService/GetEpisodeReplay',
  //   JSON.stringify({
  //     episodeId: episode.id,
  //   })
  // );

  // if (replay.error) {
  //   console.log(episode.id, replay.error);
  //   break;
  // }

  const replay = JSON.parse(fs.readFileSync('replays/monkey-replay.json'));

  console.log(replay.info.TeamNames);

  const boardSize = JSON.parse(replay.info.stateHistory[0]).board_size;
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

          const isMonkeyJump = (game) => {
            const state = game.currentState();
            const point = state.playedPoint;
            const color = state.color;
            const max = state.boardSize - 1;

            if (point.x !== 0 && point.y !== 0 && point.x !== max && point.y !== max) {
              return false;
            }

            if (
              (point.x < 3 && point.y < 3) ||
              (point.x < 3 && point.y > max - 3) ||
              (point.x > max - 3 && point.y < 3) ||
              (point.x > max - 3 && point.y > max - 3)
            ) {
              return false;
            }

            let dy, dx;
            if (point.x === 0) [dy, dx] = [0, 1];
            if (point.y === 0) [dy, dx] = [1, 0];
            if (point.x === max) [dy, dx] = [0, -1];
            if (point.y === max) [dy, dx] = [-1, 0];

            let neighbors = [];
            neighbors.push(...state.neighborsFor(point.y + dx * 2, point.x + dy * 2));
            neighbors.push(...state.neighborsFor(point.y + dx, point.x + dy));
            neighbors.push(...state.neighborsFor(point.y, point.x));
            neighbors.push(...state.neighborsFor(point.y - dx, point.x - dy));
            neighbors.push(...state.neighborsFor(point.y - dx * 2, point.x - dy * 2));
            neighbors = [...new Set(neighbors)];
            neighbors = neighbors.filter((intersection) => intersection.isEmpty());
            if (neighbors.length !== 11) {
              return false;
            }

            let pSame, pDiff;

            if (point.y === 0) {
              const p0 = state.intersectionAt(point.y, point.x);
              const p1 = state.intersectionAt(1, point.x - 3);
              const p2 = state.intersectionAt(1, point.x + 3);

              if (p1.isEmpty() === p2.isEmpty()) return false;
              
              pSame = p1.isEmpty() === false ? p1 : p2;

              if (pSame.value !== p0.value) return false; 

              const dir = pSame.x < p0.x ? -1 : 1;
              pDiff = state.intersectionAt(2, p0.x + 2 * dir);

              if (pDiff.value === point.value) return false;
            } else {
              return false;
            }

            const sameColor = state
              .groupAt(pSame.y, pSame.x)
              .filter((intersection) => intersection.isOccupiedWith(color) === true);
            if (sameColor.length < 2) {
              return false;
            }

            const oppColor = state
              .groupAt(pDiff.y, pDiff.x)
              .filter(
                (intersection) => intersection.isOccupiedWith(color) === false && intersection.isEmpty() === false
              );
            if (oppColor.length < 2) {
              return false;
            }

            console.log(`Match at ${state.moveNumber}`);

            return true;
          };

          if (isMonkeyJump(game)) {
            download = true;
          }
        }
      }
    });
  });

  if (download) {
    const teamNames = replay.info.TeamNames.join('-').replace(/ /g, '');
    const folderName = process.env.REPLAY_DOWNLOAD_FOLDER;
    const fileName = `${folderName}/${episode.id}-${teamNames}-monkey.json`;

    fs.mkdirSync(folderName, { recursive: true });
    fs.writeFileSync(fileName, JSON.stringify(replay, null, 2));

    console.log(`Downloaded to ${fileName}`);
  }

  await setTimeout(500);
}
