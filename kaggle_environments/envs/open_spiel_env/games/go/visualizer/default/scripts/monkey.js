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

for (const [index, episode] of list.episodes.entries()) {
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

  // const replay = JSON.parse(fs.readFileSync('replays/monkey-replay.json'));

  console.log(episode.id, replay.info.TeamNames, `(${index}/${list.episodes.length})`);

  const boardSize = JSON.parse(replay.info.stateHistory[0]).board_size;
  const scoring = 'area';
  const game = new Game({ boardSize, scoring });
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
            const point = state.intersectionAt(state.playedPoint.y, state.playedPoint.x);
            const color = state.color;
            const max = game.boardSize - 1;

            // Played stone is on an edge and not in a corner
            let dy, dx;
            switch (true) {
              case point.x === 0 && point.y >= 3 && point.y <= max - 3:
                dy = 0;
                dx = 1;
                break;
              case point.y === max && point.x >= 3 && point.x <= max - 3:
                dy = -1;
                dx = 0;
                break;
              case point.x === max && point.y >= 3 && point.y <= max - 3:
                dy = 0;
                dx = -1;
                break;
              case point.y === 0 && point.x >= 3 && point.x <= max - 3:
                dy = 1;
                dx = 0;
                break;
              default:
                return false;
            }

            // Space around played stone
            const emptyNeighbors = [
              ...state.neighborsFor(point.y + 2 * dx, point.x + 2 * dy),
              ...state.neighborsFor(point.y + dx, point.x + dy),
              ...state.neighborsFor(point.y, point.x),
              ...state.neighborsFor(point.y - dx, point.x - dy),
              ...state.neighborsFor(point.y - 2 * dx, point.x - 2 * dy),
            ].filter((intersection) => intersection.isEmpty());

            if ([...new Set(emptyNeighbors)].length !== 11) return false;

            // Just one of the two possible positions for a stone from the same player
            const possibleSameColorStones = [
              state.intersectionAt(point.y + dy - 3 * Math.abs(dx), point.x + dx - 3 * Math.abs(dy)),
              state.intersectionAt(point.y + dy + 3 * Math.abs(dx), point.x + dx + 3 * Math.abs(dy)),
            ].filter((intersection) => intersection.isEmpty() === false);

            if (possibleSameColorStones.length !== 1) return false;

            const sameColorStone = possibleSameColorStones.at(0);

            if (sameColorStone.isOccupiedWith(color) === false) return false;

            // Opponents stone matching the direction as the same color found stone
            const direction = sameColorStone.x * dy + sameColorStone.y * dx < point.x * dy + point.y * dx ? -1 : 1;
            const opponentColorStone = state.intersectionAt(
              point.y + 2 * (dy + direction * dx),
              point.x + 2 * (dx + direction * dy)
            );

            if (opponentColorStone.isOccupiedWith(color) || opponentColorStone.isEmpty()) return false;

            // Each found stone should be part of a group, let's say two or more of that color
            if (state.groupAt(sameColorStone.y, sameColorStone.x).length < 2) return false;
            if (state.groupAt(opponentColorStone.y, opponentColorStone.x).length < 2) return false;

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
