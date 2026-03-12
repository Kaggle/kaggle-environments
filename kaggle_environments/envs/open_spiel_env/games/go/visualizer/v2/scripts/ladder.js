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

  // const replay = JSON.parse(fs.readFileSync('replays/ladder-replay.json'));

  console.log(replay.info.TeamNames, `(${index}/${list.episodes.length})`);

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

          const isLadder = (game) => {
            if (game.currentState().moveNumber < 10) {
              return false;
            }

            // *** Version #1 ***

            // let state = game._moves.at(-1);
            // let atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            // if (atari.length !== 5) {
            //   return false;
            // }

            // state = game._moves.at(-3);
            // atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            // if (atari.length !== 4) {
            //   return false;
            // }

            // state = game._moves.at(-5);
            // atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            // if (atari.length !== 3) {
            //   return false;
            // }

            // state = game._moves.at(-7);
            // atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            // if (atari.length !== 2) {
            //   return false;
            // }

            // state = game._moves.at(-9);
            // atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
            // if (atari.length !== 1) {
            //   return false;
            // }

            // const dy = game._moves.at(-5).playedPoint.y - game._moves.at(-9).playedPoint.y;
            // const dx = game._moves.at(-5).playedPoint.x - game._moves.at(-9).playedPoint.x;

            // if (dy === 0 || dx === 0 || Math.abs(dy) !== Math.abs(dx)) {
            //   return false;
            // }

            // const py = game._moves.at(-9).playedPoint.y + 2 * dy;
            // const px = game._moves.at(-9).playedPoint.x + 2 * dx;

            // if (game._moves.at(-1).playedPoint.y !== py || game._moves.at(-1).playedPoint.x !== px) {
            //   return false;
            // }

            // *** Version #2 ***

            // let state = game.currentState();
            // const group = state.groupAt(state.playedPoint.y, state.playedPoint.x);

            // if (group.length !== 6) {
            //   return false;
            // }

            // const dy = game._moves.at(-5).playedPoint.y - state.playedPoint.y;
            // const dx = game._moves.at(-5).playedPoint.x - state.playedPoint.x;

            // if (Math.abs(dy) !== 1 || Math.abs(dx) !== 1) {
            //   return false;
            // }

            // const py = state.playedPoint.y + 2 * dy;
            // const px = state.playedPoint.x + 2 * dx;

            // if (game._moves.at(-9).playedPoint.y !== py || game._moves.at(-9).playedPoint.x !== px) {
            //   return false;
            // }

            // const p2y = game._moves.at(-3).playedPoint.y + dy;
            // const p2x = game._moves.at(-3).playedPoint.x + dx;

            // if (game._moves.at(-7).playedPoint.y !== p2y || game._moves.at(-7).playedPoint.x !== p2x) {
            //   return false;
            // }

            // for (let i = -1; i >= -9; i = i - 2) {
            //   if (
            //     group.some(
            //       (intersection) =>
            //         intersection.y === game._moves.at(i).playedPoint.y &&
            //         intersection.x === game._moves.at(i).playedPoint.x
            //     ) === false
            //   ) {
            //     return false;
            //   }
            // }

            // let neighbors = [];
            // group.forEach((point) => {
            //   neighbors.push(...state.neighborsFor(point.y, point.x));
            // });

            // neighbors = [...new Set(neighbors)];

            // for (let i = 1; i <= 5; i++) {
            //   const atari = group.filter((intersection) =>
            //     game._moves.at(-2 * i).inAtari(intersection.y, intersection.x)
            //   );
            //   if (atari.length !== 6 - i) {
            //     return false;
            //   }
            // }

            // *** Version #3 ***

            const history = game._moves;

            if (
              history.at(-1).pass ||
              history.at(-3).pass ||
              history.at(-5).pass ||
              history.at(-7).pass ||
              history.at(-9).pass
            ) {
              return false;
            }

            const d1y = history.at(-3).playedPoint.y - history.at(-1).playedPoint.y;
            const d1x = history.at(-3).playedPoint.x - history.at(-1).playedPoint.x;
            const d2y = history.at(-5).playedPoint.y - history.at(-3).playedPoint.y;
            const d2x = history.at(-5).playedPoint.x - history.at(-3).playedPoint.x;

            if (d1y !== 0 && d1x !== 0) {
              return false;
            }

            if (d2y !== 0 && d2x !== 0) {
              return false;
            }

            if (Math.abs(d1y + d2y) !== 1 || Math.abs(d1x + d2x) !== 1) {
              return false;
            }

            if (
              history.at(-7).playedPoint.y !== history.at(-5).playedPoint.y + d1y ||
              history.at(-7).playedPoint.x !== history.at(-5).playedPoint.x + d1x
            ) {
              return false;
            }

            if (
              history.at(-9).playedPoint.y !== history.at(-7).playedPoint.y + d2y ||
              history.at(-9).playedPoint.x !== history.at(-7).playedPoint.x + d2x
            ) {
              return false;
            }

            const point = history.at(-1).playedPoint;
            const group = history.at(-1).groupAt(point.y, point.x);

            if (group.length !== 6) {
              return false;
            }

            if (group.filter((intersection) => history.at(-2).inAtari(intersection.y, intersection.x)).length !== 5) {
              return false;
            }

            console.log(`Match at ${history.at(-1).moveNumber}`);

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
