import fs from 'fs';
import https from 'https';
import { createGunzip } from 'zlib';

if (!process.env.REPLAY_DOWNLOAD_FOLDER) process.exit();

function kaggleApi(path, data) {
  return new Promise((resolve, reject) => {
    const url = `https://www.kaggle.com/api${path}`;
    const options = {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'content-length': Buffer.byteLength(data),
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
    benchmarkTaskVersionFilter: {
      benchmarkModelVersionId: 57,
      benchmarkTaskVersionId: 396,
    },
    successfulOnly: true,
  })
);

const teams = [];
const downloads = [];

shuffle(list.episodes);

for (const episode of list.episodes) {
  for (const agent of episode.agents) {
    const sub = list.submissions.find((item) => item.id === agent.submissionId);
    const team = list.teams.find((item) => item.id === sub.teamId);

    if (teams.includes(team.teamName) === false) {
      downloads.push(episode.id);
      teams.push(team.teamName);
      break;
    }
  }
}

for (const id of downloads) {
  const replay = await kaggleApi(
    '/i/competitions.EpisodeService/GetEpisodeReplay',
    JSON.stringify({
      episodeId: id,
    })
  );

  console.log(replay.info.TeamNames);

  const teamNames = replay.info.TeamNames.join('-').replace(/ /g, '');
  const folderName = process.env.REPLAY_DOWNLOAD_FOLDER;
  const fileName = `${folderName}/${id}-${teamNames}.json`;

  fs.mkdirSync(folderName, { recursive: true });
  fs.writeFileSync(fileName, JSON.stringify(replay, null, 2));
}
