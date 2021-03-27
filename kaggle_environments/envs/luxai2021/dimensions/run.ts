import { create, Logger, Match } from 'dimensions-ai';
import os from 'os';
import fs from 'fs';
import { RockPaperScissorsDesign } from './rps';
let rpsDesign = new RockPaperScissorsDesign('RPS', {
  engineOptions: {
    memory: {
      active: true,
      limit: 1024 * 1024 * 400,
    },
    timeout: {
      active: true,
      max: 2000,
    },
  },
});

let myDimension = create(rpsDesign, {
  name: 'RPS',
  loggingLevel: Logger.LEVEL.NONE,
  activateStation: false,
  observe: false,
  // id: 'oLBptg',
});
import readline from 'readline';
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

let match: Match = null;

rl.on('line', async (line) => {
    // console.log(line);
    const json = JSON.parse(line);
    console.log(JSON.stringify({json}))

    // initialize a match
    if (json.type && json.type === "start") {
      fs.writeFileSync('initstate.log',line);
      match = await myDimension.createMatch([{
        file: "blank",
        name: "bot1"
      }, {
        file: "blank",
        name: "bot2"
      }], {
        detached: true,
        agentOptions: {detached: true},
        bestOf: json.config.episodeSteps
      });
    } else if (json.length) {
      // perform a step in the match
      match.step([{agentID: 0, command: mapNumToRPS(json[0].action)}, {agentID: 1, command: mapNumToRPS(json[1].action)}])
      fs.writeFileSync('state.log',`${JSON.stringify(match.state)}`);
    }
    fs.writeFileSync('log.log',`${line}`);
    // console.log(json);
})

const mapNumToRPS = (n: number) => {
  switch (n) {
    case 0:
      return "R";
    case 1:
      return "P";
    case 2:
      return "S";
  }
  return "R";
}