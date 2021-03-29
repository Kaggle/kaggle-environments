import { create, Logger, Match } from "dimensions-ai";
import readline from "readline";
import fs from "fs";
import { RockPaperScissorsDesign } from "./rps";
let rpsDesign = new RockPaperScissorsDesign("RPS");

let myDimension = create(rpsDesign, {
  name: "RPS",
  loggingLevel: Logger.LEVEL.NONE,
  activateStation: false,
  observe: false,
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false,
});

const main = async () => {
  let match: Match = null;
  for await (const line of rl) {
    const json = JSON.parse(line);

    // initialize a match
    if (json.type && json.type === "start") {
      match = await myDimension.createMatch(
        [
          {
            file: "blank",
            name: "bot1",
          },
          {
            file: "blank",
            name: "bot2",
          },
        ],
        {
          detached: true,
          agentOptions: { detached: true },
          bestOf: json.config.episodeSteps - 1,
          storeReplay: false,
          storeErrorLogs: false,
        }
      );
    } else if (json.length) {
      // perform a step in the match
      await match.step([
        { agentID: 0, command: mapNumToRPS(json[0].action) },
        { agentID: 1, command: mapNumToRPS(json[1].action) },
      ]);
      // log the match state back to kaggle's interpreter
      console.log(JSON.stringify(match.state));
    }
  }
};

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
};

main();
