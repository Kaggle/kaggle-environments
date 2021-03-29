import { create, Logger, Match, MatchEngine } from "dimensions-ai";
import readline from "readline";
import fs from "fs";
import { LuxDesign } from "@lux-ai/2020-challenge";
let haliteLuxDesign = new LuxDesign("halite lux");

let myDimension = create(haliteLuxDesign, {
  name: "Halite Lux",
  loggingLevel: Logger.LEVEL.NONE,
  activateStation: false,
  observe: false
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
          storeReplay: false,
          storeErrorLogs: false,
          mapType: json.config.mapType
        }
      );
      match.agents.forEach((agent, i) => {
        console.log(JSON.stringify(agent.messages))
        fs.writeFileSync(`agent${i}.json`,JSON.stringify(agent.messages) + "\n");
        agent.messages = [];
      });
    } else if (json.length) {
      // perform a step in the match
      // console.log(JSON.stringify(json));
      let agents = [0, 1];
      let commandsList: Array<MatchEngine.Command> = [];
      agents.forEach((agentID) => {
        let agentCommands = json[agentID].action.map((action: string) => {
          return {agentID: agentID, command: action }
        });
        commandsList.push(...agentCommands);
      });
      // console.log(JSON.stringify(commandsList));
      fs.writeFileSync("cmd.log", JSON.stringify(commandsList));
      await match.step(commandsList);
      
      // log the match state back to kaggle's interpreter
      match.agents.forEach((agent, i) => {
        console.log(JSON.stringify(agent.messages))
        fs.appendFileSync(`agent${i}.json`,JSON.stringify(agent.messages)+ "\n");
        agent.messages = [];
      });
      match.state.game.map
      fs.writeFileSync("state.json", JSON.stringify(match.state.game.map, (k, v) => {
        if (k === "configs") return undefined;
        return v;
      }));
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
