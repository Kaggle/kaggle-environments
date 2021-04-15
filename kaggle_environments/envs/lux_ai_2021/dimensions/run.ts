import { create, Logger, Match, MatchEngine } from "dimensions-ai";
import readline from "readline";
import {
  LuxDesign,
  LuxMatchConfigs,
  LuxMatchState,
} from "@lux-ai/2020-challenge";
import { DeepPartial } from "dimensions-ai/lib/main/utils/DeepPartial";
import fs from 'fs';
let haliteLuxDesign = new LuxDesign("lux_ai_2021");

let myDimension = create(haliteLuxDesign, {
  name: "Lux AI 2021",
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
      const configs: DeepPartial<LuxMatchConfigs & Match.Configs> = {
        detached: true,
        agentOptions: { detached: true },
        storeReplay: json.config.saveReplays,
        // storeReplay: false,
        storeErrorLogs: false,
        mapType: json.config.mapType,
        parameters: {
          MAX_DAYS: json.config.episodeSteps - 2,
        },
      };
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
        configs
      );
      match.agents.forEach((agent, i) => {
        console.log(JSON.stringify(agent.messages));
        agent.messages = [];
      });
      
      // fs.writeFileSync(`match_${match.id}.log`, ("Initiated"));
    } else if (json.length) {
      // perform a step in the match
      // console.log(JSON.stringify(json));
      let agents = [0, 1];
      let commandsList: Array<MatchEngine.Command> = [];
      agents.forEach((agentID) => {
        let agentCommands = json[agentID].action.map((action: string) => {
          return { agentID: agentID, command: action };
        });
        commandsList.push(...agentCommands);
      });
      const status = await match.step(commandsList);
      // fs.appendFileSync(`match_${match.id}.log`, `\nstep ${match.state.game.state.turn}`);

      // log the match state back to kaggle's interpreter
      match.agents.forEach((agent, i) => {
        console.log(JSON.stringify(agent.messages));
        agent.messages = [];
      });

      // tell kaggle interpreter about match status
      const state: LuxMatchState = match.state;
      console.log(
        JSON.stringify({
          status: status,
          turn: state.game.state.turn,
          max: match.configs.parameters.MAX_DAYS,
        })
      );
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
