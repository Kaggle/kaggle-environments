#!/usr/bin/env node
import fs from "fs";
import path from "path";
import process from "process";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const defaultReplayPath = path.resolve(
  repoRoot,
  "kaggle_environments/envs/open_spiel_env/games/repeated_poker/visualizer/default/replays/test-replay.json",
);

const args = process.argv.slice(2);
let replayPath = defaultReplayPath;
let maxHands = 2;

for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if ((arg === "-r" || arg === "--replay") && args[i + 1]) {
    replayPath = path.resolve(args[++i]);
  } else if ((arg === "-n" || arg === "--hands") && args[i + 1]) {
    maxHands = Number(args[++i]);
  } else if (arg === "-h" || arg === "--help") {
    console.log(`Usage: node scripts/inspect_repeated_poker.mjs [options]

Options:
  -r, --replay <path>   Replay JSON to inspect (default: ${defaultReplayPath})
  -n, --hands <count>   Number of hands to display (default: 2)
  -h, --help            Show this help message
`);
    process.exit(0);
  }
}

if (!fs.existsSync(replayPath)) {
  console.error(`Replay file not found: ${replayPath}`);
  process.exit(1);
}

const replay = JSON.parse(fs.readFileSync(replayPath, "utf8"));

const coreModulePath = path.resolve(
  repoRoot,
  "kaggle_environments/envs/open_spiel_env/games/repeated_poker/visualizer/default/node_modules/@kaggle-environments/core/dist/index.js",
);
const { processEpisodeData } = await import(`file://${coreModulePath}`);

const processedSteps = processEpisodeData(
  replay.steps,
  replay.info.stateHistory,
  "repeated_poker",
);

const environment = {
  ...replay,
  steps: processedSteps,
  __rawSteps: replay.steps,
};

const pokerStateModule = path.resolve(
  repoRoot,
  "kaggle_environments/envs/open_spiel_env/games/repeated_poker/visualizer/default/src/components/getRepeatedPokerStateForStep.js",
);

const { getPokerStateForStep } = await import(`file://${pokerStateModule}`);

const hands = new Map();
processedSteps.forEach((step, idx) => {
  if (!hands.has(step.hand)) hands.set(step.hand, []);
  hands.get(step.hand).push({ idx, step });
});

const sortedHandIds = Array.from(hands.keys()).sort((a, b) => a - b);

sortedHandIds.slice(0, maxHands).forEach((handId) => {
  console.log(`\n==== Hand ${handId} ====`);
  const handSteps = hands.get(handId) || [];
  handSteps.forEach(({ idx, step }) => {
    const uiState = getPokerStateForStep(environment, idx);
    const actionString = step.step?.action?.actionString ?? "END";
    console.log(
      `StepIdx ${idx} | isEnd=${step.isEndState} | action=${actionString} | currentPlayer=${uiState?.currentPlayer}`,
    );
    if (uiState) {
      uiState.players.forEach((player, seat) => {
        console.log(
          `  Player${seat}: stack=${player.stack} bet=${player.currentBet} actionDisplay='${player.actionDisplayText}' turn=${player.isTurn} dealer=${player.isDealer} cards=${player.cards.join(",")}`,
        );
      });
      console.log(
        `  community=[${(uiState.communityCards || []).join(",")}] pot=${uiState.pot}`,
      );
    } else {
      console.log("  uiState unavailable");
    }
  });
});
