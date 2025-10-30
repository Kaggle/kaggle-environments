#!/usr/bin/env node

import path from "path";
import { fileURLToPath, pathToFileURL } from "url";
import { readFile } from "fs/promises";
import { createRequire } from "module";

async function loadCoreModules(repoRoot) {
  const require = createRequire(import.meta.url);
  const corePath = path.resolve(repoRoot, "web/core/dist/index.umd.cjs");
  const timelinePath = path.resolve(
    repoRoot,
    "web/core/dist/transformers/buildTimeline.js"
  );

  const { processEpisodeData } = require(corePath);
  const { buildTimeline, getPokerStateForStep } = await import(
    pathToFileURL(timelinePath).href
  );

  return { processEpisodeData, buildTimeline, getPokerStateForStep };
}

function parseArgs(defaultReplayPath, defaultStepLimit) {
  const args = process.argv.slice(2);
  let replayPath = defaultReplayPath;
  let stepLimit = defaultStepLimit;

  const usage = `Usage: node ${path.basename(
    process.argv[1]
  )} [options]\n\nOptions:\n  -r, --replay <path>   Replay JSON to inspect (default: ${defaultReplayPath})\n  -s, --steps <count>   Number of timeline steps to print (default: ${defaultStepLimit})\n  -h, --help            Show this help message\n`;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "-h" || arg === "--help") {
      console.log(usage);
      process.exit(0);
    } else if ((arg === "-r" || arg === "--replay") && args[i + 1]) {
      replayPath = path.resolve(args[i + 1]);
      i += 1;
    } else if ((arg === "-s" || arg === "--steps") && args[i + 1]) {
      const value = Number(args[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        console.error("Steps limit must be a positive integer.");
        console.log(usage);
        process.exit(1);
      }
      stepLimit = value;
      i += 1;
    } else {
      console.error(`Unknown argument: ${arg}`);
      console.log(usage);
      process.exit(1);
    }
  }

  return { replayPath, stepLimit };
}

function extractStateHistory(replay) {
  return (
    replay?.info?.stateHistory ??
    replay?.info?.state_history ??
    replay?.stateHistory ??
    replay?.state_history ??
    []
  );
}

function formatCommunityCards(cards) {
  if (!Array.isArray(cards) || cards.length === 0) {
    return "--";
  }
  return cards.join(" ");
}

function formatWinOdds(winOdds) {
  if (!Array.isArray(winOdds) || winOdds.length === 0) {
    return "--";
  }
  return winOdds.join(" vs ");
}

async function main() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const repoRoot = path.resolve(__dirname, "../../../../../../../..");
  const defaultReplayPath = path.resolve(
    __dirname,
    "../replays/test-replay.json"
  );
  const defaultStepLimit = 5;

  const { replayPath, stepLimit } = parseArgs(
    defaultReplayPath,
    defaultStepLimit
  );

  const { processEpisodeData, buildTimeline, getPokerStateForStep } =
    await loadCoreModules(repoRoot);

  let replayRaw;
  try {
    replayRaw = await readFile(replayPath, "utf-8");
  } catch (error) {
    console.error(`Failed to read replay at ${replayPath}: ${error.message}`);
    process.exit(1);
  }

  let replay;
  try {
    replay = JSON.parse(replayRaw);
  } catch (error) {
    console.error(
      `Replay file ${replayPath} is not valid JSON: ${error.message}`
    );
    process.exit(1);
  }

  const stateHistory = extractStateHistory(replay);
  if (!Array.isArray(stateHistory) || stateHistory.length === 0) {
    console.error("Replay does not contain a state history.");
    process.exit(1);
  }

  const processedSteps = processEpisodeData(
    {
      steps: replay.steps,
      state_history: stateHistory,
      info: replay.info,
      configuration: replay.configuration,
    },
    "repeated_poker"
  );

  const environment = {
    configuration: replay.configuration,
    info: {
      ...replay.info,
      stateHistory,
    },
    steps: processedSteps,
    __processedSteps: processedSteps,
    __rawSteps: replay.steps,
  };

  const timeline = buildTimeline(environment, 2);
  environment.__timeline = timeline;

  console.log(`Loaded replay: ${replayPath}`);
  console.log(
    `Printing first ${Math.min(stepLimit, timeline.length)} of ${
      timeline.length
    } timeline steps\n`
  );

  for (let stepIndex = 0; stepIndex < timeline.length; stepIndex += 1) {
    if (stepIndex >= stepLimit) {
      break;
    }

    const event = timeline[stepIndex];
    const uiState = getPokerStateForStep(environment, stepIndex);

    const headerParts = [
      `Step ${stepIndex}`,
      `stateIndex=${event?.stateIndex ?? "N/A"}`,
      `highlight=${event?.highlightPlayer ?? "-"}`,
      `action='${event?.actionText ?? ""}'`,
    ];
    if (event?.hideHoleCards) {
      headerParts.push("hideHoleCards");
    }
    if (event?.hideCommunity) {
      headerParts.push("hideCommunity");
    }

    console.log(headerParts.join(" | "));

    if (!uiState) {
      console.log("  (UI state unavailable)\n");
      continue;
    }

    console.log(
      `  pot=${uiState.pot} | community=${formatCommunityCards(
        uiState.communityCards
      )} | winOdds=${formatWinOdds(uiState.winOdds)}`
    );

    uiState.players.forEach((player, seat) => {
      const cards =
        Array.isArray(player.cards) && player.cards.length > 0
          ? player.cards.join(" ")
          : "--";
      const flags = [
        player.isDealer ? "D" : "",
        player.isTurn ? "T" : "",
        player.isLastActor ? "LA" : "",
      ]
        .filter(Boolean)
        .join(",") || "-";
      console.log(
        `    P${seat}: stack=${player.stack} bet=${player.currentBet} cards=${cards} flags=${flags} action='${player.actionDisplayText || ""}'`
      );
    });

    console.log("");
  }
}

main().catch((error) => {
  console.error("Failed to print replay steps:", error);
  process.exit(1);
});
