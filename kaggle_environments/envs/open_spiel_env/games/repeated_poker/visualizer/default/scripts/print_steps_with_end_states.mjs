#!/usr/bin/env node

import path from "path";
import { fileURLToPath, pathToFileURL } from "url";
import { readFile } from "fs/promises";
import { createRequire } from "module";

async function loadCoreModules(repoRoot) {
  const require = createRequire(import.meta.url);
  const corePath = path.resolve(repoRoot, "web/core/dist/index.umd.cjs");
  const transformerPath = path.resolve(
    repoRoot,
    "web/core/dist/transformers/repeatedPokerTransformer.js"
  );

  const { processEpisodeData } = require(corePath);
  const { getPokerStepsWithEndStates } = await import(
    pathToFileURL(transformerPath).href
  );

  return { processEpisodeData, getPokerStepsWithEndStates };
}

function parseArgs(defaultReplayPath, defaultLimit, defaultJsonOutput) {
  const args = process.argv.slice(2);
  let replayPath = defaultReplayPath;
  let limit = defaultLimit;
  let asJson = defaultJsonOutput;

  const usage = `Usage: node ${path.basename(
    process.argv[1]
  )} [options]\n\nOptions:\n  -r, --replay <path>   Replay JSON to inspect (default: ${defaultReplayPath})\n  -l, --limit <count>   Number of entries to print (0 = all) (default: ${defaultLimit})\n  --json                Print raw JSON instead of formatted text\n  -h, --help            Show this help message\n`;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "-h" || arg === "--help") {
      console.log(usage);
      process.exit(0);
    } else if ((arg === "-r" || arg === "--replay") && args[i + 1]) {
      replayPath = path.resolve(args[i + 1]);
      i += 1;
    } else if ((arg === "-l" || arg === "--limit") && args[i + 1]) {
      const value = Number(args[i + 1]);
      if (!Number.isInteger(value) || value < 0) {
        console.error("Limit must be a non-negative integer.");
        console.log(usage);
        process.exit(1);
      }
      limit = value;
      i += 1;
    } else if (arg === "--json") {
      asJson = true;
    } else {
      console.error(`Unknown argument: ${arg}`);
      console.log(usage);
      process.exit(1);
    }
  }

  return { replayPath, limit, asJson };
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

function buildEnvironment(
  replay,
  processedSteps,
  processedInfo,
  processedConfig,
  stateHistory
) {
  return {
    configuration: processedConfig ?? replay.configuration ?? null,
    info: {
      ...(processedInfo || replay.info || {}),
      stateHistory,
    },
    steps: processedSteps,
    __processedSteps: processedSteps,
    __rawSteps: replay.steps,
  };
}

function describeStep(step, index) {
  const baseParts = [
    `Index ${index}`,
    `hand=${step.hand}`,
    `stateHistoryIndex=${step.stateHistoryIndex ?? "?"}`,
    `isEndState=${step.isEndState}`,
  ];

  if (step.postActionOf != null) {
    baseParts.push(`postActionOf=${step.postActionOf}`);
  }

  if (step.actingPlayer != null) {
    const label =
      step.actingPlayerName && step.actingPlayerName.length > 0
        ? `${step.actingPlayer}(${step.actingPlayerName})`
        : `${step.actingPlayer}`;
    baseParts.push(`actingPlayer=${label}`);
  }

  if (step.currentPlayer != null) {
    const label =
      step.currentPlayerName && step.currentPlayerName.length > 0
        ? `${step.currentPlayer}(${step.currentPlayerName})`
        : `${step.currentPlayer}`;
    baseParts.push(`currentPlayer=${label}`);
  }

  if (step.isEndState) {
    baseParts.push(`conclusion=${step.handConclusion ?? "-"}`);
    baseParts.push(`winner=${step.winner ?? "-"}`);
  } else if (step.step?.action?.actionString) {
    baseParts.push(`action=${step.step.action.actionString}`);
  } else if (step.actionText) {
    baseParts.push(`actionText=${step.actionText}`);
  }

  return baseParts.join(" | ");
}

async function main() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const repoRoot = path.resolve(__dirname, "../../../../../../../..");
  const defaultReplayPath = path.resolve(
    __dirname,
    "../replays/test-replay.json"
  );
  const defaultLimit = 10;
  const defaultJsonOutput = false;

  const { replayPath, limit, asJson } = parseArgs(
    defaultReplayPath,
    defaultLimit,
    defaultJsonOutput
  );

  const { processEpisodeData, getPokerStepsWithEndStates } =
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

  const processedResult = processEpisodeData(
    {
      steps: replay.steps,
      state_history: stateHistory,
      info: replay.info,
      configuration: replay.configuration,
    },
    "repeated_poker"
  );

  const processedSteps = Array.isArray(processedResult?.steps)
    ? processedResult.steps
    : [];

  if (processedSteps.length === 0) {
    console.error("Processed episode contains no steps.");
    process.exit(1);
  }

  const environment = buildEnvironment(
    replay,
    processedSteps,
    processedResult?.info,
    processedResult?.configuration,
    stateHistory
  );

  const stepsWithEndStates = getPokerStepsWithEndStates(environment);

  console.log(`Loaded replay: ${replayPath}`);
  console.log(
    `Derived ${stepsWithEndStates.length} entries via getPokerStepsWithEndStates`
  );

  if (asJson) {
    const subset =
      limit === 0 ? stepsWithEndStates : stepsWithEndStates.slice(0, limit);
    console.log(JSON.stringify(subset, null, 2));
    return;
  }

  const entriesToPrint =
    limit === 0 ? stepsWithEndStates : stepsWithEndStates.slice(0, limit);

  entriesToPrint.forEach((entry, index) => {
    console.log(describeStep(entry, index));
  });

  if (limit > 0 && stepsWithEndStates.length > limit) {
    console.log(
      `\n(${stepsWithEndStates.length - limit} additional entries not shown; rerun with -l 0 to display all or --json for raw output.)`
    );
  }
}

main().catch((error) => {
  console.error("Failed to print steps:", error);
  process.exit(1);
});
