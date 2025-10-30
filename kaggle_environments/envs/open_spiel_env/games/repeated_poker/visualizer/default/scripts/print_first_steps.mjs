#!/usr/bin/env node

import path from "path";
import { fileURLToPath } from "url";
import { readFile } from "fs/promises";

function parseArgs(defaultReplayPath, defaultStepLimit) {
  const args = process.argv.slice(2);
  let replayPath = defaultReplayPath;
  let stepLimit = defaultStepLimit;

  const usage = `Usage: node ${path.basename(
    process.argv[1]
  )} [options]\n\nOptions:\n  -r, --replay <path>   Replay JSON to inspect (default: ${defaultReplayPath})\n  -s, --steps <count>   Number of steps to print (default: ${defaultStepLimit})\n  -h, --help            Show this help message\n`;

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

function truncate(text, maxLength = 120) {
  if (typeof text !== "string") {
    return text;
  }
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 3)}...`;
}

function summarizeAction(action) {
  if (action == null) {
    return "-";
  }

  if (typeof action !== "object") {
    return String(action);
  }

  const parts = [];
  if (Object.prototype.hasOwnProperty.call(action, "actionString")) {
    parts.push(`actionString=${action.actionString}`);
  }
  if (Object.prototype.hasOwnProperty.call(action, "submission")) {
    parts.push(`submission=${action.submission}`);
  }
  if (Object.prototype.hasOwnProperty.call(action, "status") && action.status) {
    parts.push(`status=${truncate(action.status, 60)}`);
  }

  if (parts.length > 0) {
    return parts.join(" | ");
  }

  return truncate(JSON.stringify(action));
}

function summarizeObservation(observation) {
  if (observation == null) {
    return "-";
  }

  if (typeof observation !== "object") {
    return String(observation);
  }

  const parts = [];
  if (Object.prototype.hasOwnProperty.call(observation, "step")) {
    parts.push(`step=${observation.step}`);
  }
  if (Object.prototype.hasOwnProperty.call(observation, "currentPlayer")) {
    parts.push(`currentPlayer=${observation.currentPlayer}`);
  }
  if (Object.prototype.hasOwnProperty.call(observation, "isTerminal")) {
    parts.push(`isTerminal=${observation.isTerminal}`);
  }
  if (
    Object.prototype.hasOwnProperty.call(observation, "observationString") &&
    observation.observationString
  ) {
    parts.push(
      `observationString="${truncate(observation.observationString, 80)}"`
    );
  }
  if (
    Object.prototype.hasOwnProperty.call(
      observation,
      "serializedGameAndState"
    ) &&
    observation.serializedGameAndState
  ) {
    parts.push("serializedGameAndState=<omitted>");
  }

  if (parts.length > 0) {
    return parts.join(" | ");
  }

  return truncate(JSON.stringify(observation));
}

async function main() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const defaultReplayPath = path.resolve(
    __dirname,
    "../replays/test-replay.json"
  );
  const defaultStepLimit = 5;

  const { replayPath, stepLimit } = parseArgs(
    defaultReplayPath,
    defaultStepLimit
  );

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

  const steps = Array.isArray(replay?.steps) ? replay.steps : [];
  if (steps.length === 0) {
    console.error("Replay does not contain any steps.");
    process.exit(1);
  }

  const limit = Math.min(stepLimit, steps.length);
  console.log(`Loaded replay: ${replayPath}`);
  console.log(`Printing first ${limit} of ${steps.length} steps\n`);

  for (let stepIndex = 0; stepIndex < limit; stepIndex += 1) {
    const step = steps[stepIndex];
    console.log(`Step ${stepIndex}`);

    if (!Array.isArray(step)) {
      console.log("  (step is not an array)\n");
      continue;
    }

    step.forEach((playerStep, playerIndex) => {
      if (!playerStep || typeof playerStep !== "object") {
        console.log(`  Player ${playerIndex}: <invalid step payload>`);
        return;
      }

      const { action, observation, reward, status } = playerStep;
      const actionSummary = summarizeAction(action);
      const observationSummary = summarizeObservation(observation);
      const rewardSummary =
        reward == null ? "-" : JSON.stringify(reward, null, 0);
      const statusSummary = status ?? "-";

      console.log(
        `  Player ${playerIndex}: status=${statusSummary} | reward=${rewardSummary}`
      );
      console.log(`    action: ${actionSummary}`);
      console.log(`    observation: ${observationSummary}`);
    });

    console.log("");
  }
}

main().catch((error) => {
  console.error("Failed to print steps:", error);
  process.exit(1);
});
