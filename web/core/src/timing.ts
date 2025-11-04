/**
 * Shoutout to Riley Jones for the first implementations of token streaming helpers. Originals are at:
 * https://github.com/rileyajones/kaggle-gamearena-gamestream-ui/blob/main/ui/src/context/utils.ts
 */
import { BaseGamePlayer, BaseGameStep, ReplayMode } from "./types";

const TIME_PER_CHUNK = 80;

/**
 * Generates a list of numbers following an "ease-in-out" distribution.
 * The distribution starts slowly, accelerates in the middle, and slows down at the end.
 *
 * @param length The desired number of values to generate.
 * @param sum The target sum of the generated values.
 * @returns An array of numbers that follow the ease-in-out distribution and sum.
 */
const generateEaseInOutDistribution = (
  length: number,
  sum: number,
  easingIntensity: number,
): number[] => {
  if (length <= 0) {
    return [];
  }
  // Handle the edge case of a single value.
  if (length === 1) {
    return [sum];
  }

  const rawValues: number[] = [];
  let rawSum = 0;

  // Generate the raw values from the easing curve
  for (let i = 0; i < length; i++) {
    const normalizedTime = i / (length - 1); // Safe now because length > 1
    const value = easeInOut(normalizedTime, easingIntensity);
    rawValues.push(value);
    rawSum += value;
  }

  // If the raw sum is zero, we can't scale. This shouldn't happen for ease-in-out with length > 1.
  if (rawSum === 0) {
    return Array(length).fill(0);
  }

  // Calculate the scaling factor
  const scale = sum / rawSum;

  // Scale the raw values to match the desired sum
  const finalValues = rawValues.map((value) => value * scale);

  return finalValues;
};

// Ease-in-out function that starts slow, speeds up in the middle, and slows down at the end
const easeInOut = (normalizedTime: number, power: number): number => {
  // This is a common ease-in-out formula
  return normalizedTime < 0.5
    ? 2 * Math.pow(normalizedTime, power)
    : 1 - Math.pow(-2 * normalizedTime + 2, power) / 2;
};

/**
 * Decide how long to wait between displaying each chunk so the user can read it.
 * Long term it would be nice to have this implemented at a per-game level.
 */
export const generateDelayDistribution = (
  chunkCount: number,
  easingIntensity: number = 2,
): number[] => {
  const totalTime = TIME_PER_CHUNK * chunkCount;
  return generateEaseInOutDistribution(chunkCount, totalTime, easingIntensity);
};

/**
 * Determine how long a turn is based on how long it takes to render each chunk.
 */
export function defaultGetStepRenderTime(
  gameStep: BaseGameStep,
  replayMode: ReplayMode,
  speedModifier: number,
  defaultDuration?: number,
) {
  const stepDuration = defaultDuration ?? 2000;
  // Example: if we're at 2x speed, we want the render time to be half as long
  const multiplier = 1 / speedModifier;

  let currentPlayer: BaseGamePlayer = {
    id: -1,
    name: "System",
    thumbnail: "",
    isTurn: false,
  };
  gameStep.players.forEach((player) => {
    if (player.isTurn) {
      currentPlayer = player;
    }
  });

  if (replayMode !== "condensed") {
    if (currentPlayer.thoughts) {
      const chunks = currentPlayer.thoughts.split(" ");
      // 250ms buffer allows a bit extra time for any UI elements to render
      return chunks.length * TIME_PER_CHUNK * multiplier + 250;
    }
  }

  return stepDuration * (1 / speedModifier);
}
