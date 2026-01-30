/**
 * Shoutout to Riley Jones for the first implementations of token streaming helpers. Originals are at:
 * https://github.com/rileyajones/kaggle-gamearena-gamestream-ui/blob/main/ui/src/context/utils.ts
 */
import { BaseGamePlayer, BaseGameStep, ReplayMode } from './types';

const TIME_PER_CHUNK = 120;
const EASING_INTENSITY = 2;

/**
 * Generates a list of numbers following an "ease-in-out" distribution.
 *
 * @param length The desired number of values to generate.
 * @param sum The target sum of the generated values.
 * @returns An array of numbers that follow the ease-in-out distribution and sum.
 */
const generateDistribution = (
  distribution: (normalizedTime: number, power: number) => number,
  length: number,
  sum: number
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
    const value = distribution(normalizedTime, EASING_INTENSITY);
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
  return normalizedTime < 0.5 ? 2 * Math.pow(normalizedTime, power) : 1 - Math.pow(-2 * normalizedTime + 2, power) / 2;
};

// Ease-in function that starts slow and gradually accelerates
const easeIn = (normalizedTime: number, power: number): number => {
  return Math.pow(normalizedTime, power);
};

export const generateEaseInOutDelayDistribution = (chunkCount: number): number[] => {
  const totalTime = TIME_PER_CHUNK * chunkCount;
  return generateDistribution(easeInOut, chunkCount, totalTime);
};

export const generateEaseInDelayDistribution = (chunkCount: number): number[] => {
  const totalTime = TIME_PER_CHUNK * chunkCount;
  return generateDistribution(easeIn, chunkCount, totalTime);
};

/**
 * By default, have each token render with an even amount of time between them
 */
export const generateDefaultDelayDistribution = (chunkCount: number): number[] => {
  return Array(chunkCount).fill(TIME_PER_CHUNK);
};

/**
 * Determine how long a turn is based on how long it takes to render each chunk.
 */
export function defaultGetStepRenderTime(
  gameStep: BaseGameStep,
  replayMode: ReplayMode,
  speedModifier: number,
  defaultDuration?: number
) {
  const stepDuration = defaultDuration ?? 2200;
  // Example: if we're at 2x speed, we want the render time to be half as long
  const multiplier = 1 / speedModifier;

  let currentPlayer: BaseGamePlayer = {
    id: -1,
    name: 'System',
    thumbnail: '',
    isTurn: false,
  };

  if (!gameStep.players) {
    return stepDuration * multiplier;
  }

  gameStep.players.forEach((player) => {
    if (player.isTurn) {
      currentPlayer = player;
    }
  });

  // If we should be streaming reasoning, we want the total render time to
  // account for how long it takes each token to be displayed
  if (replayMode !== 'condensed') {
    if (currentPlayer.thoughts) {
      const chunks = currentPlayer.thoughts.split(' ');
      return chunks.length * TIME_PER_CHUNK * multiplier;
    }
  }

  return stepDuration * multiplier;
}
