import { defaultGetStepRenderTime, generateDefaultDelayDistribution } from '../timing/timing';
import { BaseGameStep, InterestingEvent, ReplayData, ReplayMode } from '../types';

const defaultGetGameStepLabel = (gameStep: BaseGameStep) => {
  if (!gameStep.players) {
    return '';
  }
  const activePlayer = gameStep.players.find((player) => player.isTurn);
  return activePlayer?.actionDisplayText ?? '';
};

const defaultGetGameStepDescription = (gameStep: BaseGameStep) => {
  if (!gameStep.players) {
    return '';
  }

  const activePlayer = gameStep.players.find((player) => player.isTurn);
  return activePlayer?.thoughts ?? '';
};

export const processEpisodeData = (environment: ReplayData, _gameName: string): ReplayData<BaseGameStep[]> => {
  // Check for a marker to see if it's already been transformed.
  if (environment.isTransformed) {
    return environment;
  }

  // Game-specific transformers now live in each game's visualizer directory
  // and are passed to ReplayAdapter via the `transformer` option.
  return environment;
};

/**
 * A top level summary of the step. Usually the action taken
 * by the player whose turn it is.
 */
export const getGameStepLabel = (gameStep: BaseGameStep, _gameName: string): string => {
  return defaultGetGameStepLabel(gameStep);
};

/**
 * More details on what happened during the step. Usually
 * the thoughts from the current player.
 */
export const getGameStepDescription = (gameStep: BaseGameStep, _gameName: string): string => {
  return defaultGetGameStepDescription(gameStep);
};

const GAME_STEP_DURATIONS: Record<string, number> = {
  orbit_wars: 550, // 2200 / 4 — real-time strategy plays best at 4x speed
};

export const getGameStepRenderTime = (
  gameStep: BaseGameStep,
  gameName: string,
  replayMode: ReplayMode,
  speedModifier: number,
  defaultDuration?: number
): number => {
  const duration = defaultDuration ?? GAME_STEP_DURATIONS[gameName];
  return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, duration);
};

export const getInterestingEvents = (_gameSteps: BaseGameStep[], _gameName: string): InterestingEvent[] => {
  return [];
};

export const getTokenRenderDistribution = (chunkCount: number, _gameName: string): number[] => {
  return generateDefaultDelayDistribution(chunkCount);
};
