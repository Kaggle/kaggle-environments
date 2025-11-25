import { defaultGetStepRenderTime } from './timing';
import { chessTransformer, getChessStepDescription, getChessStepLabel } from './transformers/chess/chessTransformer';
import { ChessStep } from './transformers/chess/chessReplayTypes';
import { getPokerStepDescription, getPokerStepLabel } from './transformers/repeated_poker/v1/repeatedPokerTransformer';
import { RepeatedPokerStep } from './transformers/repeated_poker/v2/poker-steps-types';
import {
  getPokerStepFromUrlParams,
  getPokerStepInterestingEvents,
  getPokerStepRenderTime,
  repeatedPokerTransformerV2,
} from './transformers/repeated_poker/v2/repeatedPokerTransformerV2';
import { BaseGamePlayer, BaseGameStep, InterestingEvent, ReplayData, ReplayMode } from './types';

const defaultGetGameStepLabel = (gameStep: BaseGameStep) => {
  let i = 0;
  while (i < gameStep.players.length) {
    const player: BaseGamePlayer = gameStep.players[i];
    if (player.isTurn) {
      return player.actionDisplayText ?? '';
    }
    i++;
  }
  return '';
};

const defaultGetGameStepDescription = (gameStep: BaseGameStep) => {
  let i = 0;
  while (i < gameStep.players.length) {
    const player: BaseGamePlayer = gameStep.players[i];
    if (player.isTurn) {
      return player.thoughts ?? '';
    }
    i++;
  }
  return '';
};

export const processEpisodeData = (environment: ReplayData, gameName: string): ReplayData<BaseGameStep[]> => {
  // Check for a marker to see if it's already been transformed.
  if (environment.isTransformed) {
    return environment;
  }

  let transformedSteps: BaseGameStep[] = [];
  switch (gameName) {
    case 'open_spiel_repeated_poker':
      transformedSteps = repeatedPokerTransformerV2(environment);
      break;
    case 'open_spiel_chess':
      transformedSteps = chessTransformer(environment);
      break;
    default:
      // If no transformer, return the original environment
      return environment;
  }

  // Replace the steps, add the marker, and return the modified environment.
  environment.steps = transformedSteps;
  environment.isTransformed = true;
  return environment;
};

/**
 * A top level summary of the step. Usually the action taken
 * by the player whose turn it is.
 */
export const getGameStepLabel = (gameStep: BaseGameStep, gameName: string): string => {
  switch (gameName) {
    case 'open_spiel_repeated_poker':
      return getPokerStepLabel(gameStep as RepeatedPokerStep);
    case 'open_spiel_chess':
      return getChessStepLabel(gameStep as ChessStep);
    default:
      return defaultGetGameStepLabel(gameStep);
  }
};

/**
 * More details on what happened during the step. Usually
 * the thoughts from the current player.
 */
export const getGameStepDescription = (gameStep: BaseGameStep, gameName: string): string => {
  switch (gameName) {
    case 'open_spiel_repeated_poker':
      return getPokerStepDescription(gameStep as RepeatedPokerStep);
    case 'open_spiel_chess':
      return getChessStepDescription(gameStep as ChessStep);
    default:
      return defaultGetGameStepDescription(gameStep);
  }
};

export const getGameStepRenderTime = (
  gameStep: BaseGameStep,
  gameName: string,
  replayMode: ReplayMode,
  speedModifier: number,
  defaultDuration?: number
): number => {
  switch (gameName) {
    case 'open_spiel_repeated_poker':
      return getPokerStepRenderTime(gameStep as RepeatedPokerStep, replayMode, speedModifier);
    default:
      return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, defaultDuration);
  }
};

export const getStepFromUrlParams = (params: URLSearchParams, gameName: string, gameSteps: BaseGameStep[]): number => {
  switch (gameName) {
    case 'open_spiel_repeated_poker':
      return getPokerStepFromUrlParams(params, gameSteps as RepeatedPokerStep[]);
    default:
      return Number(params.get('step'));
  }
};

export const getInterestingEvents = (gameSteps: BaseGameStep[], gameName: string): InterestingEvent[] => {
  switch (gameName) {
    case 'open_spiel_repeated_poker':
      return getPokerStepInterestingEvents(gameSteps as RepeatedPokerStep[]);
    default:
      return [];
  }
};
