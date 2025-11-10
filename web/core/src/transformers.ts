import { defaultGetStepRenderTime } from './timing';
import { chessTransformer, getChessStepDescription, getChessStepLabel } from './transformers/chess/chessTransformer';
import { ChessStep } from './transformers/chess/chessReplayTypes';
import { getPokerStepDescription, getPokerStepLabel } from './transformers/repeated_poker/v1/repeatedPokerTransformer';
import { RepeatedPokerStep } from './transformers/repeated_poker/v2/poker-steps-types';
import {
  getPokerStepRenderTime,
  repeatedPokerTransformerV2,
} from './transformers/repeated_poker/v2/repeatedPokerTransformerV2';
import { BaseGamePlayer, BaseGameStep, ReplayMode } from './types';

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

export const processEpisodeData = (environment: any, gameName: string): BaseGameStep[] => {
  switch (gameName) {
    case 'repeated_poker':
      return repeatedPokerTransformerV2(environment);
    case 'chess':
      return chessTransformer(environment);
    default:
      return [];
  }
};

/**
 * A top level summary of the step. Usually the action taken
 * by the player whose turn it is.
 */
export const getGameStepLabel = (gameStep: BaseGameStep, gameName: string): string => {
  switch (gameName) {
    case 'repeated_poker':
      return getPokerStepLabel(gameStep as RepeatedPokerStep);
    case 'chess':
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
    case 'repeated_poker':
      return getPokerStepDescription(gameStep as RepeatedPokerStep);
    case 'chess':
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
    case 'repeated_poker':
      return getPokerStepRenderTime(gameStep as RepeatedPokerStep, replayMode, speedModifier);
    default:
      return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, defaultDuration);
  }
};
