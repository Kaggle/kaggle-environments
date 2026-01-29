import { defaultGetStepRenderTime, generateDefaultDelayDistribution, generateEaseInDelayDistribution } from './timing';
import { chessTransformer, getChessStepDescription, getChessStepLabel } from './transformers/chess/chessTransformer';
import { ChessStep } from './transformers/chess/chessReplayTypes';
import {
  connectFourTransformer,
  getConnectFourStepDescription,
  getConnectFourStepLabel,
} from './transformers/connect_four/connectFourTransformer';
import { ConnectFourStep } from './transformers/connect_four/connectFourReplayTypes';
import {
  werewolfTransformer,
  getWerewolfStepLabel,
  getWerewolfStepDescription,
  getWerewolfStepRenderTime,
  getWerewolfStepInterestingEvents,
} from './transformers/werewolf/werewolfTransformer';

// Re-export utility functions for external use
export {
  createNameReplacer,
  createPlayerCapsule,
  disambiguateDisplayNames,
} from './transformers/werewolf/nameReplacer';
export type { PlayerConfig, OutputFormat } from './transformers/werewolf/nameReplacer';
import { WerewolfStep } from './transformers/werewolf/werewolfReplayTypes';
import { getPokerStepDescription, getPokerStepLabel } from './transformers/repeated_poker/v1/repeatedPokerTransformer';
import { RepeatedPokerStep } from './transformers/repeated_poker/v2/poker-steps-types';
import {
  getPokerStepFromUrlParams,
  getPokerStepInterestingEvents,
  getPokerStepRenderTime,
  processPokerFile,
  repeatedPokerTransformerV2,
} from './transformers/repeated_poker/v2/repeatedPokerTransformerV2';
import { BaseGameStep, EpisodeSlice, InterestingEvent, ReplayData, ReplayMode } from './types';

const defaultGetGameStepLabel = (gameStep: BaseGameStep) => {
  const activePlayer = gameStep.players.find((player) => player.isTurn);
  return activePlayer?.actionDisplayText ?? '';
};

const defaultGetGameStepDescription = (gameStep: BaseGameStep) => {
  const activePlayer = gameStep.players.find((player) => player.isTurn);
  return activePlayer?.thoughts ?? '';
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
    case 'open_spiel_connect_four':
      transformedSteps = connectFourTransformer(environment);
      break;
    case 'werewolf':
      // Werewolf transformer modifies environment in place and adds visualizerData
      return werewolfTransformer(environment) as any;
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
    case 'open_spiel_connect_four':
      return getConnectFourStepLabel(gameStep as ConnectFourStep);
    case 'werewolf':
      return getWerewolfStepLabel(gameStep as unknown as WerewolfStep);
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
    case 'open_spiel_connect_four':
      return getConnectFourStepDescription(gameStep as ConnectFourStep);
    case 'werewolf':
      return getWerewolfStepDescription(gameStep as unknown as WerewolfStep);
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
    case 'werewolf':
      return getWerewolfStepRenderTime(gameStep, replayMode, speedModifier);
    default:
      return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, defaultDuration);
  }
};

export const getStepFromUrlParams = (params: URLSearchParams, gameName: string, gameSteps: BaseGameStep[]): number => {
  switch (gameName) {
    case 'open_spiel_repeated_poker':
      return getPokerStepFromUrlParams(params, gameSteps as RepeatedPokerStep[]);
    default:
      return params.get('step') === null ? -1 : Number(params.get('step'));
  }
};

export const getInterestingEvents = (gameSteps: BaseGameStep[], gameName: string): InterestingEvent[] => {
  switch (gameName) {
    case 'open_spiel_repeated_poker':
      return getPokerStepInterestingEvents(gameSteps as RepeatedPokerStep[]);
    case 'werewolf':
      return getWerewolfStepInterestingEvents(gameSteps as unknown as WerewolfStep[]);
    default:
      return [];
  }
};

export const getEpisodesFromFile = async (file: File, gameName: string): Promise<EpisodeSlice[]> => {
  switch (gameName) {
    case 'open_spiel_repeated_poker': {
      const hands = await processPokerFile(file);
      return hands;
    }
    default:
      return [];
  }
};

export const getTokenRenderDistribution = (chunkCount: number, gameName: string): number[] => {
  switch (gameName) {
    case 'open_spiel_chess':
    case 'werewolf':
      return generateEaseInDelayDistribution(chunkCount);

    default:
      return generateDefaultDelayDistribution(chunkCount);
  }
};
