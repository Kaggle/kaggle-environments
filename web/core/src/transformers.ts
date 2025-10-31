import { getPokerStepDescription, getPokerStepLabel } from './transformers/repeated_poker/v1/repeatedPokerTransformer';
import { RepeatedPokerStep } from './transformers/repeated_poker/v2/poker-steps-types';
import { debugRepeatedPokerTransformer } from './transformers/repeated_poker/v2/repeatedPokerTransformerV2';
import { GameStep, PokerGameStep } from './types';

export const processEpisodeData = (environment: any, gameName: string): RepeatedPokerStep[] => {
  switch (gameName) {
    case 'repeated_poker':
      return debugRepeatedPokerTransformer(environment);
    default:
      return environment.steps;
  }
};

export const getGameStepLabel = (gameStep: GameStep, gameName: string): string => {
  switch (gameName) {
    case 'repeated_poker':
      return getPokerStepLabel(gameStep as PokerGameStep);
    default:
      return '';
  }
};

export const getGameStepDescription = (gameStep: GameStep, players: string[], gameName: string): string => {
  switch (gameName) {
    case 'repeated_poker':
      return getPokerStepDescription(gameStep as PokerGameStep, players);
    default:
      return '';
  }
};
