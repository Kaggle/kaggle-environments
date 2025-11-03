import { getPokerStepDescription, getPokerStepLabel } from './transformers/repeated_poker/v1/repeatedPokerTransformer';
import { RepeatedPokerStep } from './transformers/repeated_poker/v2/poker-steps-types';
import { repeatedPokerTransformerV2 } from './transformers/repeated_poker/v2/repeatedPokerTransformerV2';
import { BaseGameStep } from './types';

export const processEpisodeData = (environment: any, gameName: string): RepeatedPokerStep[] => {
  switch (gameName) {
    case 'repeated_poker':
      return repeatedPokerTransformerV2(environment);
    default:
      return environment.steps;
  }
};

export const getGameStepLabel = (gameStep: BaseGameStep, gameName: string): string => {
  switch (gameName) {
    case 'repeated_poker':
      return getPokerStepLabel(gameStep as RepeatedPokerStep);
    default:
      return '';
  }
};

export const getGameStepDescription = (gameStep: BaseGameStep, gameName: string): string => {
  switch (gameName) {
    case 'repeated_poker':
      return getPokerStepDescription(gameStep as RepeatedPokerStep);
    default:
      return '';
  }
};
