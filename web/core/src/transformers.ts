import {
  getPokerStepDescription,
  getPokerStepLabel,
  getPokerStepsWithEndStates,
} from "./transformers/repeatedPokerTransformer";
import { GameStep, PokerGameStep } from "./types";

export const processEpisodeData = (
  environment: any,
  gameName: string,
): GameStep[] => {
  switch (gameName) {
    case "repeated_poker":
      return getPokerStepsWithEndStates(environment);
    default:
      return environment.steps;
  }
};

export const getGameStepLabel = (
  gameStep: GameStep,
  gameName: string,
): string => {
  switch (gameName) {
    case "repeated_poker":
      return getPokerStepLabel(gameStep as PokerGameStep);
    default:
      return "";
  }
};

export const getGameStepDescription = (
  gameStep: GameStep,
  gameName: string,
): string => {
  switch (gameName) {
    case "repeated_poker":
      return getPokerStepDescription(gameStep as PokerGameStep);
    default:
      return "";
  }
};
