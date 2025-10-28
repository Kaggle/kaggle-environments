import { getPokerStepsWithEndStates } from "./transformers/repeatedPokerTransformer";
import { GameStep } from "./types";

export const processEpisodeData =  (steps: any[], stateHistory: any[], players: any[], gameName: string): GameStep[] => {
    switch(gameName) {
        case "repeated_poker":
        return getPokerStepsWithEndStates(steps, stateHistory, players);
        default:
            return steps;
    }
}