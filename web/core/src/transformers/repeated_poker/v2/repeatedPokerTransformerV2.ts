import { defaultGetStepRenderTime } from '../../../timing';
import { InterestingEvent, ReplayMode } from '../../../types';
import { PokerReplay, PokerReplayStepHistoryParsed } from './poker-replay-types';
import { RepeatedPokerStep } from './poker-steps-types';

import { createVisualStepsFromRepeatedPokerReplay } from './repeatedPokerTransformerUtils';

export const repeatedPokerTransformerV2 = (environment: any) => {
  const repeatedPokerReplay: PokerReplay = environment as PokerReplay;
  const repeatedPokerReplayStateHistory = environment.info.stateHistory;
  const agents = environment.info.Agents;

  const parsedStateHistorySteps = repeatedPokerReplayStateHistory.map((stateHistory: string) => {
    const stateHistoryObject = JSON.parse(stateHistory);
    stateHistoryObject['current_universal_poker_json'] = JSON.parse(stateHistoryObject['current_universal_poker_json']);
    if (stateHistoryObject['prev_universal_poker_json']) {
      stateHistoryObject['prev_universal_poker_json'] = JSON.parse(stateHistoryObject['prev_universal_poker_json']);
    }
    return stateHistoryObject;
  });
  const onlyPlayerSteps = repeatedPokerReplay.steps.flat().filter((step) => step.action.submission > -1);

  let stepToMapIndex = 0;
  const stateHistoryStepsWithReplaySteps: PokerReplayStepHistoryParsed[] = parsedStateHistorySteps.map(
    (parsedStateHistoryStep: PokerReplayStepHistoryParsed) => {
      if (parsedStateHistoryStep.current_universal_poker_json.current_player > -1) {
        parsedStateHistoryStep.step = onlyPlayerSteps[stepToMapIndex];
        parsedStateHistoryStep.stepIndex = stepToMapIndex;
        stepToMapIndex++;
      }
      return parsedStateHistoryStep;
    }
  );

  // this is a thing you can comment out to see the input/output for the first full game + 1 step in the input
  // console.log(stateHistoryStepsWithReplaySteps.slice(0,13));
  // console.log(createVisualStepsFromRepeatedPokerReplay(stateHistoryStepsWithReplaySteps.slice(0,13), agents));

  return createVisualStepsFromRepeatedPokerReplay(stateHistoryStepsWithReplaySteps, agents);
};

export const getPokerStepRenderTime = (
  gameStep: RepeatedPokerStep,
  replayMode: ReplayMode,
  speedModifier: number,
  defaultDuration?: number
) => {
  const defaultTime = defaultGetStepRenderTime(gameStep, replayMode, speedModifier, defaultDuration);
  const player = gameStep.players.find((player) => player.isTurn);

  switch (gameStep.stepType) {
    case 'small-blind-post':
    case 'big-blind-post':
    case 'deal-flop':
      return defaultTime * 0.5;
    case 'deal-player-hands':
    case 'deal-turn':
    case 'deal-river':
      return defaultTime * 0.75;
    case 'player-action':
      if (player && (player.actionDisplayText?.includes('Fold') || player.actionDisplayText?.includes('Check'))) {
        return defaultTime * 0.75;
      }
      return defaultTime;
    case 'final':
      return defaultTime * 1.2;
    default:
      return defaultTime;
  }
};

export const getPokerStepInterestingEvents = (gameSteps: RepeatedPokerStep[]): InterestingEvent[] => {
  const interestingEvents: InterestingEvent[] = [];
  const largePotIndices = new Set(gameSteps.filter((s) => s.pot >= 300).map((s) => s.currentHandIndex));
  let lastHandIndex = -1;

  for (const step of gameSteps) {
    if (step.currentHandIndex > lastHandIndex) {
      if (largePotIndices.has(step.currentHandIndex)) {
        interestingEvents.push({
          step: step.step,
          description: `Big Pot`,
        });
      }
      lastHandIndex = step.currentHandIndex;
    }
  }

  return interestingEvents;
};
