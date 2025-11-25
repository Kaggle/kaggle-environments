import { defaultGetStepRenderTime } from '../../../timing';
import { InterestingEvent, ReplayMode } from '../../../types';
import { PokerReplay, PokerReplayStepHistoryParsed } from './poker-replay-types';
import { RepeatedPokerStep, RepeatedPokerStepPlayer } from './poker-steps-types';

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
  const largePotIndices = new Set(
    gameSteps
      .filter((s) => {
        return s.pot >= 350 && s.pot < 400;
      })
      .map((s) => s.currentHandIndex)
  );
  const showdownIndices = new Set(gameSteps.filter((s) => s.pot === 400).map((s) => s.currentHandIndex));
  let lastHandIndex = -1;

  for (const step of gameSteps) {
    const hasLargePot = largePotIndices.has(step.currentHandIndex);
    const hasShowdown = showdownIndices.has(step.currentHandIndex);

    // Check for high hands
    const highHandTypes = ['Full House', 'Four of a Kind', 'Straight Flush', 'Royal Flush'];
    const handSteps = gameSteps.filter((s) => s.currentHandIndex === step.currentHandIndex);
    const hasHighHand = handSteps.some((s) => s.bestHandRankTypes?.some((rank) => highHandTypes.includes(rank)));

    // Check for upsets (winner had < 20% odds on turn or river)
    const turnOrRiverStep = handSteps.find((s) => s.stepType === 'deal-turn' || s.stepType === 'deal-river');
    const winnerIndex =
      handSteps
        .find((s) => s.stepType === 'final')
        ?.players.findIndex((p) => (p as RepeatedPokerStepPlayer).isWinner) ?? -1;
    const hasUpset = turnOrRiverStep && turnOrRiverStep.winOdds && turnOrRiverStep.winOdds[winnerIndex * 2] < 0.2;

    if (step.currentHandIndex > lastHandIndex) {
      if (hasUpset) {
        interestingEvents.push({
          step: step.step,
          description: 'Upset',
        });
      } else if (hasHighHand) {
        interestingEvents.push({
          step: step.step,
          description: 'High Hand',
        });
      } else if (hasLargePot) {
        interestingEvents.push({
          step: step.step,
          description: 'Big Pot',
        });
      } else if (hasShowdown) {
        interestingEvents.push({
          step: step.step,
          description: 'All-in Showdown',
        });
      }
      lastHandIndex = step.currentHandIndex;
    }
  }

  return interestingEvents;
};

export const getPokerStepFromUrlParams = (params: URLSearchParams, gameSteps: RepeatedPokerStep[]): number => {
  const hand = params.get('hand');

  if (hand !== null) {
    // Assume the hand URL param is 1-indexed
    return gameSteps.findIndex((s) => s.currentHandIndex === Number(hand) - 1);
  }

  return params.get('step') === null ? -1 : Number(params.get('step'));
};
