import { PokerReplay, PokerReplayInfoAgent, PokerReplayStepHistoryParsed } from './poker-replay-types';
import { RepeatedPokerStep, RepeatedPokerStepPlayer, RepeatedPokerStepType } from './poker-steps-types';

const getRepeatedPokerStepPlayers = (
  step: PokerReplayStepHistoryParsed,
  agents: PokerReplayInfoAgent[],
  stepType: RepeatedPokerStepType
) => {
  const playerZeroStackSize = stepType === 'small-blind-post' && step.dealer === 1 ? 200 - step.small_blind : 200;
  const playerZeroCurrentBet = stepType === 'small-blind-post' && step.dealer === 1 ? step.small_blind : 0;
  const playerZeroActionDisplayText = stepType === 'small-blind-post' && step.dealer === 1 ? 'SB' : '';
  const playerZero: RepeatedPokerStepPlayer = {
    id: 0,
    name: agents[0].Name,
    thumbnail: agents[0].ThumbnailUrl,
    cards: step.current_universal_poker_json.player_hands[0],
    chipStack: playerZeroStackSize,
    currentBet: playerZeroCurrentBet,
    actionDisplayText: playerZeroActionDisplayText,
    thoughts: step.step?.action?.thoughts ?? '',
    isDealer: step.dealer === 0,
    isTurn: false,
    isWinner: false
  };
  const playerOneStackSize = stepType === 'small-blind-post' && step.dealer === 0 ? 200 - step.small_blind : 200;
  const playerOneCurrentBet = stepType === 'small-blind-post' && step.dealer === 0 ? step.small_blind : 0;
  const playerOneActionDisplayText = stepType === 'small-blind-post' && step.dealer === 0 ? 'SB' : '';
  const playerOne: RepeatedPokerStepPlayer = {
    id: 1,
    name: agents[1].Name,
    thumbnail: agents[1].ThumbnailUrl,
    cards: step.current_universal_poker_json.player_hands[1],
    chipStack: playerOneStackSize,
    currentBet: playerOneCurrentBet,
    actionDisplayText: playerOneActionDisplayText,
    thoughts: step.step?.action?.thoughts ?? '',
    isDealer: step.dealer === 1,
    isTurn: false,
    isWinner: false
  };

  return [playerZero, playerOne];
};

export const debugRepeatedPokerTransformer = (environment: any) => {
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
  const stateHistoryStepsWithReplaySteps = parsedStateHistorySteps.map(
    (parsedStateHistoryStep: PokerReplayStepHistoryParsed) => {
      if (parsedStateHistoryStep.current_universal_poker_json.current_player > -1) {
        parsedStateHistoryStep.step = onlyPlayerSteps[stepToMapIndex];
        parsedStateHistoryStep.stepIndex = stepToMapIndex;
        stepToMapIndex++;
      }
      return parsedStateHistoryStep;
    }
  );

  const repeatedPokerFinalSteps: RepeatedPokerStep[] = [];

  let currentHandNumber: number | null = null;
  let startOfGameMovesToProcess: number | null = null;
  stateHistoryStepsWithReplaySteps.forEach((step: PokerReplayStepHistoryParsed) => {
    // This happens at the start of a new hand and the start of a the game
    if (currentHandNumber === null || currentHandNumber < step.hand_number) {
      startOfGameMovesToProcess = 4;
      // This is the small blind synthetic move
      if (startOfGameMovesToProcess === 4) {
        repeatedPokerFinalSteps.push({
          stepType: 'small-blind-post',
          communityCards: [],
          pot: 0,
          step: step.stepIndex ?? 0,
          winOdds: [],
          fiveCardBestHands: [],
          currentPlayer: -1,
          players: getRepeatedPokerStepPlayers(step, agents, 'small-blind-post')
        });
        startOfGameMovesToProcess--;
      }
    }
  });

  return stateHistoryStepsWithReplaySteps;
};
