import {
  PokerReplayInfoAgent,
  PokerReplayStepHistoryParsed,
  PokerReplayUniversalPokerJson,
} from './poker-replay-types';
import { RepeatedPokerStep, RepeatedPokerStepPlayer, RepeatedPokerStepType } from './poker-steps-types';

interface GeneratorResult {
  newSteps: RepeatedPokerStep[];
  // The number of replay steps consumed by the generator.
  rawStepsConsumed: number;
}

// The signature for any function that generates visual steps.
type StepGenerator = (
  // array of all remaining replay steps, starting with the current one.
  remainingRawSteps: PokerReplayStepHistoryParsed[],
  agents: PokerReplayInfoAgent[],
  startIndex: number
) => GeneratorResult;

// We aren't quite sure where to get this from so just hardcode it to 200 for now
const STARTING_STACK_SIZE = 200;
const FIRST_ACTOR_BY_STREET = [1, 0, 0, 0];
const NUM_PLAYERS = 2;
const ALL_IN_STRING = 'ALL-IN';

export function getBettingStringFromACPCState(acpcState: string): string {
  if (!acpcState) {
    return '';
  }

  const lines = acpcState.trim().split('\n');
  if (lines.length === 0) {
    return '';
  }

  const stateParts = lines[0].split(':');
  if (stateParts.length < 4) {
    return '';
  }

  return stateParts.slice(2, stateParts.length - 1).join(':');
}

export function getReadableActionsFromACPC(acpcState: string): string[] {
  const bettingString = getBettingStringFromACPCState(acpcState);
  if (!bettingString) {
    return [];
  }

  const moves: string[] = [];
  const streets = bettingString.split('/');

  const totalContributions: number[] = [2, 1];
  let streetBaselines: number[] = [0, 0];

  // Example:
  // Input: "r5c/cc/r11c/r122r200c"
  // Output: ["Raise 5", "Call 3", "Check", "Check", "Bet 6", "Call 6", "Bet 111", "Raise 189", "Call 78"]
  streets.forEach((streetAction, streetIndex) => {
    const trimmedAction = streetAction.trim();
    if (streetIndex > 0) {
      streetBaselines = [...totalContributions];
    }

    if (!trimmedAction) {
      return;
    }

    let actingPlayer = FIRST_ACTOR_BY_STREET[Math.min(streetIndex, FIRST_ACTOR_BY_STREET.length - 1)];

    let i = 0;
    while (i < trimmedAction.length) {
      const char = trimmedAction[i];
      const currentMax = Math.max(...totalContributions);
      const highestBaseline = Math.max(...streetBaselines);

      // the existance of a 'r' char in the ACPC string signifies the actor is 'Raising' by some amount
      if (char === 'r') {
        let amount = '';
        i++;
        while (i < trimmedAction.length && trimmedAction[i] >= '0' && trimmedAction[i] <= '9') {
          amount += trimmedAction[i];
          i++;
        }
        const targetTotal = parseInt(amount || '0', 10);
        const previousTotal = totalContributions[actingPlayer];
        const roundBaseline = streetBaselines[actingPlayer];
        const roundTotal = Math.max(targetTotal - roundBaseline, 0);
        const hasOutstandingBet = currentMax > highestBaseline;
        const verb = hasOutstandingBet ? 'Raise' : 'Bet';

        if (!Number.isFinite(targetTotal)) {
          throw new Error(`Invalid raise amount '${amount}' parsed from betting string '${bettingString}'.`);
        }
        if (targetTotal <= previousTotal) {
          throw new Error(
            `Invalid raise target ${targetTotal} for player ${actingPlayer} (previous total ${previousTotal}).`
          );
        }

        moves.push(`${verb} ${roundTotal}`);
        totalContributions[actingPlayer] = targetTotal;
      } else if (char === 'c') {
        const previousTotal = totalContributions[actingPlayer];
        if (previousTotal === currentMax) {
          moves.push('Check');
        } else {
          const callAmount = currentMax - previousTotal;
          moves.push(callAmount > 0 ? `Call ${callAmount}` : 'Call');
          totalContributions[actingPlayer] = currentMax;
        }
        i++;

        // the existance of an 'f' char in the ACPC string signifies the actor is 'Folding'
      } else if (char === 'f') {
        moves.push('Fold');
        i++;
      } else {
        throw new Error(`Unknown betting token '${char}' encountered in '${bettingString}'.`);
      }

      actingPlayer = (actingPlayer + 1) % NUM_PLAYERS;
    }
  });

  return moves;
}

export function getCommunityCardsFromACPC(acpcState: string): string {
  if (!acpcState) {
    return '';
  }

  const lines = acpcState.trim().split('\n');
  if (lines.length === 0) {
    return '';
  }

  const stateParts = lines[0].split(':');
  if (stateParts.length < 4) {
    return '';
  }

  // Last part contains the cards (e.g., "6cKd|AsJc/7hQh6d/2h")
  const cardString = stateParts[stateParts.length - 1];

  // Split by '|' to separate player hands from community cards
  const cardSegments = cardString.split('|');

  if (cardSegments.length < 2) {
    return '';
  }

  // After '|' we have: player1hand/flop/turn/river
  // Split by '/' and skip the first segment (player 1's hand)
  const segments = cardSegments[1].split('/');

  // Community cards start from index 1 (skip player 1's hand at index 0)
  const communitySegments = segments.slice(1);
  const fullBoardString = communitySegments.join('');

  return fullBoardString;
}

// Calculates the total chip contribution per player at the END of the previous street.
// This serves as the "baseline" to subtract from the total to get the current street's bet.
function getStreetBaselinesFromACPC(acpcState: string): number[] {
  const bettingString = getBettingStringFromACPCState(acpcState);
  if (!bettingString) return [0, 0];

  const streets = bettingString.split('/');

  // If we are in pre-flop (length 1), there are no previous streets.
  if (streets.length <= 1) {
    return [0, 0];
  }

  // We sum up contributions from all streets EXCEPT the last one (the current one)
  const previousStreets = streets.slice(0, streets.length - 1);

  const totalContributions: number[] = [2, 1]; // Standard blinds start

  previousStreets.forEach((streetAction, streetIndex) => {
    const trimmedAction = streetAction.trim();
    if (!trimmedAction) return;

    let actingPlayer = FIRST_ACTOR_BY_STREET[Math.min(streetIndex, FIRST_ACTOR_BY_STREET.length - 1)];

    let i = 0;
    while (i < trimmedAction.length) {
      const char = trimmedAction[i];
      const currentMax = Math.max(...totalContributions);

      if (char === 'r') {
        let amount = '';
        i++;
        while (i < trimmedAction.length && trimmedAction[i] >= '0' && trimmedAction[i] <= '9') {
          amount += trimmedAction[i];
          i++;
        }
        const targetTotal = parseInt(amount || '0', 10);
        if (Number.isFinite(targetTotal)) {
          totalContributions[actingPlayer] = targetTotal;
        }
      } else if (char === 'c') {
        totalContributions[actingPlayer] = currentMax;
        i++;
      } else {
        i++;
      }
      actingPlayer = (actingPlayer + 1) % NUM_PLAYERS;
    }
  });

  return totalContributions;
}

const getReadableActionDelta = (
  beforeJson: PokerReplayUniversalPokerJson,
  afterJson: PokerReplayUniversalPokerJson
): string | null => {
  if (!beforeJson || !afterJson) {
    return null;
  }

  const beforeMoves = getReadableActionsFromACPC(beforeJson.acpc_state);
  const afterMoves = getReadableActionsFromACPC(afterJson.acpc_state);

  if (afterMoves.length <= beforeMoves.length) {
    return null;
  }

  const newMoves = afterMoves.slice(beforeMoves.length);

  if (newMoves.length === 0) {
    return null;
  }

  return newMoves[newMoves.length - 1];
};

// Utility to create the player action step from both a raw step and the end of a community card sequence.
const createPlayerActionStep = (
  stateBeforeAction: PokerReplayStepHistoryParsed,
  stateAfterAction: PokerReplayStepHistoryParsed,
  agents: PokerReplayInfoAgent[],
  stepIndex: number
): RepeatedPokerStep => {
  const { current_universal_poker_json: beforeJson } = stateBeforeAction;
  const { current_universal_poker_json: afterJson } = stateAfterAction;

  const actionObject = stateBeforeAction.step;

  const actingPlayerId = beforeJson.current_player;
  const actionString = actionObject?.action?.actionString ?? '';

  const parsedActionStringTuple = actionString.match(/move=(\S+)/);
  let parsedActionString = '';
  if (parsedActionStringTuple && parsedActionStringTuple[1]) {
    parsedActionString = parsedActionStringTuple[1];
  }

  const readableActionDisplay = getReadableActionDelta(beforeJson, afterJson);
  const activeActionDisplay = readableActionDisplay ?? parsedActionString ?? '';

  const baselines = getStreetBaselinesFromACPC(beforeJson.acpc_state);

  const streetContributions = [0, 1].map((id) => afterJson.player_contributions[id] - baselines[id]);
  const maxStreetContribution = Math.max(...streetContributions);
  const players = [0, 1].map((id) => {
    const callAmount = maxStreetContribution - streetContributions[id];
    const waitingActionDisplay = callAmount === 0 ? '' : `${callAmount} to call`;
    const chipStack = STARTING_STACK_SIZE - afterJson.player_contributions[id];
    const isAllIn = chipStack === 0;
    
    let actionDisplayText: string;
    if (isAllIn) {
      actionDisplayText = ALL_IN_STRING;
    } else if (id === actingPlayerId) {
      actionDisplayText = activeActionDisplay;
    } else {
      actionDisplayText = waitingActionDisplay;
    }
    
    return {
      id,
      name: agents[id].Name,
      thumbnail: agents[id].ThumbnailUrl,
      cards: afterJson.player_hands[id],
      chipStack,
      currentBet: afterJson.player_contributions[id],
      currentBetForStreet: Math.max(0, afterJson.player_contributions[id] - baselines[id]),
      reward: null,
      actionDisplayText,
      thoughts: id === actingPlayerId ? (actionObject?.action?.thoughts ?? '') : '',
      isDealer: stateAfterAction.dealer === id,
      isTurn: id === actingPlayerId,
      isWinner: false,
    };
  });

  const communityCards = getCommunityCardsFromACPC(beforeJson.acpc_state);

  return {
    stepType: 'player-action',
    communityCards,
    pot: afterJson.pot_size,
    step: stepIndex,
    winOdds: afterJson.odds,
    bestFiveCardHands: afterJson.best_five_card_hands,
    bestHandRankTypes: afterJson.best_hand_rank_types,
    currentPlayer: actingPlayerId,
    currentHandIndex: stateAfterAction.hand_number,
    players,
  };
};

// Reusable helper to create the 'final' step of a hand.
const createFinalHandStep = (
  finalJson: PokerReplayUniversalPokerJson,
  handRewards: number[],
  finishedHandNumber: number,
  dealerId: number,
  agents: PokerReplayInfoAgent[],
  stepIndex: number
): RepeatedPokerStep => {
  const players: RepeatedPokerStepPlayer[] = [0, 1].map((id) => {
    const reward = handRewards[id];
    const isWinner = reward >= 0;
    const actionDisplayText = reward > 0 ? `WINS ${reward}` : reward === 0 ? 'SPLIT POT' : '';
    return {
      id,
      name: agents[id].Name,
      thumbnail: agents[id].ThumbnailUrl,
      cards: finalJson.player_hands[id],
      chipStack: STARTING_STACK_SIZE - finalJson.player_contributions[id],
      currentBet: finalJson.player_contributions[id],
      currentBetForStreet: 0,
      reward,
      actionDisplayText: actionDisplayText,
      thoughts: '',
      isDealer: dealerId === id,
      isTurn: false,
      isWinner,
    };
  });

  const communityCards = getCommunityCardsFromACPC(finalJson.acpc_state);

  return {
    stepType: 'final',
    communityCards,
    pot: finalJson.pot_size,
    step: stepIndex,
    winOdds: finalJson.odds,
    bestFiveCardHands: finalJson.best_five_card_hands,
    bestHandRankTypes: finalJson.best_hand_rank_types,
    currentPlayer: -1,
    currentHandIndex: finishedHandNumber,
    players,
  };
};

const generateFinalReplaySequence: StepGenerator = (remainingRawSteps, agents, startIndex) => {
  const lastStepOfReplay = remainingRawSteps[0];

  // At the very end of the tape, the current JSON is the final state.
  const finalJson = lastStepOfReplay.current_universal_poker_json;
  const handIndex = lastStepOfReplay.hand_number;
  const finalRewards = lastStepOfReplay.hand_returns?.[handIndex] ?? [0, 0];

  const finalHandStep = createFinalHandStep(
    finalJson,
    finalRewards,
    lastStepOfReplay.hand_number,
    lastStepOfReplay.dealer,
    agents,
    startIndex
  );

  // Generate Game Over step based on the final step
  // This is mostly for UI so just using empty state mostly, but we do mark the over all winner for the action string
  const allReturns = lastStepOfReplay.hand_returns ?? [];
  const totalReturns = [0, 0];

  allReturns.forEach((handReturn) => {
    totalReturns[0] += handReturn[0];
    totalReturns[1] += handReturn[1];
  });

  // Determine overall winner based on net returns.
  // TODO - handle ties
  let overallWinnerId = -1;
  if (totalReturns[0] > totalReturns[1]) overallWinnerId = 0;
  else if (totalReturns[1] > totalReturns[0]) overallWinnerId = 1;

  // Generate Game Over step with overall stats
  const gameOverPlayers: RepeatedPokerStepPlayer[] = (finalHandStep.players as RepeatedPokerStepPlayer[]).map(
    (player) => {
      const totalWinnings = totalReturns[player.id];
      const isOverallWinner = player.id === overallWinnerId;

      return {
        ...player,
        chipStack: totalWinnings,
        currentBet: 0,
        cards: '',
        reward: totalWinnings,
        isWinner: isOverallWinner,
        actionDisplayText: '',
        isTurn: false,
        isDealer: false,
      };
    }
  );

  const gameOverStep: RepeatedPokerStep = {
    stepType: 'game-over',
    communityCards: '',
    pot: 0,
    step: startIndex + 1,
    winOdds: [],
    bestFiveCardHands: ['', ''],
    bestHandRankTypes: ['', ''],
    currentPlayer: -1,
    currentHandIndex: lastStepOfReplay.hand_number,
    players: gameOverPlayers,
  };

  return {
    newSteps: [finalHandStep, gameOverStep],
    rawStepsConsumed: 1,
  };
};

const generatePlayerActionStep: StepGenerator = (remainingRawSteps, agents, startIndex) => {
  const stateBeforeAction = remainingRawSteps[0];
  const stateAfterAction = remainingRawSteps[1]; // Look ahead one step

  if (!stateAfterAction) {
    // This should, in theory, be caught by the router, but good to have.
    throw new Error(`Data inconsistency: generatePlayerActionStep was called but no "after" state exists.`);
  }

  const newStep = createPlayerActionStep(stateBeforeAction, stateAfterAction, agents, startIndex);

  return {
    newSteps: [newStep],
    rawStepsConsumed: 1,
  };
};

const generateHandEndStepSequence: StepGenerator = (remainingRawSteps, agents, startIndex) => {
  const stateBeforeEnd = remainingRawSteps[0]; // Last step of current hand
  const nextHandStartStep = remainingRawSteps[1]; // First step of *next* hand

  if (!nextHandStartStep || !nextHandStartStep.prev_universal_poker_json) {
    throw new Error(
      `Data inconsistency: generateHandEndSequence was called for hand ${stateBeforeEnd.hand_number} ` +
        `but the next step (for hand ${nextHandStartStep?.hand_number}) is missing 'prev_universal_poker_json'.`
    );
  }

  const visualSteps: RepeatedPokerStep[] = [];
  let visualStepIndex = startIndex;

  // Generate the final action if it exists (e.g., the 'Call' that ends the hand)
  // We use the final confirmed JSON for this visualization so the pot looks right immediately.
  const finalJson = nextHandStartStep.prev_universal_poker_json;

  if (stateBeforeEnd.step) {
    const stateAfterAction = {
      ...stateBeforeEnd,
      current_universal_poker_json: finalJson,
      dealer: stateBeforeEnd.dealer,
    } as PokerReplayStepHistoryParsed;

    visualSteps.push(createPlayerActionStep(stateBeforeEnd, stateAfterAction, agents, visualStepIndex++));
  }

  const handIndex = stateBeforeEnd.hand_number;
  const finalRewards = nextHandStartStep.hand_returns?.[handIndex] ?? [0, 0];

  visualSteps.push(
    createFinalHandStep(
      finalJson,
      finalRewards,
      stateBeforeEnd.hand_number,
      stateBeforeEnd.dealer,
      agents,
      visualStepIndex++
    )
  );

  return {
    newSteps: visualSteps,
    rawStepsConsumed: 1, // Still only consumes the one "end" step
  };
};

const generatePreFlopStepSequence: StepGenerator = (remainingRawSteps, agents, startIndex) => {
  if (!remainingRawSteps || remainingRawSteps.length === 0) {
    throw new Error('Cannot generate pre-flop sequence: No raw steps were provided to the generator.');
  }

  // Find the index of the first action step.
  let firstActionStepIndex = -1;
  for (let i = 0; i < remainingRawSteps.length; i++) {
    const step = remainingRawSteps[i];
    // We stop when we find the first step that is a player action.
    if (step.step) {
      firstActionStepIndex = i;
      break;
    }
    // Safety break if we somehow hit a new hand
    if (i > 0 && step.hand_number > remainingRawSteps[0].hand_number) {
      break;
    }
  }

  if (firstActionStepIndex === -1) {
    throw new Error(`Invalid pre-flop sequence: Could not find a player action step after the pre-flop deal.`);
  }

  const rawStepsConsumed = firstActionStepIndex;

  if (rawStepsConsumed !== 4) {
    throw new Error(
      `Invalid pre-flop sequence: Expected to consume exactly 4 raw steps for the pre-flop deal, but found ${rawStepsConsumed}.`
    );
  }

  // Get the two steps we need for data:
  // The last "deal" step (Raw Step 3)
  const lastDealStep = remainingRawSteps[rawStepsConsumed - 1];
  // The first "action" step (Raw Step 4), which we "peek" at but do not consume.
  const firstActionStep = remainingRawSteps[rawStepsConsumed];

  // Blinds/dealer info comes from the deal steps.
  const { dealer, small_blind, big_blind } = lastDealStep;

  // The complete hand/odds data comes from the firstActionStep
  const { player_hands, best_five_card_hands, best_hand_rank_types, odds } =
    firstActionStep.current_universal_poker_json;

  // in heads up the dealer is the small blind
  const sbPlayerId = dealer;
  const bbPlayerId = 1 - dealer;

  // Visual Step 1: Small Blind Post
  const sbPostPlayers: RepeatedPokerStepPlayer[] = [0, 1].map((id) => ({
    id,
    name: agents[id].Name,
    thumbnail: agents[id].ThumbnailUrl,
    cards: '',
    chipStack: id === sbPlayerId ? STARTING_STACK_SIZE - small_blind : STARTING_STACK_SIZE,
    currentBet: id === sbPlayerId ? small_blind : 0,
    currentBetForStreet: id === sbPlayerId ? small_blind : 0,
    reward: null,
    actionDisplayText: id === sbPlayerId ? 'SB' : '',
    thoughts: '',
    isDealer: id === dealer,
    isTurn: id === sbPlayerId,
    isWinner: false,
  }));

  const sbPostStep: RepeatedPokerStep = {
    stepType: 'small-blind-post',
    communityCards: '',
    pot: small_blind,
    step: startIndex,
    winOdds: [],
    bestFiveCardHands: ['', ''],
    bestHandRankTypes: ['', ''],
    currentPlayer: -1,
    currentHandIndex: lastDealStep.hand_number,
    players: sbPostPlayers,
  };

  // Visual Step 2: Big Blind Post
  const bbPostPlayers: RepeatedPokerStepPlayer[] = sbPostPlayers.map((player) =>
    player.id === bbPlayerId
      ? {
          ...player,
          chipStack: STARTING_STACK_SIZE - big_blind,
          currentBet: big_blind,
          currentBetForStreet: big_blind,
          actionDisplayText: 'BB',
          isTurn: true,
        }
      : {
          ...player,
          actionDisplayText: '',
          isTurn: false
        }
  );
  const bbPostStep: RepeatedPokerStep = {
    ...sbPostStep,
    stepType: 'big-blind-post',
    pot: small_blind + big_blind,
    currentPlayer: -1,
    step: startIndex + 1,
    players: bbPostPlayers,
  };

  // Visual Step 3: Deal Player Cards
  const dealCardsPlayers: RepeatedPokerStepPlayer[] = bbPostPlayers.map((player) => ({
    ...player,
    isTurn: false,
    actionDisplayText: '',
    cards: player_hands[player.id], // Using hands from firstActionStep
  }));
  const dealCardsStep: RepeatedPokerStep = {
    ...bbPostStep,
    stepType: 'deal-player-hands',
    step: startIndex + 2,
    currentPlayer: -1,
    players: dealCardsPlayers,
    bestFiveCardHands: best_five_card_hands, // Using data from firstActionStep
    bestHandRankTypes: best_hand_rank_types,
    winOdds: odds, // Using data from firstActionStep
  };

  return {
    newSteps: [sbPostStep, bbPostStep, dealCardsStep],
    rawStepsConsumed: rawStepsConsumed,
  };
};

const generateCommunityCardStepSequence: StepGenerator = (remainingRawSteps, agents, startIndex) => {
  const preDealStep = remainingRawSteps[0]; // State at the end of the previous betting round.
  const preDealBoardLength = preDealStep.current_universal_poker_json.board_cards.length;

  // Look ahead to find the "before action" step OR the "end of hand"
  let rawStepsConsumed = 1; // Start by consuming the preDealStep
  let actionStepIndex = -1;
  let endOfHand = false;

  for (let i = 1; i < remainingRawSteps.length; i++) {
    const curr = remainingRawSteps[i];

    // : Check for new hand *before* consuming ---
    // Most often this is a hand transition, but it can also be the very end of the episode.
    if (curr.hand_number > preDealStep.hand_number) {
      // We found a new hand before finding an action. This is the ALL-IN case.
      endOfHand = true;
      break; // Stop, but *don't* consume this step.
    }

    rawStepsConsumed++; // Consume this step

    // The original safety break is now redundant, but we'll leave it
    if (i > 0 && curr.hand_number > remainingRawSteps[0].hand_number) {
      break;
    }

    if (curr.step) {
      // This is the "before" action step.
      actionStepIndex = i;
      break;
    }
  }

  // --- CASE 1: Normal path, we found an action step ---
  if (actionStepIndex !== -1) {
    const stateBeforeAction = remainingRawSteps[actionStepIndex];
    const stateAfterAction = remainingRawSteps[actionStepIndex + 1];

    if (!stateAfterAction) {
      const stepIdentifier = (stateBeforeAction as any).stepIndex ?? 'unknown';
      throw new Error(
        `Data inconsistency: Found a post-flop action at raw step index ${stepIdentifier} ` +
          `but no subsequent state step follows.`
      );
    }

    // Determine the step type
    const communityCards = getCommunityCardsFromACPC(stateBeforeAction.current_universal_poker_json.acpc_state);

    let stepType: RepeatedPokerStepType;
    if (communityCards.length === 6) stepType = 'deal-flop';
    else if (communityCards.length === 8) stepType = 'deal-turn';
    else if (communityCards.length === 10) stepType = 'deal-river';
    else {
      throw new Error(`Unexpected board cards length: ${communityCards.length}.`);
    }

    // --- Visual Step 1: The "Deal" Step ---
    const dealStepPlayers: RepeatedPokerStepPlayer[] = [0, 1].map((id) => {
      const chipStack = STARTING_STACK_SIZE - stateBeforeAction.current_universal_poker_json.player_contributions[id];
      
      return {
        id,
        name: agents[id].Name,
        thumbnail: agents[id].ThumbnailUrl,
        cards: stateBeforeAction.current_universal_poker_json.player_hands[id],
        chipStack,
        currentBet: stateBeforeAction.current_universal_poker_json.player_contributions[id],
        currentBetForStreet: 0,
        reward: null,
        actionDisplayText: chipStack === 0 ? ALL_IN_STRING : '',
        thoughts: '',
        isDealer: preDealStep.dealer === id,
        isTurn: false,
        isWinner: false,
      }
    });

    const dealStep: RepeatedPokerStep = {
      stepType,
      communityCards,
      pot: stateBeforeAction.current_universal_poker_json.pot_size,
      step: startIndex,
      winOdds: stateBeforeAction.current_universal_poker_json.odds,
      bestFiveCardHands: stateBeforeAction.current_universal_poker_json.best_five_card_hands,
      bestHandRankTypes: stateBeforeAction.current_universal_poker_json.best_hand_rank_types,
      currentPlayer: -1,
      currentHandIndex: stateBeforeAction.hand_number,
      players: dealStepPlayers,
    };

    // --- Visual Step 2: The First "Player Action" Step on this Street ---
    const actionStep = createPlayerActionStep(
      stateBeforeAction,
      stateAfterAction,
      agents,
      startIndex + 1 // This is the second visual step
    );

    return {
      newSteps: [dealStep, actionStep],
      rawStepsConsumed: rawStepsConsumed,
    };
  }

  // --- CASE 2: All-In Runout, we hit end of hand OR the end of the episode---
  if (endOfHand || actionStepIndex === -1) {
    const newSteps: RepeatedPokerStep[] = [];
    let visualStepIndex = startIndex;

    // 1. Gather all relevant JSON states for this runout.
    const runoutJsonStates: PokerReplayUniversalPokerJson[] = [];
    // Start at 1 because index 0 (preDealStep) is already gathered by the calling context usually,
    // but for runouts we need to process from the *next* card onwards.
    // Actually, wait, we need to include the current step's state too if it wasn't processed in CASE 1.
    // Let's stick to gathering from 0 to be safe and let the board-length check filter duplicates.
    for (let i = 0; i < rawStepsConsumed; i++) {
      runoutJsonStates.push(remainingRawSteps[i].current_universal_poker_json);
    }

    // Peek ahead: The absolute final state of this hand (e.g. the River in an all-in)
    // often only exists in the 'prev_' JSON of the *next* hand's first step.
    const nextHandStep = endOfHand ? remainingRawSteps[rawStepsConsumed] : null;
    if (nextHandStep?.prev_universal_poker_json) {
      runoutJsonStates.push(nextHandStep.prev_universal_poker_json);
    }

    // 2. Linear scan to generate DEAL steps for new streets
    // We start with 0 so the very first step (e.g. Flop) is detected if it's new.
    let lastProcessedBoardLength = preDealBoardLength === 0 ? -1 : preDealBoardLength;
    // (Optimization: if preDeal was 0, setting to -1 ensures standard flop (len 6) is caught)

    for (let i = 0; i < runoutJsonStates.length; i++) {
      const currentState = runoutJsonStates[i];
      const currentBoardLen = currentState.board_cards.length;

      // Detect new streets (Flop=6, Turn=8, River=10)
      if (currentBoardLen > lastProcessedBoardLength && [6, 8, 10].includes(currentBoardLen)) {
        // Find the BEST state for this street (latest one before next deal)
        let bestStateIndex = i;
        while (
          bestStateIndex + 1 < runoutJsonStates.length &&
          runoutJsonStates[bestStateIndex + 1].board_cards.length === currentBoardLen
        ) {
          bestStateIndex++;
        }

        i = bestStateIndex; // Fast-forward loop
        const stateForDeal = runoutJsonStates[bestStateIndex];

        let stepType: RepeatedPokerStepType;
        if (currentBoardLen === 6) stepType = 'deal-flop';
        else if (currentBoardLen === 8) stepType = 'deal-turn';
        else stepType = 'deal-river';

        // When dealing cards, the acpc uses '2c' as a placeholder for cards about to be dealt. This removes those.
        const communityCards = getCommunityCardsFromACPC(stateForDeal.acpc_state).substring(0, currentBoardLen);

        newSteps.push({
          stepType,
          communityCards,
          pot: stateForDeal.pot_size,
          step: visualStepIndex++,
          winOdds: stateForDeal.odds,
          bestFiveCardHands: stateForDeal.best_five_card_hands,
          bestHandRankTypes: stateForDeal.best_hand_rank_types,
          currentPlayer: -1,
          currentHandIndex: preDealStep.hand_number,
          players: [0, 1].map((id) => {
            const chipStack = STARTING_STACK_SIZE - stateForDeal.player_contributions[id];

            return {
              id,
              name: agents[id].Name,
              thumbnail: agents[id].ThumbnailUrl,
              cards: stateForDeal.player_hands[id],
              chipStack,
              currentBet: stateForDeal.player_contributions[id],
              currentBetForStreet: 0,
              reward: null,
              actionDisplayText: chipStack === 0 ? ALL_IN_STRING : '',
              thoughts: '',
              isDealer: preDealStep.dealer === id,
              isTurn: false,
              isWinner: false,
          }}),
        });

        lastProcessedBoardLength = currentBoardLen;
      }
    }

    // 3. Generate the FINAL summary step if we reached the end of the hand.
    // This ensures we don't miss the winner banners.
    if (endOfHand && nextHandStep && nextHandStep.prev_universal_poker_json) {
      const finalJson = nextHandStep.prev_universal_poker_json;
      const handIndex = preDealStep.hand_number;
      const finalRewards = nextHandStep.hand_returns?.[handIndex] ?? [0, 0];

      newSteps.push(
        createFinalHandStep(
          finalJson,
          finalRewards,
          preDealStep.hand_number,
          preDealStep.dealer,
          agents,
          visualStepIndex++
        )
      );
    }

    // 4. Handle end of REPLAY (last hand of the game, no next hand exists).
    // This occurs when we've consumed all remaining raw steps without finding a next hand.
    const isEndOfReplay = !endOfHand && rawStepsConsumed >= remainingRawSteps.length;
    if (isEndOfReplay) {
      // Use the last raw step for final state data
      const lastRawStep = remainingRawSteps[remainingRawSteps.length - 1];
      const finalJson = lastRawStep.current_universal_poker_json;
      const handIndex = preDealStep.hand_number;
      const finalRewards = lastRawStep.hand_returns?.[handIndex] ?? [0, 0];

      // Generate final hand step
      const finalHandStep = createFinalHandStep(
        finalJson,
        finalRewards,
        preDealStep.hand_number,
        preDealStep.dealer,
        agents,
        visualStepIndex++
      );
      newSteps.push(finalHandStep);

      // Generate game-over step
      const allReturns = lastRawStep.hand_returns ?? [];
      const totalReturns = [0, 0];
      allReturns.forEach((handReturn) => {
        totalReturns[0] += handReturn[0];
        totalReturns[1] += handReturn[1];
      });

      let overallWinnerId = -1;
      if (totalReturns[0] > totalReturns[1]) overallWinnerId = 0;
      else if (totalReturns[1] > totalReturns[0]) overallWinnerId = 1;

      const gameOverPlayers: RepeatedPokerStepPlayer[] = (finalHandStep.players as RepeatedPokerStepPlayer[]).map(
        (player) => {
          const totalWinnings = totalReturns[player.id];
          const isOverallWinner = player.id === overallWinnerId;
          return {
            ...player,
            chipStack: totalWinnings,
            currentBet: 0,
            cards: '',
            reward: totalWinnings,
            isWinner: isOverallWinner,
            actionDisplayText: '',
            isTurn: false,
            isDealer: false,
          };
        }
      );

      const gameOverStep: RepeatedPokerStep = {
        stepType: 'game-over',
        communityCards: '',
        pot: 0,
        step: visualStepIndex++,
        winOdds: [],
        bestFiveCardHands: ['', ''],
        bestHandRankTypes: ['', ''],
        currentPlayer: -1,
        currentHandIndex: preDealStep.hand_number,
        players: gameOverPlayers,
      };
      newSteps.push(gameOverStep);
    }

    return {
      newSteps: newSteps,
      rawStepsConsumed: rawStepsConsumed,
    };
  }

  // --- CASE 3: Error ---
  // This means the loop finished without finding an action OR a new hand OR the end of the episode.
  throw new Error(
    `Data inconsistency: Could not find a player action step OR end of hand after the community card deal on hand ${preDealStep.hand_number}.`
  );
};

// Returning null from this function will throw an error in the outer loop
function determineGenerator(remainingSteps: PokerReplayStepHistoryParsed[]): StepGenerator | null {
  if (!remainingSteps || remainingSteps.length === 0) {
    return null;
  }

  const currentReplayStep = remainingSteps[0];
  const nextReplayStep = remainingSteps.length > 1 ? remainingSteps[1] : null;

  // Rule 1: Is this the very last step of the entire replay?
  if (nextReplayStep === null) {
    return generateFinalReplaySequence;
  }

  const isHandBoundary = nextReplayStep.hand_number > currentReplayStep.hand_number;

  // Rule 2: Community Cards take precedence over simple hand boundaries.
  // This ensures all-in runouts (Turn/River) are generated before we close the hand.
  // We also need a special 2nd condition here to handle the 'all-in' on the Turn case,
  // we only have information about the River from the prev_universal_poker_json on the start of the next hand.
  if (
    nextReplayStep.current_universal_poker_json.board_cards.length >
      currentReplayStep.current_universal_poker_json.board_cards.length ||
    (isHandBoundary &&
      nextReplayStep.prev_universal_poker_json.board_cards.length >
        currentReplayStep.current_universal_poker_json.board_cards.length)
  ) {
    return generateCommunityCardStepSequence;
  }

  // Rule 3: Is this a boundary between hands
  if (isHandBoundary) {
    // This is the *last* step of the current hand.
    // It could be an action (Fold) or a result (Runout).
    // `generateHandEndSequence` handles both cases.
    return generateHandEndStepSequence;
  }

  // Rule 4: Is this the start of a new hand?
  const hands = currentReplayStep.current_universal_poker_json.player_hands;
  if (hands[0] === '' && hands[1] === '') {
    return generatePreFlopStepSequence;
  }

  // Rule 5: Is this a standard player action?
  if (currentReplayStep.step) {
    return generatePlayerActionStep;
  }

  // If no rule matches, it's an unhandled state.
  return null;
}

export const createVisualStepsFromRepeatedPokerReplay = (
  parsedSteps: PokerReplayStepHistoryParsed[],
  agents: PokerReplayInfoAgent[]
): RepeatedPokerStep[] => {
  const allVisualSteps: RepeatedPokerStep[] = [];
  let rawStepIndex = 0;

  while (rawStepIndex < parsedSteps.length) {
    const remainingRawSteps = parsedSteps.slice(rawStepIndex);
    const generator = determineGenerator(remainingRawSteps);

    if (!generator) {
      // It means we hit the last un-processable step or an error.
      // If it's an error, we throw. If it's just the end, we break.
      const problematicStep = remainingRawSteps[0];
      const nextReplayStep = remainingRawSteps.length > 1 ? remainingRawSteps[1] : null;

      if (nextReplayStep === null) {
        break;
      }

      throw new Error(
        `Could not determine a generator for the raw step at index ${rawStepIndex}. ` +
          `This indicates an unexpected data format or an unhandled game state. ` +
          `Problematic Step Data: ${JSON.stringify(problematicStep, null, 2)}`
      );
    }

    const result = generator(remainingRawSteps, agents, allVisualSteps.length);
    allVisualSteps.push(...result.newSteps);
    // All generators consume at least 1 step.
    rawStepIndex += result.rawStepsConsumed;
  }
  return allVisualSteps;
};
