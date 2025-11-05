import { PokerReplayInfoAgent, PokerReplayStepHistoryParsed, PokerReplayUniversalPokerJson } from './poker-replay-types';
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

  // TODO(jhtschultz): Better/more clear actionString support
  const parsedActionStringTuple = actionString.match(/move=(\S+)/);
  let parsedActionString = '';
  if (parsedActionStringTuple && parsedActionStringTuple[1]) {
    parsedActionString = parsedActionStringTuple[1];
  }

  const readableActionDisplay = getReadableActionDelta(beforeJson, afterJson);
  const finalActionDisplay = readableActionDisplay ?? parsedActionString ?? '';
  const players = [0, 1].map(id => ({
    id, name: agents[id].Name, thumbnail: agents[id].ThumbnailUrl,
    cards: afterJson.player_hands[id],
    chipStack: STARTING_STACK_SIZE - afterJson.player_contributions[id],
    currentBet: afterJson.player_contributions[id], 
    reward: null,
    actionDisplayText: id === actingPlayerId ? finalActionDisplay : '',
    thoughts: id === actingPlayerId ? (actionObject?.action?.thoughts ?? '') : '',
    isDealer: stateAfterAction.dealer === id,
    isTurn: id === actingPlayerId,
    isWinner: false,
  }));

  return {
    stepType: 'player-action',
    communityCards: afterJson.board_cards,
    pot: afterJson.pot_size,
    step: stepIndex,
    winOdds: afterJson.odds,
    fiveCardBestHands: afterJson.best_five_card_hands,
    currentPlayer: actingPlayerId,
    players,
  };
};

const generatePlayerActionStep: StepGenerator = (remainingRawSteps, agents, startIndex) => {
  const stateBeforeAction = remainingRawSteps[0];
  const stateAfterAction = remainingRawSteps[1]; // Look ahead one step

  if (!stateAfterAction) {
    // This should, in theory, be caught by the router, but good to have.
    throw new Error(
      `Data inconsistency: generatePlayerActionStep was called but no "after" state exists.`
    );
  }

  const newStep = createPlayerActionStep(
    stateBeforeAction,
    stateAfterAction,
    agents,
    startIndex
  );

  return {
    newSteps: [newStep],
    rawStepsConsumed: 1
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
  
  // This is the data we need for the *final* summary, regardless of what came before.
  const finalJson = nextHandStartStep.prev_universal_poker_json;

  // --- CASE 1: Hand ends on an Action (e.g., Fold) ---
  if (stateBeforeEnd.step) {
    // We must create a "fake" stateAfterAction to show the result of the action (e.g., pot size, cards)
    // This "after" state's data comes from the `prev_universal_poker_json` of the *next* hand.
    const stateAfterAction = {
      ...stateBeforeEnd, // Copy base properties
      current_universal_poker_json: finalJson, // Use final state JSON
      dealer: stateBeforeEnd.dealer // IMPORTANT: Preserve dealer from *before* action
    } as PokerReplayStepHistoryParsed;

    const actionStep = createPlayerActionStep(
      stateBeforeEnd,   // state *before* action
      stateAfterAction, // state *after* action
      agents,
      visualStepIndex
    );
    visualSteps.push(actionStep);
    visualStepIndex++;
  }
  
  // --- CASE 2: The "Final" Hand Summary (ALWAYS runs) ---
  // This runs for both "Fold" (Case 1) and "Runout" (no .step object) scenarios.

  const finalStepPlayers: RepeatedPokerStepPlayer[] = [0, 1].map((id) => {
    // hand_returns is 0-indexed, hand_number is 1-indexed
    const handReturnIndex = nextHandStartStep.hand_number - 1;
    const reward = nextHandStartStep.hand_returns[handReturnIndex][id];
    const isWinner = reward > 0;
    const actionDisplayText = isWinner ? `WINS ${reward}` : '';

    return {
      id,
      name: agents[id].Name,
      thumbnail: agents[id].ThumbnailUrl,
      cards: finalJson.player_hands[id],
      chipStack: STARTING_STACK_SIZE - finalJson.player_contributions[id],
      currentBet: finalJson.player_contributions[id],
      reward,
      actionDisplayText,
      thoughts: '',
      isDealer: stateBeforeEnd.dealer === id, // Get dealer from the last step of *this* hand
      isTurn: false,
      isWinner,
    };
  });

  const finalStep: RepeatedPokerStep = {
    stepType: 'final',
    communityCards: finalJson.board_cards,
    pot: finalJson.pot_size,
    step: visualStepIndex, // This is either startIndex or startIndex + 1
    winOdds: finalJson.odds,
    fiveCardBestHands: finalJson.best_five_card_hands,
    currentPlayer: -1,
    players: finalStepPlayers,
  };
  visualSteps.push(finalStep);
  
  return {
    newSteps: visualSteps,
    rawStepsConsumed: 1 // Still only consumes the one "end" step
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
  const {
    dealer,
    small_blind,
    big_blind,
  } = lastDealStep; 

  // The complete hand/odds data comes from the firstActionStep
  const { player_hands, best_five_card_hands, odds } =
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
    reward: null,
    actionDisplayText: id === sbPlayerId ? 'SB' : '',
    thoughts: '',
    isDealer: id === dealer,
    isTurn: false,
    isWinner: false
  }));
  const sbPostStep: RepeatedPokerStep = {
    stepType: 'small-blind-post',
    communityCards: "",
    pot: small_blind,
    step: startIndex,
    winOdds: [],
    fiveCardBestHands: ['', ''],
    currentPlayer: -1,
    players: sbPostPlayers
  };

  // Visual Step 2: Big Blind Post
  const bbPostPlayers: RepeatedPokerStepPlayer[] = sbPostPlayers.map((player) =>
    player.id === bbPlayerId
      ? { 
          ...player, 
          chipStack: STARTING_STACK_SIZE - big_blind,
          currentBet: big_blind, 
          actionDisplayText: 'BB' 
        }
      : { ...player, 
          actionDisplayText: '' }
  );
  const bbPostStep: RepeatedPokerStep = {
    ...sbPostStep,
    stepType: 'big-blind-post',
    pot: small_blind + big_blind,
    currentPlayer: -1,
    step: startIndex + 1,
    players: bbPostPlayers
  };

  // Visual Step 3: Deal Player Cards
  const dealCardsPlayers: RepeatedPokerStepPlayer[] = bbPostPlayers.map((player) => ({
    ...player,
    actionDisplayText: '',
    cards: player_hands[player.id] // Using hands from firstActionStep
  }));
  const dealCardsStep: RepeatedPokerStep = {
    ...bbPostStep,
    stepType: 'deal-player-hands',
    step: startIndex + 2,
    currentPlayer: -1,
    players: dealCardsPlayers,
    fiveCardBestHands: best_five_card_hands, // Using data from firstActionStep
    winOdds: odds // Using data from firstActionStep
  };

  return {
    newSteps: [sbPostStep, bbPostStep, dealCardsStep],
    rawStepsConsumed: rawStepsConsumed
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
    const boardCards = stateBeforeAction.current_universal_poker_json.board_cards;
    let stepType: RepeatedPokerStepType;
    if (boardCards.length === 6) stepType = 'deal-flop';
    else if (boardCards.length === 8) stepType = 'deal-turn';
    else if (boardCards.length === 10) stepType = 'deal-river';
    else {
      throw new Error(`Unexpected board cards length: ${boardCards.length}.`);
    }

    // --- Visual Step 1: The "Deal" Step ---
    const dealStepPlayers: RepeatedPokerStepPlayer[] = [0, 1].map((id) => ({
      id, name: agents[id].Name, thumbnail: agents[id].ThumbnailUrl,
      cards: stateBeforeAction.current_universal_poker_json.player_hands[id],
      chipStack: STARTING_STACK_SIZE - stateBeforeAction.current_universal_poker_json.player_contributions[id],
      currentBet: stateBeforeAction.current_universal_poker_json.player_contributions[id],
      reward: null,
      actionDisplayText: '',
      thoughts: '',
      isDealer: preDealStep.dealer === id,
      isTurn: false,
      isWinner: false,
    }));

    const dealStep: RepeatedPokerStep = {
      stepType,
      communityCards: boardCards,
      pot: stateBeforeAction.current_universal_poker_json.pot_size,
      step: startIndex,
      winOdds: stateBeforeAction.current_universal_poker_json.odds,
      fiveCardBestHands: stateBeforeAction.current_universal_poker_json.best_five_card_hands,
      currentPlayer: -1,
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

  // --- CASE 2: All-In Runout, we hit end of hand ---
  // (This logic is new)
  if (endOfHand) {
    // We found no action. We must generate *only* the deal step.
    // The main loop consumed all steps up to the hand end. We need to
    // find the *correct* step for *this* street's deal.
    
    // 1. Determine which street we're looking for
    const targetBoardLength = (preDealBoardLength === 0) ? 6 : (preDealBoardLength === 6) ? 8 : 10;

    // 2. Find the *last* step that has that board length
    let lastStepOfThisStreetIdx = -1;
    // We only scan up to `rawStepsConsumed` (the steps we already validated)
    for (let i = 1; i < rawStepsConsumed; i++) {
        const currBoardLength = remainingRawSteps[i].current_universal_poker_json.board_cards.length;
        if (currBoardLength === targetBoardLength) {
            lastStepOfThisStreetIdx = i;
        }
        // If we find a *later* street, stop. lastStepOfThisStreetIdx will hold the correct one.
        if (currBoardLength > targetBoardLength) {
            break;
        }
    }

    // If we didn't find an exact match (e.g., hand ended mid-deal),
    // just use the last step we consumed.
    if (lastStepOfThisStreetIdx === -1) {
        lastStepOfThisStreetIdx = rawStepsConsumed - 1;
    }

    const stateForDeal = remainingRawSteps[lastStepOfThisStreetIdx];
    // We only consume the steps *up to and including* this one.
    const newRawStepsConsumed = lastStepOfThisStreetIdx + 1; 

    const boardCards = stateForDeal.current_universal_poker_json.board_cards;
    let stepType: RepeatedPokerStepType;
    if (boardCards.length === 6) stepType = 'deal-flop';
    else if (boardCards.length === 8) stepType = 'deal-turn';
    else if (boardCards.length === 10) stepType = 'deal-river';
    else {
      // This street's deal didn't complete, or data is bad.
      // This is an error because `determineGenerator` should only call us
      // if a new street is *starting*.
       throw new Error(
         `Unexpected board cards length ${boardCards.length} in all-in sequence for hand ${preDealStep.hand_number}. ` +
         `Was looking for length ${targetBoardLength}.`
       );
    }
    
    // --- Visual Step: The "Deal" Step (All-In) ---
    const dealStepPlayers: RepeatedPokerStepPlayer[] = [0, 1].map((id) => ({
        id, name: agents[id].Name, thumbnail: agents[id].ThumbnailUrl,
        cards: stateForDeal.current_universal_poker_json.player_hands[id],
        chipStack: STARTING_STACK_SIZE - stateForDeal.current_universal_poker_json.player_contributions[id],
        currentBet: stateForDeal.current_universal_poker_json.player_contributions[id],
        reward: null,
        actionDisplayText: '',
        thoughts: '',
        isDealer: preDealStep.dealer === id,
        isTurn: false,
        isWinner: false,
    }));

    const dealStep: RepeatedPokerStep = {
        stepType,
        communityCards: boardCards,
        pot: stateForDeal.current_universal_poker_json.pot_size,
        step: startIndex,
        winOdds: stateForDeal.current_universal_poker_json.odds,
        fiveCardBestHands: stateForDeal.current_universal_poker_json.best_five_card_hands,
        currentPlayer: -1,
        players: dealStepPlayers,
    };

    return {
      newSteps: [dealStep], // <-- ONLY the deal step
      rawStepsConsumed: newRawStepsConsumed, // <-- Consume ONLY steps for this street
    };
  }

  // --- CASE 3: Error ---
  // This means the loop finished without finding an action OR a new hand
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

    // Rule 1: Is this a boundary between hands
    if (nextReplayStep && nextReplayStep.hand_number > currentReplayStep.hand_number) {
        // This is the *last* step of the current hand.
        // It could be an action (Fold) or a result (Runout).
        // `generateHandEndSequence` handles both cases.
        return generateHandEndStepSequence;
    }
    // Rule 2: Is this the very last step of the entire replay?
    // TODO - we need an actual final state here - I
    // It should at the very least call generateFinalPlayerActionStep
    if (nextReplayStep === null) {
      return null
    }

    // Rule 3: Is this the start of a new hand?
    const hands = currentReplayStep.current_universal_poker_json.player_hands;
    if (hands[0] === '' && hands[1] === '') {
        return generatePreFlopStepSequence;
    }

    // Rule 4: Are community cards about to be dealt?
    if (nextReplayStep.current_universal_poker_json.board_cards.length > currentReplayStep.current_universal_poker_json.board_cards.length) {
        return generateCommunityCardStepSequence;
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
