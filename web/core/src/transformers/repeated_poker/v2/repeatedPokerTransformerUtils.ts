import { PokerReplayInfoAgent, PokerReplayStepHistoryParsed, } from './poker-replay-types';
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
const STARTING_STACK_SIZE = 200

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
  
  const players = [0, 1].map(id => ({
    id, name: agents[id].Name, thumbnail: agents[id].ThumbnailUrl,
    cards: afterJson.player_hands[id],
    chipStack: STARTING_STACK_SIZE - afterJson.player_contributions[id],
    currentBet: afterJson.player_contributions[id], 
    reward: null,
    actionDisplayText: id === actingPlayerId ? (actionString.toUpperCase() ?? '') : '',
    thoughts: id === actingPlayerId ? (actionObject?.action?.thoughts ?? '') : '',
    isDealer: stateAfterAction.dealer === id,
    isTurn: id === actingPlayerId,
    isWinner: false,
  }));

  return {
    stepType: 'player-action',
    communityCards: afterJson.board_cards.match(/.{1,2}/g) || [],
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

const generateFinalPlayerActionStep: StepGenerator = (remainingRawSteps, agents, startIndex) => {
  const stateBeforeAction = remainingRawSteps[0];
  const nextHandStartStep = remainingRawSteps[1];

  // The true "after" state is stored in the *next* hand's 'prev_universal_poker_json'
  // TODO - pretty skeptial of this, need to investigate a bit more
  const stateAfterAction = {
    ...stateBeforeAction,
    current_universal_poker_json: nextHandStartStep.prev_universal_poker_json,
  } as PokerReplayStepHistoryParsed;
  
  // --- VISUAL STEP 1: The Player Action (e.g., "FOLD") ---
  const actionStep = createPlayerActionStep(
    stateBeforeAction,
    stateAfterAction,
    agents,
    startIndex
  );

  // --- VISUAL STEP 2: The "Final" Hand Summary ---
  const finalJson = stateAfterAction.current_universal_poker_json;

  const finalStepPlayers: RepeatedPokerStepPlayer[] = [0, 1].map((id) => {
    const reward = nextHandStartStep.hand_returns[nextHandStartStep.hand_number - 1][id];
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
      isDealer: stateAfterAction.dealer === id,
      isTurn: false,
      isWinner,
    };
  });

  const finalStep: RepeatedPokerStep = {
    stepType: 'final',
    communityCards: finalJson.board_cards.match(/.{1,2}/g) || [],
    pot: finalJson.pot_size,
    step: startIndex + 1, // This is the second step we're adding
    winOdds: finalJson.odds,
    fiveCardBestHands: finalJson.best_five_card_hands,
    currentPlayer: -1,
    players: finalStepPlayers,
  };

  return {
    newSteps: [actionStep, finalStep], // Return BOTH steps
    rawStepsConsumed: 1 // Still only consumes the one "action" step
  };
};


const generatePreFlopStepSequence: StepGenerator = (remainingRawSteps, agents, startIndex) => {
  if (!remainingRawSteps || remainingRawSteps.length === 0) {
    throw new Error('Cannot generate pre-flop sequence: No raw steps were provided to the generator.');
  }

  // 1. Find the index of the first action step.
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
  const startingStack = 200;

  // Visual Step 1: Small Blind Post
  const sbPostPlayers: RepeatedPokerStepPlayer[] = [0, 1].map((id) => ({
    id,
    name: agents[id].Name,
    thumbnail: agents[id].ThumbnailUrl,
    cards: '',
    chipStack: id === sbPlayerId ? startingStack - small_blind : startingStack,
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
    communityCards: [],
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
          chipStack: startingStack - big_blind,
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

  // Look ahead to find the "before action" step (which has the .step object)
  let rawStepsConsumed = 1; // Start by consuming the preDealStep
  let actionStepIndex = -1;
  for (let i = 1; i < remainingRawSteps.length; i++) {
    const curr = remainingRawSteps[i];
    rawStepsConsumed++; // Consume this step
    // Safety break if we somehow hit a new hand
    if (curr.hand_number > preDealStep.hand_number) {
        // We found a new hand before finding an action, which is an error
        actionStepIndex = -1; // Ensure it fails the check below
        break;
    }
    if (curr.step) {
      // This is the "before" action step.
      actionStepIndex = i;
      break;
    }
  }

  if (actionStepIndex === -1) {
     throw new Error(
      `Data inconsistency: Could not find a player action step after the community card deal on hand ${preDealStep.hand_number}.`
     );
  }

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
  // We use the stateBeforeAction, as it has the complete board.
  const boardCards = stateBeforeAction.current_universal_poker_json.board_cards.match(/.{1,2}/g) || [];
  let stepType: RepeatedPokerStepType;
  if (boardCards.length === 3) stepType = 'deal-flop';
  else if (boardCards.length === 4) stepType = 'deal-turn';
  else if (boardCards.length === 5) stepType = 'deal-river';
  else {
    throw new Error(`Unexpected board cards length: ${boardCards.length}.`);
  }

  // --- Visual Step 1: The "Deal" Step ---
  // We use the state *before* the action to show the cards being dealt.
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

  // We must fix the currentBet value, as the factory uses the total contribution.
  // We want the contribution *for this street*.
  /* actionStep.players.forEach(player => {
    player.currentBet = stateAfterAction.current_universal_poker_json.player_contributions[player.id] - preDealStep.current_universal_poker_json.player_contributions[player.id];
  }); */

  return {
    newSteps: [dealStep, actionStep],
    rawStepsConsumed: rawStepsConsumed,
  };
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
        if (currentReplayStep.step) {
            // It's the FINAL player action (e.g., the Fold).
            // This generator will now create BOTH the action and the summary.
            return generateFinalPlayerActionStep;
        } else {
            // This is a "result" step (no action) just before the hand change.
            // This should not happen in our model and indicates a data error.
            return null; 
        }
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

            // If it wasn't the last step, it's a real error.
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
