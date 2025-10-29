import { PokerGameStep } from "../types";

const suitMap: Record<string, string> = {
  s: "♠️",
  h: "❤️",
  d: "♦️",
  c: "♣️",
};

const rankMap: Record<string, string> = {
  "2": "Two",
  "3": "Three",
  "4": "Four",
  "5": "Five",
  "6": "Six",
  "7": "Seven",
  "8": "Eight",
  "9": "Nine",
  T: "Ten",
  J: "Jack",
  Q: "Queen",
  K: "King",
  A: "Ace",
};

const _isStateHistoryAgentAction = (stateHistoryEntry: string): boolean =>
  JSON.parse(JSON.parse(stateHistoryEntry).current_universal_poker_json)
    .current_player !== -1;

const _isStateHistoryEntryInitial = (stateHistoryEntry: string): boolean => {
  const state = JSON.parse(
    JSON.parse(stateHistoryEntry).current_universal_poker_json,
  );
  return state.acpc_state.startsWith("STATE:0::2c2c|2c2c");
};

const _getMoveHistoryFromACPC = (acpcState: string): string => {
  // Parse the ACPC state line to extract the betting string
  // Example ACPC state: "STATE:0:r5c/cr11c/:6cKd|AsJc/7hQh6d/2c"
  const lines = acpcState.trim().split("\n");
  if (lines.length < 1) {
    return "";
  }

  const stateLine = lines[0]; // First line contains the state
  const stateParts = stateLine.split(":");

  // The betting string is everything between the 2nd colon and the last colon
  // stateParts[0] = "STATE"
  // stateParts[1] = "0" (hand number)
  // stateParts[2...-1] = betting string
  // stateParts[last] = cards
  if (stateParts.length < 4) {
    return "";
  }

  const bettingString = stateParts.slice(2, stateParts.length - 1).join(":");
  return bettingString;
};

function _getMovesFromBettingStringACPC(bettingString: string): string[] {
  const moves = [];

  // Split the action string by street (e.g., ["r5c", "cr11f"])
  const streets = bettingString.split("/");

  // Process each street's actions
  for (let streetIndex = 0; streetIndex < streets.length; streetIndex++) {
    const streetAction = streets[streetIndex];
    let i = 0;

    while (i < streetAction.length) {
      const char = streetAction[i];

      if (char === "r") {
        // 'r' (raise)
        let amount = "";
        i++;
        // Continue to parse all digits of the raise amount
        while (
          i < streetAction.length &&
          streetAction[i] >= "0" &&
          streetAction[i] <= "9"
        ) {
          amount += streetAction[i];
          i++;
        }
        moves.push(`r${amount}`);
      } else {
        moves.push(char);
        i++;
      }
    }
  }

  return moves;
}

const _getEndCondition = (
  stateHistory: any[],
  stateHistoryPointer: number,
  currentPlayer: string,
): {
  handConclusion: "fold" | "showdown";
  winner: -1 | 0 | 1; // -1 for the rare event of a tie
  bestFiveCardHands?: string[];
  bestHandRankType?: string[];
} => {
  const current_player = parseInt(currentPlayer);

  if (stateHistoryPointer >= stateHistory.length - 1) {
    return {
      // TODO: handle tail end
      // for now, fold + tie = impossible state
      handConclusion: "fold",
      winner: -1,
      bestFiveCardHands: [],
    };
  }

  let next_prev_universal_poker_json = {
    acpc_state: "",
    best_five_card_hands: ["", ""],
    best_hand_rank_types: ["", ""],
  };

  // since the current_universal_poker_json does not contain the end move in it's history,
  // we need to go to the prev_universal_poker_json of the next one
  try {
    next_prev_universal_poker_json = JSON.parse(
      JSON.parse(stateHistory[stateHistoryPointer + 1])
        .prev_universal_poker_json,
    );
  } catch {
    console.error("prev_universal_poker_json parsing failed");
  }

  // if the stateHistory doesn't end in a fold, it was a showdown
  const bettingString = _getMoveHistoryFromACPC(
    next_prev_universal_poker_json.acpc_state,
  );

  const moves = _getMovesFromBettingStringACPC(bettingString);

  // Fold case
  if (moves.pop() === "f") {
    return {
      handConclusion: "fold",
      winner: current_player === 0 ? 1 : 0,
    };
  }

  // Showdown case
  return {
    handConclusion: "showdown",
    winner: current_player === 0 ? 1 : 0,
    bestFiveCardHands: next_prev_universal_poker_json.best_five_card_hands,
    bestHandRankType: next_prev_universal_poker_json.best_hand_rank_types,
  };
};

const _getHumanReadableHand = (cards: string): string => {
  if (!cards || cards.length === 0) return "";

  // Split the input into individual cards (each card is 2 characters)
  const cardArray = [];
  for (let i = 0; i < cards.length; i += 2) {
    if (i + 1 < cards.length) {
      const rank = cards[i];
      const suit = cards[i + 1];

      if (rankMap[rank] && suitMap[suit]) {
        cardArray.push(`${rankMap[rank]} of ${suitMap[suit]}`);
      }
    }
  }

  // Join the cards with commas and 'and' for the last card
  if (cardArray.length === 0) return "";
  if (cardArray.length === 1) return cardArray[0];

  return (
    cardArray.slice(0, -1).join(", ") + ", " + cardArray[cardArray.length - 1]
  );
};

/**
 * Finds new cards by comparing current and previous card states
 */
const _findNewCards = (
  currentCardString: string,
  previousCardString: string,
): { newCards: string; isNewRound: boolean } => {
  // Parse the current card string
  const currentCardSegments = currentCardString.split("|");
  const currentPlayerCards = currentCardSegments[0] || "";
  const currentCommunitySegments =
    currentCardSegments.length > 1 ? currentCardSegments[1].split("/") : [];

  // Parse the previous card string
  const previousCardSegments = previousCardString.split("|");
  const previousPlayerCards = previousCardSegments[0] || "";
  const previousCommunitySegments =
    previousCardSegments.length > 1 ? previousCardSegments[1].split("/") : [];

  // Check if this is a new hand (player cards changed)
  const isNewRound = currentPlayerCards !== previousPlayerCards;

  let newCards = "";

  // If not a new hand, find which street has new cards
  if (!isNewRound) {
    // Find the first street that differs or is new
    for (let i = 0; i < currentCommunitySegments.length; i++) {
      if (
        i >= previousCommunitySegments.length ||
        currentCommunitySegments[i] !== previousCommunitySegments[i]
      ) {
        // This street has new cards
        newCards = currentCommunitySegments[i];
        break;
      }
    }
  }

  return { newCards, isNewRound };
};

const _parseCards = (
  currentStateHistory: string,
  previousStateHistory: string,
  players: string[],
) => {
  const currentState = JSON.parse(
    JSON.parse(currentStateHistory).current_universal_poker_json,
  );
  const previousState = JSON.parse(
    JSON.parse(previousStateHistory).current_universal_poker_json,
  );

  /**
   * Example lines:
   * STATE:0:r5c/cr9c/:Ks4s|5hAs/2dJs7s/Qh
   * Spent: [P0: 9  P1: 9  ]
   */
  const lines = currentState.trim().split("\n");
  if (lines.length < 2) {
    return "";
  }

  const stateParts = lines[0].split(":");

  // Extract the card string from the current state
  const currentCardString = stateParts[stateParts.length - 1]; // example: "6cKd|AsJc/7hQh6d/2c"

  // Extract betting information
  const spentLine = lines.find((line: string) => line.startsWith("Spent:"));
  let blindsInfo = "";

  if (spentLine) {
    // Parse the spent line to extract player bets
    const spentMatch = spentLine.match(/\[P0:\s*(\d+)\s*P1:\s*(\d+)\s*\]/);
    if (spentMatch && spentMatch.length >= 3) {
      const p0Spent = parseInt(spentMatch[1], 10);
      const p1Spent = parseInt(spentMatch[2], 10);

      // In poker, typically the small blind is half the big blind
      if (p0Spent > 0 && p1Spent > 0) {
        const smallBlind = Math.min(p0Spent, p1Spent);
        const bigBlind = Math.max(p0Spent, p1Spent);

        if (p0Spent < p1Spent) {
          blindsInfo = `\n\n**Blinds:** ${players[0]} (SB: ${smallBlind}), ${players[1]} (BB: ${bigBlind})`;
        } else {
          blindsInfo = `\n\n**Blinds:** ${players[1]} (SB: ${smallBlind}), ${players[0]} (BB: ${bigBlind})`;
        }
      }
    }
  }

  // Get previous card string if available
  let previousCardString = "";
  if (previousState) {
    const prevLines = previousState.trim().split("\n");
    if (prevLines.length >= 1) {
      const prevStateParts = prevLines[0].split(":");
      previousCardString = prevStateParts[prevStateParts.length - 1];
    }
  }

  // Find new cards by comparing current and previous states
  const { newCards, isNewRound } = _findNewCards(
    currentCardString,
    previousCardString,
  );

  // Split card string by '/' to separate hand block from board blocks
  const cardSegments = currentCardString.split("/"); // example: ["6cKd|AsJc", "7hQh6d", "2c"]

  // Parse player hands
  let cards;

  // Parse the first segment (player hands)
  if (cardSegments[0]) {
    const playerHands = cardSegments[0].split("|");
    if (playerHands.length >= 2) {
      // example: "6cKd"
      cards = `**Player 1:** ${_getHumanReadableHand(playerHands[0])}\n\n**Player 2:** ${_getHumanReadableHand(playerHands[1])}`;
    }
  }

  // Append blinds info to the cards string
  if (cards && blindsInfo) {
    cards += blindsInfo;
  }

  const revealedCards = newCards
    ? `**Revealed:** ${_getHumanReadableHand(newCards)}`
    : "";

  // Make sure we always return something meaningful for a new round
  if (isNewRound) {
    return cards ? `**New Round**\n\n${cards}` : "**New Round**";
  }

  return revealedCards;
};

export const getPokerStepLabel = (gameStep: PokerGameStep) => {
  const decision = gameStep.step?.action?.actionString ?? "";
  if (decision.length > 0) {
    return decision
      .split("move=")[1]
      .replace(/([a-zA-Z])(\d)/g, "$1 $2")
      .replace(/(\d)([A-Z])/g, "$1 $2");
  }

  return "";
};

export const getPokerStepDescription = (gameStep: PokerGameStep) => {
  if (gameStep.step?.action?.thoughts) {
    return gameStep.step.action.thoughts;
  } else if (gameStep.isEndState) {
    if (gameStep.handConclusion === "showdown") {
      return `
## Player ${gameStep.winner} wins round ${gameStep.hand + 1} 
### Wins with ${gameStep.bestHandRankType} 
### ${gameStep.bestFiveCardHands} 
`;
    } else {
      return `
## Player ${gameStep.winner} wins round ${gameStep.hand + 1} 
### Other player folds
`;
    }
  }

  // TODO player names
  return _parseCards(gameStep.stateHistory, "", ["Player 1", "Player 2"]);
};

export const getPokerStepsWithEndStates = (
  steps: any[],
  stateHistory: any[],
): PokerGameStep[] => {
  const stepsWithEndStates: PokerGameStep[] = [];
  let handCount = 0;
  let stateHistoryPointer = 0;

  const advanceToNextAgentEntry = () => {
    while (
      stateHistoryPointer < stateHistory.length &&
      !_isStateHistoryAgentAction(stateHistory[stateHistoryPointer])
    ) {
      stateHistoryPointer++;
    }
  };

  advanceToNextAgentEntry();

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    let lastActionPointer = -1;

    step.forEach((s: any) => {
      if (s.action.submission !== -1) {
        if (stateHistoryPointer >= stateHistory.length) {
          return;
        }

        const preActionPointer = stateHistoryPointer;

        stepsWithEndStates.push({
          hand: handCount,
          isEndState: false,
          step: s,
          stateHistory: stateHistory[stateHistoryPointer],
          stateHistoryIndex: preActionPointer,
        });

        lastActionPointer = preActionPointer;
        stateHistoryPointer++;

        const postActionPointer = stateHistoryPointer;

        if (
          postActionPointer < stateHistory.length &&
          !_isStateHistoryAgentAction(stateHistory[postActionPointer]) &&
          !_isStateHistoryEntryInitial(stateHistory[postActionPointer])
        ) {
          stepsWithEndStates.push({
            hand: handCount,
            isEndState: false,
            step: null,
            stateHistory: stateHistory[postActionPointer],
            stateHistoryIndex: postActionPointer,
            postActionOf: preActionPointer,
          });
        }

        advanceToNextAgentEntry();
      }
    });

    let lookaheadPointer = lastActionPointer + 1;
    while (
      lookaheadPointer < stateHistory.length &&
      !_isStateHistoryAgentAction(stateHistory[lookaheadPointer]) &&
      !_isStateHistoryEntryInitial(stateHistory[lookaheadPointer])
    ) {
      lookaheadPointer++;
    }

    const isEndState =
      lastActionPointer !== -1 &&
      (lookaheadPointer >= stateHistory.length ||
        _isStateHistoryEntryInitial(stateHistory[lookaheadPointer]));

    if (isEndState) {
      const endState = _getEndCondition(
        stateHistory,
        lastActionPointer,
        step[0].observation.currentPlayer,
      );

      stepsWithEndStates.push({
        hand: handCount,
        isEndState: true,
        step: null,
        stateHistory: stateHistory[lastActionPointer],
        stateHistoryIndex: lastActionPointer,
        ...endState,
      });

      handCount++;
    }
  }

  return stepsWithEndStates;
};
