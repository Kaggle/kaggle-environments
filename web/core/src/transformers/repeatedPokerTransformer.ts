import { PokerGameStep } from "../types";
const PLACEHOLDER_CARD = "2c";

interface PokerEvent {
  order?: number;
  stateIndex?: number;
  highlightPlayer: number | null;
  actionText: string;
  hideHoleCards: boolean;
  hideCommunity: boolean;
}

function _getActionStringsFromACPC(
  bettingString: string,
  nextPlayerIndex: number | null,
  numPlayers = 2,
) {
  const moves = [];
  const streets = bettingString.split("/");
  for (let streetIndex = 0; streetIndex < streets.length; streetIndex++) {
    const streetAction = streets[streetIndex];
    let i = 0;
    while (i < streetAction.length) {
      const char = streetAction[i];
      if (char === "r") {
        let amount = "";
        i++;
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

  const lastMove = moves.length > 0 ? moves[moves.length - 1] : null;
  const actionStrings = Array(numPlayers).fill("");

  if (lastMove) {
    if (typeof nextPlayerIndex === "number" && nextPlayerIndex >= 0) {
      const lastActor = (nextPlayerIndex + numPlayers - 1) % numPlayers;
      actionStrings[lastActor] = lastMove;
    } else {
      const inferredActor = (moves.length - 1) % numPlayers;
      actionStrings[inferredActor] = lastMove;
    }
  }

  return actionStrings;
}

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

export function splitCards(cardString: string) {
  if (!cardString) {
    return [];
  }
  return cardString.match(/.{1,2}/g) || [];
}

function formatActionDisplay(actionString: string) {
  if (!actionString) {
    return "";
  }
  const playerMatch = actionString.match(/move=([^\s]+)/);
  if (!playerMatch) {
    return "";
  }
  const moveRaw = playerMatch[1];
  const moveLower = moveRaw.toLowerCase();
  if (moveLower.startsWith("bet") || moveLower.startsWith("raise")) {
    const amountMatch = moveRaw.match(/\d+/);
    return amountMatch ? `r${amountMatch[0]}` : "r";
  }
  if (moveLower === "call") {
    return "c";
  }
  if (moveLower === "check") {
    return "k";
  }
  if (moveLower === "fold") {
    return "f";
  }
  return moveRaw;
}

function getBlinds(configuration: any) {
  const blindConfig =
    configuration?.openSpielGameParameters?.universal_poker_game_string?.blind;
  if (typeof blindConfig !== "string") {
    return { bigBlind: null, smallBlind: null };
  }
  const parts = blindConfig
    .trim()
    .split(/\s+/)
    .map((entry) => Number(entry))
    .filter((n) => !Number.isNaN(n));
  if (parts.length >= 2) {
    return { bigBlind: parts[0], smallBlind: parts[1] };
  }
  return { bigBlind: null, smallBlind: null };
}

interface StepHistoryResult {
  cards: string[];
  communityCards: "";
  bets: number[];
  playerActionStrings: number[];
  winOdds: string[];
}

export function parseStepHistoryData(
  universalPokerJSON: any,
  nextPlayerIndex: number | null,
  numPlayers = 2,
) {
  const result: StepHistoryResult = {
    cards: [],
    communityCards: "",
    bets: [],
    playerActionStrings: Array(numPlayers).fill(""),
    winOdds: ["0", "0"],
  };

  if (!universalPokerJSON) {
    return result;
  }

  const lines = universalPokerJSON.acpc_state.trim().split("\n");
  if (lines.length < 2) {
    return result;
  }

  const stateLine = lines[0];
  const spentLine = lines[1];

  if (spentLine) {
    const p0BetMatch = spentLine.match(/P0:\s*(\d+)/);
    const p1BetMatch = spentLine.match(/P1:\s*(\d+)/);

    const bets = [0, 0];

    if (p0BetMatch) {
      bets[0] = parseInt(p0BetMatch[1], 10);
    }

    if (p1BetMatch) {
      bets[1] = parseInt(p1BetMatch[1], 10);
    }

    result.bets = bets;
  }

  if (stateLine) {
    const stateParts = stateLine.split(":");
    const cardString = stateParts[stateParts.length - 1];
    const cardSegments = cardString.split("/");

    if (cardSegments[0]) {
      const playerHands = cardSegments[0].split("|");
      if (playerHands.length >= 2) {
        result.cards = [playerHands[0], playerHands[1]];
      }
    }

    result.communityCards = cardSegments.slice(1).filter(Boolean).join("");

    const bettingString = stateParts.slice(2, stateParts.length - 1).join(":");
    if (bettingString) {
      result.playerActionStrings = _getActionStringsFromACPC(
        bettingString,
        nextPlayerIndex,
        numPlayers,
      );
    }
  }

  const odds = universalPokerJSON.odds || [];
  const p0WinOdds = Number(odds[0] ?? 0).toLocaleString(undefined, {
    style: "percent",
    minimumFractionDigits: 2,
  });
  const p1WinOdds = Number(odds[1] ?? 0).toLocaleString(undefined, {
    style: "percent",
    minimumFractionDigits: 2,
  });
  result.winOdds = [p0WinOdds, p1WinOdds];

  return result;
}

export function getCommunityCardsFromUniversal(
  universal: string,
  numPlayers: number,
) {
  const parsed = parseStepHistoryData(universal, null, numPlayers);
  const cards = splitCards(parsed.communityCards);
  const actual = cards.filter(
    (card) => card && card.toLowerCase() !== PLACEHOLDER_CARD,
  );
  if (actual.length < 3) {
    return [];
  }
  return actual;
}

export function buildTimeline(environment: any, numPlayers: 2) {
  const stateHistory = environment?.info?.stateHistory || [];
  if (!stateHistory.length) {
    return [];
  }

  const parsedStates = stateHistory.map((entry: string, idx: number) => {
    const outer = JSON.parse(entry);
    return {
      idx,
      outer,
      universal: JSON.parse(outer.current_universal_poker_json),
    };
  });

  const hands = [];
  let currentHandNumber = parsedStates[0]?.outer?.hand_number ?? 0;
  let currentStates: any = [];
  parsedStates.forEach((stateInfo: any) => {
    const handNumber = stateInfo.outer?.hand_number ?? currentHandNumber;
    if (handNumber !== currentHandNumber) {
      if (currentStates.length) {
        hands.push({ handNumber: currentHandNumber, states: currentStates });
      }
      currentStates = [];
      currentHandNumber = handNumber;
    }
    currentStates.push(stateInfo);
  });
  if (currentStates.length) {
    hands.push({ handNumber: currentHandNumber, states: currentStates });
  }

  const processedSteps = environment || [];
  const actionsByHand = new Map();
  processedSteps.forEach((step: PokerGameStep) => {
    const handNumber = step?.hand ?? 0;
    if (!actionsByHand.has(handNumber)) {
      actionsByHand.set(handNumber, []);
    }
    if (
      !step?.isEndState &&
      step?.step?.action &&
      step.step.action.submission !== -1
    ) {
      const actionString = step.step.action.actionString || "";
      const playerMatch = actionString.match(/player=(\d+)/);
      const playerIndex = playerMatch ? parseInt(playerMatch[1], 10) : null;
      actionsByHand.get(handNumber).push({
        playerIndex,
        actionText: formatActionDisplay(actionString),
        stateHistoryIndex: step.stateHistoryIndex,
      });
    }
  });

  const events: PokerEvent[] = [];
  let orderCounter = 0;
  const pushEvent = (stateIndex: number, event: PokerEvent) => {
    events.push({
      order: orderCounter++,
      stateIndex,
      highlightPlayer: event.highlightPlayer,
      actionText: event.actionText,
      hideHoleCards: event.hideHoleCards,
      hideCommunity: event.hideCommunity,
    });
  };

  hands.forEach(({ handNumber, states }) => {
    if (!states.length) {
      return;
    }

    const firstState = states[0];
    const dealer = firstState.outer?.dealer ?? 0;
    const { bigBlind, smallBlind } = getBlinds(environment.configuration);
    const smallBlindPlayer = dealer % numPlayers;
    const bigBlindPlayer = (dealer + 1) % numPlayers;

    pushEvent(firstState.idx, {
      highlightPlayer: null,
      actionText: "",
      hideHoleCards: true,
      hideCommunity: true,
    });
    pushEvent(firstState.idx, {
      highlightPlayer: smallBlindPlayer,
      actionText: smallBlind != null ? `SB ${smallBlind}` : "SB",
      hideHoleCards: true,
      hideCommunity: true,
    });
    pushEvent(firstState.idx, {
      highlightPlayer: bigBlindPlayer,
      actionText: bigBlind != null ? `BB ${bigBlind}` : "BB",
      hideHoleCards: true,
      hideCommunity: true,
    });

    const firstActionState =
      states.find(
        (stateInfo: any) => stateInfo.universal.current_player !== -1,
      ) || firstState;
    pushEvent(firstActionState.idx, {
      highlightPlayer: null,
      actionText: "",
      hideHoleCards: false,
      hideCommunity: true,
    });

    const actions = actionsByHand.get(handNumber) || [];
    actions.forEach((action: any) => {
      const targetIndex =
        typeof action.stateHistoryIndex === "number"
          ? action.stateHistoryIndex
          : states[0].idx;
      const postState =
        states.find((stateInfo: any) => stateInfo.idx > targetIndex) ||
        states[states.length - 1];
      pushEvent(postState.idx, {
        highlightPlayer: action.playerIndex,
        actionText: action.actionText,
        hideHoleCards: false,
        hideCommunity: false,
      });
    });

    let currentStageCommunityLength = 0;
    states.forEach((stateInfo: any) => {
      const communityCards = getCommunityCardsFromUniversal(
        stateInfo.universal,
        numPlayers,
      );
      const communityLength = communityCards.length;
      if (communityLength > currentStageCommunityLength) {
        currentStageCommunityLength = communityLength;
        if (
          communityLength === 3 ||
          communityLength === 4 ||
          communityLength === 5
        ) {
          pushEvent(stateInfo.idx, {
            highlightPlayer: null,
            actionText: "",
            hideHoleCards: false,
            hideCommunity: false,
          });
        }
      }
    });
  });

  events.sort((a: PokerEvent, b: PokerEvent) => a.order! - b.order!);
  return events.map(
    ({
      stateIndex,
      highlightPlayer,
      actionText,
      hideHoleCards,
      hideCommunity,
    }) => ({
      stateIndex,
      highlightPlayer,
      actionText,
      hideHoleCards: !!hideHoleCards,
      hideCommunity: !!hideCommunity,
    }),
  );
}

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
  return "TODO";
};

interface TimelineEvent {
  stateIndex: number | undefined;
  highlightPlayer: number | null;
  actionText: string;
  hideHoleCards: boolean;
  hideCommunity: boolean;
}

export const getPokerStepsWithEndStates = (
  environment: any,
): PokerGameStep[] => {
  const stepsWithEndStates: PokerGameStep[] = [];
  let handCount = 0;
  let stateHistoryPointer = 0;

  const stateHistory = environment.state_history ?? [];
  const steps = environment.steps ?? [];

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

    if (step) {
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
    }

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

  // Build timeline from the environment
  const timeline: TimelineEvent[] = buildTimeline(
    environment,
    /* numPlayers= */ 2,
  );

  // Create new steps based on the timeline
  const timelineSteps = timeline.map(
    (timelineEvent: TimelineEvent): PokerGameStep => {
      // Find the corresponding step in stepsWithEndStates by matching stateHistoryIndex
      const matchingStep = stepsWithEndStates.find(
        (step) => step.stateHistoryIndex === timelineEvent.stateIndex,
      );

      if (!matchingStep) {
        // Find the hand number by looking at the closest previous step
        let handNumber = 0;
        if (timelineEvent.stateIndex !== undefined) {
          const previousSteps = stepsWithEndStates.filter(
            (step) =>
              step.stateHistoryIndex !== undefined &&
              timelineEvent.stateIndex !== undefined &&
              step.stateHistoryIndex < timelineEvent.stateIndex,
          );
          if (previousSteps.length > 0) {
            handNumber = previousSteps[previousSteps.length - 1].hand;
          }
        }

        // Create a new step if there's no matching step
        return {
          hand: handNumber,
          isEndState: false,
          step: null,
          stateHistory:
            timelineEvent.stateIndex !== undefined
              ? stateHistory[timelineEvent.stateIndex]
              : "",
          stateHistoryIndex: timelineEvent.stateIndex,
          actionText: timelineEvent.actionText,
          // highlightPlayer: timelineEvent.highlightPlayer,
          // hideHoleCards: timelineEvent.hideHoleCards,
          // hideCommunity: timelineEvent.hideCommunity,
        };
      }

      // Return a new step with timeline data included
      return {
        ...matchingStep,
        actionText: timelineEvent.actionText,
        // highlightPlayer: timelineEvent.highlightPlayer,
        // hideHoleCards: timelineEvent.hideHoleCards,
        // hideCommunity: timelineEvent.hideCommunity,
      };
    },
  );

  console.log(timelineSteps);

  return timelineSteps;
};
