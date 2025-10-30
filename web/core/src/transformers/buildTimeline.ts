import { _getReadableMovesFromBettingStringACPC } from "./repeatedPokerTransformer";

const PLACEHOLDER_CARD = "2c";

export function getActionStringsFromACPC(
  bettingString: string,
  player: number,
  numPlayers: number = 2,
): string[] {
  console.log(player);
  // Initialize action strings for each player
  const actionStrings = Array(numPlayers).fill("");

  // If there's no betting string, return empty strings
  if (!bettingString) return actionStrings;

  // Split the betting string by streets
  const streets = bettingString.split("/");

  // In heads-up poker:
  // Preflop: SB (player 1) acts first, then BB (player 0)
  // Postflop: BB (player 0) acts first, then SB (player 1)

  // Track which player made each action
  let currentPlayer = 1; // SB acts first preflop
  const playerActions: { [key: number]: string } = {};

  for (let streetIndex = 0; streetIndex < streets.length; streetIndex++) {
    // Reset player order for postflop streets
    if (streetIndex > 0) {
      currentPlayer = 0; // BB acts first postflop
    }

    const streetBetting = streets[streetIndex];
    let i = 0;

    while (i < streetBetting.length) {
      const char = streetBetting[i];

      if (char === "c" || char === "f") {
        // Call/Check or Fold - simple actions
        const readableAction = _getReadableMovesFromBettingStringACPC(char)[0];
        playerActions[currentPlayer] = readableAction;
        i++;
      } else if (char === "r") {
        // Raise/Bet - need to extract the amount
        let raiseAmount = "";
        i++; // Move past 'r'

        // Extract the raise amount
        while (
          i < streetBetting.length &&
          streetBetting[i] >= "0" &&
          streetBetting[i] <= "9"
        ) {
          raiseAmount += streetBetting[i];
          i++;
        }

        const readableAction = _getReadableMovesFromBettingStringACPC(
          `r${raiseAmount}`,
        )[0];
        playerActions[currentPlayer] = readableAction;
      } else {
        // Unknown character, just skip
        i++;
      }

      // Move to next player
      currentPlayer = (currentPlayer + 1) % numPlayers;
    }
  }

  // Set the action strings for each player
  for (let i = 0; i < numPlayers; i++) {
    if (playerActions[i]) {
      actionStrings[i] = playerActions[i];
    }
  }

  return actionStrings;
}

interface UniversalPokerJSON {
  acpc_state: string;
  odds?: number[];
  current_player?: number;
  starting_stacks?: number[];
  player_contributions?: number[];
}

interface ParsedStepHistoryData {
  cards: string[];
  communityCards: string;
  bets: number[];
  playerActionStrings: string[];
  winOdds: (string | number)[]; // Changed to allow both number and string for win odds
}

function _parseStepHistoryData(
  universalPokerJSON: UniversalPokerJSON | null,
  numPlayers: number = 2,
): ParsedStepHistoryData {
  const result: ParsedStepHistoryData = {
    cards: [],
    communityCards: "",
    bets: [],
    playerActionStrings: Array(numPlayers).fill(""),
    winOdds: [0, 0],
  };

  if (!universalPokerJSON) {
    return result;
  }

  const lines: string[] = universalPokerJSON.acpc_state.trim().split("\n");
  if (lines.length < 2) {
    return result;
  }

  const stateLine: string = lines[0];
  const spentLine: string = lines[1];

  if (spentLine) {
    const p0BetMatch = spentLine.match(/P0:\s*(\d+)/);
    const p1BetMatch = spentLine.match(/P1:\s*(\d+)/);

    const bets: number[] = [0, 0];

    if (p0BetMatch) {
      bets[0] = parseInt(p0BetMatch[1], 10);
    }

    if (p1BetMatch) {
      bets[1] = parseInt(p1BetMatch[1], 10);
    }

    result.bets = bets;
  }

  if (stateLine) {
    const stateParts: string[] = stateLine.split(":");
    const cardString: string = stateParts[stateParts.length - 1];
    const cardSegments: string[] = cardString.split("/");

    if (cardSegments[0]) {
      const playerHands: string[] = cardSegments[0].split("|");
      if (playerHands.length >= 2) {
        result.cards = [playerHands[0], playerHands[1]];
      }
    }

    result.communityCards = cardSegments.slice(1).filter(Boolean).join("");

    const bettingString: string = stateParts
      .slice(2, stateParts.length - 1)
      .join(":");
    if (bettingString) {
      result.playerActionStrings = getActionStringsFromACPC(
        bettingString,
        numPlayers,
      );
    }
  }

  const odds: number[] = universalPokerJSON.odds || [];
  const p0WinOdds: string = Number(odds[0] ?? 0).toLocaleString(undefined, {
    style: "percent",
    minimumFractionDigits: 2,
  });
  const p1WinOdds: string = Number(odds[1] ?? 0).toLocaleString(undefined, {
    style: "percent",
    minimumFractionDigits: 2,
  });
  result.winOdds = [p0WinOdds, p1WinOdds];

  return result;
}

function splitCards(cardString: string | null): string[] {
  if (!cardString) {
    return [];
  }
  return cardString.match(/.{1,2}/g) || [];
}

function isPlaceholderString(cardString: string | null): boolean {
  return typeof cardString === "string" && /^((2c)+)$/i.test(cardString);
}

function sanitizeCardList(cards: string[] | null): string[] {
  return (cards || []).filter((card) => card);
}

function formatActionDisplay(actionString: string | null): string {
  if (!actionString) {
    return "";
  }
  const playerMatch = actionString.match(/move=([^\s]+)/);
  if (!playerMatch) {
    return "";
  }
  const moveRaw: string = playerMatch[1];
  const moveLower: string = moveRaw.toLowerCase();
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

interface BlindConfig {
  bigBlind: number | null;
  smallBlind: number | null;
}

function getBlinds(configuration: any): BlindConfig {
  const blindConfig =
    configuration?.openSpielGameParameters?.universal_poker_game_string?.blind;
  if (typeof blindConfig !== "string") {
    return { bigBlind: null, smallBlind: null };
  }
  const parts: number[] = blindConfig
    .trim()
    .split(/\s+/)
    .map((entry: string) => Number(entry))
    .filter((n: number) => !Number.isNaN(n));
  if (parts.length >= 2) {
    return { bigBlind: parts[0], smallBlind: parts[1] };
  }
  return { bigBlind: null, smallBlind: null };
}

function getCommunityCardsFromUniversal(
  universal: UniversalPokerJSON,
  numPlayers: number,
): string[] {
  const parsed: ParsedStepHistoryData = _parseStepHistoryData(
    universal,
    numPlayers,
  );
  const cards: string[] = splitCards(parsed.communityCards);
  const actual: string[] = cards.filter(
    (card) => card && card.toLowerCase() !== PLACEHOLDER_CARD,
  );
  if (actual.length < 3) {
    return [];
  }
  return actual;
}

function getHandCardsFromUniversal(
  universal: UniversalPokerJSON,
  numPlayers: number,
): string[][] {
  const parsed: ParsedStepHistoryData = _parseStepHistoryData(
    universal,
    numPlayers,
  );
  return (parsed.cards || []).map((cardString: string) => {
    if (isPlaceholderString(cardString)) {
      return [];
    }
    return sanitizeCardList(splitCards(cardString));
  });
}

interface TimelineEvent {
  order: number;
  stateIndex: number;
  highlightPlayer: number | null;
  actionText: string;
  hideHoleCards: boolean;
  hideCommunity: boolean;
}

interface HandState {
  idx: number;
  outer: any;
  universal: UniversalPokerJSON;
}

interface HandData {
  handNumber: number;
  states: HandState[];
}

export function buildTimeline(
  environment: any,
  numPlayers: number,
): Omit<TimelineEvent, "order">[] {
  const stateHistory: string[] = environment?.info?.stateHistory || [];
  if (!stateHistory.length) {
    return [];
  }

  const parsedStates: HandState[] = stateHistory.map(
    (entry: string, idx: number) => {
      const outer = JSON.parse(entry);
      return {
        idx,
        outer,
        universal: JSON.parse(outer.current_universal_poker_json),
      };
    },
  );

  const hands: HandData[] = [];
  let currentHandNumber: number = parsedStates[0]?.outer?.hand_number ?? 0;
  let currentStates: HandState[] = [];
  parsedStates.forEach((stateInfo: HandState) => {
    const handNumber: number =
      stateInfo.outer?.hand_number ?? currentHandNumber;
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

  const processedSteps: any[] =
    environment.__processedSteps || environment.steps || [];
  const actionsByHand = new Map<
    number,
    {
      playerIndex: number | null;
      actionText: string;
      stateHistoryIndex: number | null;
    }[]
  >();
  processedSteps.forEach((step: any) => {
    const handNumber: number = step?.hand ?? 0;
    if (!actionsByHand.has(handNumber)) {
      actionsByHand.set(handNumber, []);
    }
    if (
      !step?.isEndState &&
      step?.step?.action &&
      step.step.action.submission !== -1
    ) {
      const actionString: string = step.step.action.actionString || "";
      const playerMatch = actionString.match(/player=(\d+)/);
      const playerIndex: number | null = playerMatch
        ? parseInt(playerMatch[1], 10)
        : null;
      actionsByHand.get(handNumber)?.push({
        playerIndex,
        actionText: formatActionDisplay(actionString),
        stateHistoryIndex: step.stateHistoryIndex,
      });
    }
  });

  const events: TimelineEvent[] = [];
  let orderCounter: number = 0;
  const pushEvent = (
    stateIndex: number,
    event: Omit<TimelineEvent, "order" | "stateIndex">,
  ) => {
    events.push({
      order: orderCounter++,
      stateIndex,
      highlightPlayer: event.highlightPlayer,
      actionText: event.actionText,
      hideHoleCards: event.hideHoleCards,
      hideCommunity: event.hideCommunity,
    });
  };

  hands.forEach(({ handNumber, states }: HandData) => {
    if (!states.length) {
      return;
    }

    const firstState: HandState = states[0];
    const dealer: number = firstState.outer?.dealer ?? 0;
    const { bigBlind, smallBlind } = getBlinds(environment.configuration);
    const smallBlindPlayer: number = dealer % numPlayers;
    const bigBlindPlayer: number = (dealer + 1) % numPlayers;

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

    const firstActionState: HandState =
      states.find(
        (stateInfo: HandState) => stateInfo.universal.current_player !== -1,
      ) || firstState;
    pushEvent(firstActionState.idx, {
      highlightPlayer: null,
      actionText: "",
      hideHoleCards: false,
      hideCommunity: true,
    });

    const actions = actionsByHand.get(handNumber) || [];
    actions.forEach(
      (action: {
        playerIndex: number | null;
        actionText: string;
        stateHistoryIndex: number | null;
      }) => {
        const targetIndex: number =
          typeof action.stateHistoryIndex === "number"
            ? action.stateHistoryIndex
            : states[0].idx;
        const postState: HandState =
          states.find((stateInfo: HandState) => stateInfo.idx > targetIndex) ||
          states[states.length - 1];
        pushEvent(postState.idx, {
          highlightPlayer: action.playerIndex,
          actionText: action.actionText,
          hideHoleCards: false,
          hideCommunity: false,
        });
      },
    );

    let currentStageCommunityLength: number = 0;
    states.forEach((stateInfo: HandState) => {
      const communityCards: string[] = getCommunityCardsFromUniversal(
        stateInfo.universal,
        numPlayers,
      );
      const communityLength: number = communityCards.length;
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

  events.sort((a: TimelineEvent, b: TimelineEvent) => a.order - b.order);
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

function getTimeline(
  environment: any,
  numPlayers: number,
): Omit<TimelineEvent, "order">[] {
  if (!environment.__timeline) {
    environment.__timeline = buildTimeline(environment, numPlayers);
  }
  return environment.__timeline;
}

interface UniversalStateInfo {
  outer: any;
  universal: UniversalPokerJSON;
}

function getUniversalState(
  environment: any,
  index: number,
): UniversalStateInfo | null {
  const entry: string | undefined = environment?.info?.stateHistory?.[index];
  if (!entry) {
    return null;
  }
  const outer = JSON.parse(entry);
  return {
    outer,
    universal: JSON.parse(outer.current_universal_poker_json),
  };
}

interface PlayerState {
  id: string;
  name: string;
  thumbnail: string;
  stack: number;
  cards: string[];
  currentBet: number;
  isDealer: boolean;
  isTurn: boolean;
  isLastActor: boolean;
  reward: number | null;
  actionDisplayText: string;
}

interface PokerStateForStep {
  players: PlayerState[];
  communityCards: string[];
  pot: number;
  isTerminal: boolean;
  rawObservation: UniversalPokerJSON;
  step: number;
  winOdds: (string | number)[];
  fiveCardBestHands: any[]; // 'any' for unknown structure
  currentPlayer: number;
  winner: number;
}

export const getPokerStateForStep = (
  environment: any,
  step: number,
): PokerStateForStep | null => {
  const numPlayers: number = 2;
  if (!environment || !environment.info?.stateHistory) {
    return null;
  }

  const timeline: Omit<TimelineEvent, "order">[] = getTimeline(
    environment,
    numPlayers,
  );

  const event: Omit<TimelineEvent, "order"> | undefined = timeline[step];
  if (!event) {
    return null;
  }
  const stateInfo: UniversalStateInfo | null = getUniversalState(
    environment,
    event.stateIndex,
  );
  if (!stateInfo) {
    return null;
  }

  const parsedStateHistory: ParsedStepHistoryData = _parseStepHistoryData(
    stateInfo.universal,
    numPlayers,
  );

  const startingStacks: number[] =
    stateInfo.universal?.starting_stacks || Array(numPlayers).fill(0);
  const contributions: number[] =
    stateInfo.universal?.player_contributions ||
    parsedStateHistory.bets ||
    Array(numPlayers).fill(0);
  const rewards: any[] = stateInfo.outer?.hand_returns || [];
  const communityCards: string[] = getCommunityCardsFromUniversal(
    stateInfo.universal,
    numPlayers,
  );

  const players: PlayerState[] = Array(numPlayers)
    .fill(null)
    .map((_, i: number) => {
      const agentName: string =
        environment?.info?.TeamNames?.[i] || `Player ${i}`;
      const thumbnail: string = environment?.info?.Agents?.[i]?.ThumbnailUrl;
      return {
        id: `player${i}`,
        name: agentName,
        thumbnail,
        stack: startingStacks[i] - (contributions[i] || 0),
        cards: [],
        currentBet: contributions[i] || 0,
        isDealer: stateInfo.outer?.dealer === i,
        isTurn: stateInfo.universal?.current_player === i,
        isLastActor: event.highlightPlayer === i,
        reward: rewards[0]?.[i] ?? null,
        actionDisplayText: event.highlightPlayer === i ? event.actionText : "",
      };
    });

  const handCards: string[][] = getHandCardsFromUniversal(
    stateInfo.universal,
    numPlayers,
  );
  players.forEach((player: PlayerState, index: number) => {
    if (event.hideHoleCards) {
      player.cards = [];
    } else {
      player.cards = handCards[index] || [];
    }
  });

  const displayCommunity: string[] = event.hideCommunity ? [] : communityCards;

  return {
    players,
    communityCards: displayCommunity,
    pot: contributions.reduce(
      (sum: number, value: number | null) => sum + (value || 0),
      0,
    ),
    isTerminal: false,
    rawObservation: stateInfo.universal,
    step,
    winOdds: parsedStateHistory.winOdds,
    fiveCardBestHands: [],
    currentPlayer: stateInfo.universal?.current_player ?? -1,
    winner: -1,
  };
};
