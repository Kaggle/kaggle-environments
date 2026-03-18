import { ConnectFourReplay, ConnectFourPlayer, ConnectFourStep, ConnectFourBoardState } from './connectFourReplayTypes';

function parseBoardState(stateHistoryEntry?: string): ConnectFourBoardState {
  if (!stateHistoryEntry || typeof stateHistoryEntry !== 'string') {
    return {
      board: [],
      currentPlayer: '',
      isTerminal: false,
      winner: null,
    };
  }

  try {
    const parsed = JSON.parse(stateHistoryEntry);
    return {
      board: parsed.board || [],
      currentPlayer: parsed.current_player || '',
      isTerminal: parsed.is_terminal || false,
      winner: parsed.winner || null,
    };
  } catch {
    return {
      board: [],
      currentPlayer: '',
      isTerminal: false,
      winner: null,
    };
  }
}

export function getConnectFourStepLabel(step: ConnectFourStep) {
  if (step.step === 0 || step.isTerminal) {
    return '';
  }

  return step.players.find((player) => player.isTurn)?.actionDisplayText ?? '';
}

export function getConnectFourStepDescription(step: ConnectFourStep) {
  if (step.isTerminal) {
    return step.winner ?? '';
  } else if (step.step === 0) {
    return 'Game Begins';
  }

  return step.players.find((player) => player.isTurn)?.thoughts ?? '';
}

export function deriveWinnerFromRewards(players: ConnectFourPlayer[], teamNames: string[]) {
  if (players.length < 2) return '';

  const player0Reward = players[0].reward;
  const player1Reward = players[1].reward;

  if (player0Reward === player1Reward) {
    return 'Draw';
  }

  const winnerPlayerIndex = player0Reward === 1 ? 0 : 1;
  const piece = winnerPlayerIndex === 0 ? 'X' : 'O';
  const name = teamNames[winnerPlayerIndex] || `Player ${winnerPlayerIndex + 1}`;

  return `${piece} (${name}) Wins!`;
}

export const connectFourTransformer = (environment: any) => {
  const connectFourReplay = environment as ConnectFourReplay;
  const agents = environment.info?.TeamNames || [];
  const stateHistory = environment.info?.stateHistory || [];

  const connectFourSteps: ConnectFourStep[] = [];

  // Add initial step with empty board
  connectFourSteps.push({
    step: 0,
    players: [],
    boardState: parseBoardState(stateHistory[0]), // Initial empty board
    isTerminal: false,
    winner: null,
  });

  // Track actual moves to properly index into stateHistory
  // stateHistory[0] = initial board, stateHistory[n] = board after move n
  let moveCount = 0;

  connectFourReplay.steps.forEach((step) => {
    // Check if this step contains an actual move (submission !== -1)
    const hasMove = step.some((player) => player.action?.submission !== undefined && player.action?.submission !== -1);

    // Skip setup steps that don't have any actual moves
    if (!hasMove) {
      return;
    }

    moveCount++;

    // Each step contains a tuple of players, one who acted and one who's waiting
    const stepPlayers: ConnectFourPlayer[] = step.map((player, playerIndex): ConnectFourPlayer => {
      return {
        id: playerIndex,
        name: agents[playerIndex] || `Player ${playerIndex + 1}`,
        thumbnail: '',
        isTurn: player.action?.submission !== undefined && player.action?.submission !== -1,
        actionDisplayText: player.action?.actionString ?? '',
        thoughts: player.action?.thoughts ?? '',
        reward: player.reward,
      };
    });

    // Get board state from stateHistory using moveCount
    // This ensures the board reflects the state AFTER this step's action
    const boardState = parseBoardState(stateHistory[moveCount]);

    connectFourSteps.push({
      step: moveCount, // 1-indexed since step 0 is initial board
      players: stepPlayers,
      boardState,
      isTerminal: false,
      winner: null,
    });
  });

  // Artificially insert a step at the end to emphasize the win state
  connectFourSteps.push({
    step: moveCount,
    players: [],
    boardState: connectFourSteps[moveCount].boardState,
    isTerminal: true,
    winner: deriveWinnerFromRewards(connectFourSteps[moveCount].players, agents),
  });

  return connectFourSteps;
};
