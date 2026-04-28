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

export function deriveWinnerFromRewards(rewards: (number | null | undefined)[], teamNames: string[]) {
  if (rewards.length < 2) return '';

  const player0Reward = rewards[0] ?? null;
  const player1Reward = rewards[1] ?? null;

  if (player0Reward === player1Reward) {
    return 'Draw';
  }

  // Higher reward wins (handles null vs -1, 1 vs -1, etc.)
  const r0 = player0Reward ?? -Infinity;
  const r1 = player1Reward ?? -Infinity;
  const winnerPlayerIndex = r0 > r1 ? 0 : 1;
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
    forfeitReason: null,
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
      forfeitReason: null,
    });
  });

  // The original replay's last step holds the terminal rewards (e.g. [1, -1])
  // even when the losing agent's response was cut off and never produced a
  // valid move. Using these instead of the last in-game step's rewards
  // (which are typically null mid-play) ensures we still declare a winner
  // when an agent errors out instead of falling back to "Draw".
  const lastReplayStep = connectFourReplay.steps[connectFourReplay.steps.length - 1] ?? [];
  const terminalRewards = lastReplayStep.map((p) => p.reward);
  const finalBoardState = { ...connectFourSteps[moveCount].boardState };
  if (!finalBoardState.winner && terminalRewards.length >= 2) {
    const r0 = terminalRewards[0] ?? null;
    const r1 = terminalRewards[1] ?? null;
    if (r0 !== r1) {
      finalBoardState.winner = (r0 ?? -Infinity) > (r1 ?? -Infinity) ? 'x' : 'o';
    }
  }

  // If rewards indicate a winner but the on-board game state never reached
  // a real "in-a-row" win, the loser must have forfeited (e.g. response cut
  // off / unparsable submission). Surface that explicitly so the UI doesn't
  // imply a normal win.
  let forfeitReason: string | null = null;
  if (terminalRewards.length >= 2) {
    const r0 = terminalRewards[0] ?? null;
    const r1 = terminalRewards[1] ?? null;
    const gameWonOnBoard =
      connectFourSteps[moveCount].boardState.winner === 'x' || connectFourSteps[moveCount].boardState.winner === 'o';
    if (r0 !== r1 && !gameWonOnBoard) {
      const loserIndex = (r0 ?? -Infinity) < (r1 ?? -Infinity) ? 0 : 1;
      const winnerIndex = 1 - loserIndex;
      const loserName = agents[loserIndex] || `Player ${loserIndex + 1}`;
      const winnerName = agents[winnerIndex] || `Player ${winnerIndex + 1}`;
      forfeitReason = `${loserName} failed to produce valid input. ${winnerName} wins by default.`;
    }
  }

  // Artificially insert a step at the end to emphasize the win state
  connectFourSteps.push({
    step: moveCount,
    players: [],
    boardState: finalBoardState,
    isTerminal: true,
    winner: deriveWinnerFromRewards(terminalRewards, agents),
    forfeitReason,
  });

  return connectFourSteps;
};
