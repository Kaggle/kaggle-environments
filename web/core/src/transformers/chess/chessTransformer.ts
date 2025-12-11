import { ChessReplay, ChessPlayer, ChessStep, FenState } from './chessReplayTypes';

function parseFen(fen?: string): FenState {
  if (!fen || typeof fen !== 'string') {
    return {
      board: [],
      activeColor: '',
      castling: '',
      enPassant: '',
      halfmoveClock: '',
      fullmoveNumber: '',
    };
  }

  const [piecePlacement, activeColor, castling, enPassant, halfmoveClock, fullmoveNumber] = fen.split(' ');

  // Within the context of the replay, the active color is the color of the player that just completed their move.
  // This is the opposite of the active color in the fen string, which is the color of the player that is about to move.
  // Therefore, we need to invert the "active" color.
  const playerColor = String(activeColor).toLowerCase() === 'w' ? 'Black' : 'White';

  const board = [];
  const rows = piecePlacement.split('/');

  for (const row of rows) {
    const boardRow = [];
    for (const char of row) {
      if (isNaN(parseInt(char))) {
        boardRow.push(char);
      } else {
        for (let i = 0; i < parseInt(char); i++) {
          boardRow.push(null);
        }
      }
    }
    board.push(boardRow);
  }

  return {
    board,
    activeColor: playerColor,
    castling,
    enPassant,
    halfmoveClock,
    fullmoveNumber,
  };
}

export function getChessStepLabel(step: ChessStep) {
  if (step.isTerminal) {
    return '';
  }

  return step.players.find((player) => player.isTurn)?.actionDisplayText ?? '';
}

export function getChessStepDescription(step: ChessStep) {
  if (step.isTerminal) {
    return step.winner ?? '';
  }

  return step.players.find((player) => player.isTurn)?.thoughts ?? '';
}

export function deriveWinnerFromRewards(players: ChessPlayer[]) {
  if (players.length < 2) return '';

  const player0Reward = players[0].reward;
  const player1Reward = players[1].reward;

  if (player0Reward === player1Reward) {
    return 'Draw';
  }

  const winnerPlayerIndex = player0Reward === 1 ? 0 : 1;
  const color = winnerPlayerIndex === 0 ? 'Black' : 'White';

  return `🎉 ${color} (${players[winnerPlayerIndex].name}) Wins!`;
}

export const chessTransformer = (environment: any) => {
  const chessReplay = environment as ChessReplay;
  const agents = environment.info.TeamNames;

  const chessSteps: ChessStep[] = [];

  chessReplay.steps.forEach((step, index) => {
    // Each step contains a tuple of players, one who acted and one who's waiting
    const stepPlayers: ChessPlayer[] = step.map((player, index): ChessPlayer => {
      return {
        id: index,
        name: agents[index],
        thumbnail: '',
        isTurn: player.action?.submission !== -1,
        actionDisplayText: player.action?.actionString ?? '',
        thoughts: player.action?.thoughts ?? '',
        reward: player.reward,
      };
    });

    // Ignore setup steps where no one acted
    if (stepPlayers.findIndex((player) => player.isTurn) !== -1) {
      chessSteps.push({
        step: index,
        players: stepPlayers,
        // Both agents have the same observation string for the step, just grab the first one
        fenState: parseFen(step[0].observation.observationString),
        isTerminal: false,
        winner: '',
      });
    }
  });

  const lastStep = chessSteps[chessSteps.length - 1];
  const winDescription = deriveWinnerFromRewards(lastStep.players);

  // Artificially insert a step at the end to emphasize the win state
  chessSteps.push({
    players: [
      {
        id: -1,
        name: 'System',
        thumbnail: '',
        isTurn: false,
        actionDisplayText: '',
        thoughts: '',
        reward: 0,
      },
    ],
    isTerminal: true,
    fenState: lastStep.fenState,
    step: lastStep.step + 1,
    winner: winDescription,
  });

  return chessSteps;
};
