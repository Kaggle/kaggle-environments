/* eslint-disable @typescript-eslint/no-explicit-any */
import { ChessReplay, ChessPlayer, ChessStep, FenState, ChessReplayStep } from './chessReplayTypes';

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

function deriveWinner(step: ChessReplayStep[]): string | null {
  if (step[0].observation.isTerminal === false) return null;
  if (step[0].reward === step[1].reward) return null;
  return step[0].reward === 1 ? 'white' : 'black';
}

export const chessTransformer = (environment: any): ChessStep[] => {
  const chessReplay = environment as ChessReplay;
  const chessSteps: ChessStep[] = [];

  const extraStepPlayers = [0, 1].map(
    (index): ChessPlayer => ({
      id: index,
      name: environment.info.TeamNames[index],
      thumbnail: '',
      isTurn: false,
      actionDisplayText: '',
      thoughts: '',
      reward: null,
      generateReturns: null,
    })
  );

  chessSteps.push({
    step: chessSteps.length,
    players: extraStepPlayers,
    fenState: parseFen(''),
    isTerminal: false,
    winner: null,
  });

  for (const step of chessReplay.steps) {
    // Each step contains a tuple of players, one who acted and one who's waiting
    const stepPlayers: ChessPlayer[] = step.map((player, index): ChessPlayer => {
      return {
        id: index,
        name: environment.info.TeamNames[index],
        thumbnail: '',
        isTurn: player.action?.submission !== -1,
        actionDisplayText: player.action?.actionString ?? '',
        thoughts: player.action?.thoughts ?? '',
        reward: player.reward,
        generateReturns: player.action?.generate_returns ?? null,
      };
    });

    // Ignore setup steps where no one acted
    if (stepPlayers.findIndex((player) => player.isTurn) !== -1) {
      chessSteps.push({
        step: chessSteps.length,
        players: stepPlayers,
        // Both agents have the same observation string for the step, just grab the first one
        fenState: parseFen(step[0].observation.observationString),
        isTerminal: false,
        winner: '',
      });
    }
  }

  const lastReplayStep = chessReplay.steps[chessReplay.steps.length - 1];

  chessSteps.push({
    players: extraStepPlayers,
    isTerminal: true,
    fenState: chessSteps[chessSteps.length - 1].fenState,
    step: chessSteps.length,
    winner: deriveWinner(lastReplayStep),
  });

  return chessSteps;
};
