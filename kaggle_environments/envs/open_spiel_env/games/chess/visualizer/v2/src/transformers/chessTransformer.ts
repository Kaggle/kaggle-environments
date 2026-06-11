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

function deriveWinner(step: ChessReplayStep[]): string | null {
  if (step[0].reward === 1) return 'black';
  if (step[1].reward === 1) return 'white';
  return null;
}

function deriveStatus(step: ChessReplayStep[]): string | null {
  const forfeit = step.find((p) => ['TIMEOUT', 'ERROR', 'INVALID'].includes(p.status));
  return forfeit?.status || null;
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
    status: null,
  });

  for (const step of chessReplay.steps) {
    // Each step contains a tuple of players, one who acted and one who's waiting
    const stepPlayers: ChessPlayer[] = step.map((player, index): ChessPlayer => {
      const submission = player.action?.submission;
      return {
        id: index,
        name: environment.info.TeamNames[index],
        thumbnail: '',
        // A turn requires submission to be a real action id. -1 means the player
        // didn't act this step (inactive or forfeited); null/undefined shows up
        // in pre-game init steps where the harness pings agents without
        // requesting a move — we don't want those rendered as spurious turns.
        isTurn: typeof submission === 'number' && submission !== -1,
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
        status: null,
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
    status: deriveStatus(lastReplayStep),
  });

  return chessSteps;
};
