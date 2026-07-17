// Lines of Action replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs so the
// right-hand Game Log can render each agent's reasoning. Mirrors the
// pattern used by the dark_hex / amazons / Y transformers.
//
// Forfeit handling (illegal-move / TIMEOUT / ERROR) is delegated to the
// shared helpers in @kaggle-environments/core so that every OpenSpiel game
// labels early terminations the same way.

import {
  detectForfeit,
  buildForfeitReason,
  deriveWinnerFromRewards,
  parseThoughts,
  OpenSpielRawPlayer,
} from '@kaggle-environments/core';

interface LoaPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
  forfeited: boolean;
  forfeitLastAttempt: string | null;
}

export type LoaCell = '.' | 'x' | 'o';

export interface LoaBoardState {
  board: LoaCell[][];
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  move_number: number;
  last_move: string | null;
}

export interface LoaMove {
  fromRow: number;
  fromCol: number;
  toRow: number;
  toCol: number;
  capture: boolean;
}

export interface LoaStep {
  step: number;
  players: LoaPlayer[];
  boardState: LoaBoardState | null;
  lastMove: LoaMove | null;
  lastActor: number | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): LoaBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as LoaBoardState;
  } catch {
    return null;
  }
}

function parseMoveString(s: string | null | undefined): LoaMove | null {
  if (!s || s.length < 5) return null;
  // Format: "<col><row><x|->/<col><row>" e.g. "b1-h1" or "c3xa3".
  const fromCol = s.charCodeAt(0) - 'a'.charCodeAt(0);
  const fromRow = parseInt(s[1], 10) - 1;
  const sep = s[2];
  const toCol = s.charCodeAt(3) - 'a'.charCodeAt(0);
  const toRow = parseInt(s[4], 10) - 1;
  if (
    fromCol < 0 ||
    fromCol > 7 ||
    toCol < 0 ||
    toCol > 7 ||
    isNaN(fromRow) ||
    isNaN(toRow) ||
    (sep !== '-' && sep !== 'x')
  ) {
    return null;
  }
  return { fromRow, fromCol, toRow, toCol, capture: sep === 'x' };
}

export const loaTransformer = (environment: any): LoaStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player X', 'Player O'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: LoaStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: LoaPlayer[] = step.map((p, i): LoaPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender has submission === -1 but should still be
      // treated as "acting" so the step is retained and their thoughts /
      // last-attempt render in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Player X' : 'Player O'),
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    // Skip the env's setup step where both players submit -1 and neither
    // forfeited.
    if (!players.some((pl) => pl.isTurn)) return;

    const boardState = parseBoardState(step);
    const lastMove = parseMoveString(boardState?.last_move);
    // Move parity: black (x, player 0) moves on odd move_number, after which it's white's turn.
    const lastActor = boardState && boardState.move_number > 0 ? (boardState.move_number - 1) % 2 : null;
    const observationTerminal = !!step[0]?.observation?.isTerminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    out.push({
      step: index,
      players,
      boardState,
      lastMove,
      lastActor,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    });
  });

  return out;
};
