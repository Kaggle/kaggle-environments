// Quoridor replay transformer.
//
// Quoridor is a sequential game: each step has exactly one acting player and
// the other submits -1. We use the proxy's JSON observation as the source of
// truth for board state, pawns, walls, and walls-remaining counts, and pair
// each step with the actor + parsed move so the renderer can highlight the
// most recent action.
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

interface QuoridorPlayer {
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

export type QuoridorPlayerCode = 'x' | 'o';

export type QuoridorMoveKind = 'pawn' | 'wall_v' | 'wall_h';

export interface QuoridorMove {
  kind: QuoridorMoveKind;
  col: number; // 0-indexed column
  row: number; // 0-indexed row
  raw: string; // original label, e.g. "e8", "a1v"
}

export interface QuoridorBoardState {
  board_size: number;
  num_players: number;
  cells: (number | null)[][];
  pawns: Record<string, string>;
  vertical_walls: string[];
  horizontal_walls: string[];
  walls_remaining: Record<string, number>;
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  move_number: number;
}

export interface QuoridorStep {
  step: number;
  players: QuoridorPlayer[];
  boardState: QuoridorBoardState | null;
  lastMove: QuoridorMove | null;
  lastActor: QuoridorPlayerCode | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): QuoridorBoardState | null {
  // Quoridor has perfect information so both players see the same observation;
  // try player 0 first, fall back to player 1 in case the active player slot
  // is the only one populated on a particular step.
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as QuoridorBoardState;
  } catch {
    return null;
  }
}

export function parseMove(label: string | null | undefined): QuoridorMove | null {
  if (!label) return null;
  const match = /^([a-y])(\d{1,2})([vh])?$/i.exec(label.trim());
  if (!match) return null;
  const col = match[1].toLowerCase().charCodeAt(0) - 'a'.charCodeAt(0);
  const row = parseInt(match[2], 10) - 1;
  const suffix = match[3]?.toLowerCase();
  if (col < 0 || row < 0) return null;
  return {
    kind: suffix === 'v' ? 'wall_v' : suffix === 'h' ? 'wall_h' : 'pawn',
    col,
    row,
    raw: label.toLowerCase(),
  };
}

export const quoridorTransformer = (environment: any): QuoridorStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player X', 'Player O'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: QuoridorStep[] = [];

  rawSteps.forEach((step) => {
    const forfeit = detectForfeit(step);

    const players: QuoridorPlayer[] = step.map((p, i): QuoridorPlayer => {
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

    const actingIdx = players.findIndex((pl) => pl.isTurn);
    const actor: QuoridorPlayerCode | null = actingIdx === 0 ? 'x' : actingIdx === 1 ? 'o' : null;
    const lastMove = parseMove(players[actingIdx]?.actionDisplayText ?? null);

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    out.push({
      // Renumber sequentially so the slider position matches `steps[step]`
      // after filtering out the env's setup step.
      step: out.length,
      players,
      boardState: parseBoardState(step),
      lastMove,
      lastActor: actor,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    });
  });

  return out;
};
