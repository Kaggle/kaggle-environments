// Capture the Flag replay transformer.
//
// CTF is a simultaneous-move game: both players submit an action every step,
// and the framework then resolves a chance node for move-order initiative.
// After every "real" step both players have submission != -1.

import {
  buildForfeitReason,
  detectForfeit,
  deriveWinnerFromRewards,
  OpenSpielRawPlayer,
  parseThoughts,
} from '@kaggle-environments/core';

export interface CtfPlayer {
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

export type CtfCell = '.' | '*' | 'A' | 'B' | 'a' | 'b';

export interface CtfBoardState {
  board: CtfCell[][];
  num_rows: number;
  num_cols: number;
  a_base: [number, number];
  b_base: [number, number];
  obstacles: [number, number][];
  a_pos: [number, number];
  b_pos: [number, number];
  flag_a_pos: [number, number];
  flag_b_pos: [number, number];
  carrier_a: number | null;
  carrier_b: number | null;
  score: [number, number];
  score_limit: number;
  horizon: number;
  move_number: number;
  current_player: string | number;
  action_names: string[];
  is_terminal: boolean;
  winner: number | 'draw' | null;
}

export interface CtfStep {
  step: number;
  players: CtfPlayer[];
  boardState: CtfBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): CtfBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as CtfBoardState;
  } catch {
    return null;
  }
}

export const captureTheFlagTransformer = (environment: any): CtfStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player A', 'Player B'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: CtfStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: CtfPlayer[] = step.map((p, i): CtfPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      const isTurn = (submission !== undefined && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? `Player ${String.fromCharCode(65 + i)}`,
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

    // Drop the env's setup step where both players submit -1.
    if (!players.some((pl) => pl.isTurn)) return;

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    const isTerminal = observationTerminal || forfeit !== null;

    out.push({
      step: index,
      players,
      boardState: parseBoardState(step),
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    });
  });

  return out;
};
