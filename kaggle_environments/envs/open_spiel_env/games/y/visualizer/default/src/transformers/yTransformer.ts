// Y replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs. Y is a
// sequential game: only the active player produces a real action in any
// given step; the inactive player just submits -1.
//
// We surface the LLM's `thoughts` (and `generate_returns`, when present, in
// the Go-harness shape) so the right-hand Game Log can render reasoning.
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

interface YPlayer {
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

export type YCell = 'x' | 'o' | null;

export interface YBoardState {
  board: YCell[][];
  board_size: number;
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  last_move: string | null;
  move_number: number;
}

export interface YStep {
  step: number;
  players: YPlayer[];
  boardState: YBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): YBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as YBoardState;
  } catch {
    return null;
  }
}

export const yTransformer = (environment: any): YStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player X', 'Player O'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: YStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: YPlayer[] = step.map((p, i): YPlayer => {
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

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
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
