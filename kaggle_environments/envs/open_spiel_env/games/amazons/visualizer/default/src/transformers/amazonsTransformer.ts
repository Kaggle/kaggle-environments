// Amazons replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs. Each Amazons
// turn is three sub-actions (from / to / shoot), and only the active player
// produces an action in any given step — the inactive player just waits.
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

interface AmazonsPlayer {
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

export type AmazonsCell = 'X' | 'O' | '#' | '.';

export interface AmazonsBoardState {
  board: AmazonsCell[][];
  num_rows?: number;
  num_cols?: number;
  current_player: string;
  phase: 'from' | 'to' | 'shoot' | null;
  move_number: number;
  is_terminal: boolean;
  winner: string | null;
}

export interface AmazonsStep {
  step: number;
  players: AmazonsPlayer[];
  boardState: AmazonsBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash).
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): AmazonsBoardState | null {
  // Both agents see the same observationString; pick whichever is populated.
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AmazonsBoardState;
  } catch {
    return null;
  }
}

export const amazonsTransformer = (environment: any): AmazonsStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Black (X)', 'White (O)'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: AmazonsStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: AmazonsPlayer[] = step.map((p, i): AmazonsPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender has submission === -1 but should still be
      // treated as "acting" so the step is retained and their thoughts /
      // last-attempt render in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Black (X)' : 'White (O)'),
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

    // Skip steps where neither player acted (e.g. the env's setup step where
    // both submit -1). The side-panel uses isTurn to pick the active player.
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
