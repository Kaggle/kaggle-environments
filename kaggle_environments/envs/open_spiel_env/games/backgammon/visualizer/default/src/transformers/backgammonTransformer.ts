// Backgammon replay transformer.
//
// Converts raw OpenSpiel step arrays into the structured form the side-panel
// UI expects (one `players` entry per seat, plus a parsed `boardState`).
// Backgammon is sequential: only the active player submits a real action;
// the inactive player sends -1.
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

interface BackgammonPlayer {
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

export interface BackgammonPoint {
  player: 'x' | 'o';
  count: number;
}

export interface BackgammonDie {
  value: number;
  used: boolean;
}

export interface BackgammonBoardState {
  board: (BackgammonPoint | null)[];
  bar: { x: number; o: number };
  off: { x: number; o: number };
  dice: BackgammonDie[];
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  move_number: number;
}

export interface BackgammonStep {
  step: number;
  players: BackgammonPlayer[];
  boardState: BackgammonBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash).
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): BackgammonBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as BackgammonBoardState;
  } catch {
    return null;
  }
}

export const backgammonTransformer = (environment: any): BackgammonStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: BackgammonStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: BackgammonPlayer[] = step.map((p, i): BackgammonPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender has submission === -1 but should still be
      // treated as "acting" so the step is retained and their thoughts /
      // last-attempt render in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Player 1' : 'Player 2'),
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

    // Skip the env's setup step where both players submit -1.
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
