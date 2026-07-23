// Shogi replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs and parses the
// proxy's JSON observation into a typed board state. Shogi is sequential:
// only the active player submits a real action; the other side submits -1.
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

interface ShogiPlayer {
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

// Each cell is either '.', a single piece letter (uppercase = Sente / Black,
// lowercase = Gote / White), or a 2-char string starting with '+' marking a
// promoted piece (e.g. '+P', '+r').
export type ShogiCell = string;

export interface ShogiHandCounts {
  [piece: string]: number;
}

export interface ShogiBoardState {
  board: ShogiCell[][];
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  captured: { b: ShogiHandCounts; w: ShogiHandCounts };
  move_number: number;
  last_move: string | null;
  move_history: string[];
  sfen: string;
}

export interface ShogiStep {
  step: number;
  players: ShogiPlayer[];
  boardState: ShogiBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): ShogiBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as ShogiBoardState;
  } catch {
    return null;
  }
}

export const shogiTransformer = (environment: any): ShogiStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: ShogiStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: ShogiPlayer[] = step.map((p, i): ShogiPlayer => {
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

    // Keep the initial step (index 0) so the starting board renders; skip
    // any later "both submit -1" steps that aren't a forfeit.
    if (!players.some((pl) => pl.isTurn) && index > 0) return;

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
