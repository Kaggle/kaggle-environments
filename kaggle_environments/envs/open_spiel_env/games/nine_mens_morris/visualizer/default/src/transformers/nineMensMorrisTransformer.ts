// Nine Men's Morris replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs to display
// model thoughts / action labels, and pre-parses the observation JSON into
// a `boardState` the renderer can consume directly.
//
// Nine Men's Morris is sequential: only the active player produces a real
// action in a step; the inactive player submits -1. The env also emits two
// setup steps at the start with both submissions == -1, which we drop.
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

interface NineMensMorrisPlayer {
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

export interface NineMensMorrisBoardState {
  board: string[]; // length 24, each 'W' | 'B' | '.'
  current_player: string;
  phase: string;
  men_to_deploy: Record<string, number>;
  num_men: Record<string, number>;
  turn_number: number;
  is_terminal: boolean;
  winner: string | null;
  last_action: string | null;
}

export interface NineMensMorrisStep {
  step: number;
  players: NineMensMorrisPlayer[];
  boardState: NineMensMorrisBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): NineMensMorrisBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as NineMensMorrisBoardState;
  } catch {
    return null;
  }
}

export const nineMensMorrisTransformer = (environment: any): NineMensMorrisStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player W', 'Player B'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: NineMensMorrisStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: NineMensMorrisPlayer[] = step.map((p, i): NineMensMorrisPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender has submission === -1 but should still be
      // treated as "acting" so the step is retained and their thoughts /
      // last-attempt render in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Player W' : 'Player B'),
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

    // Skip the env's setup steps where both players submit -1 and neither
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
