// Mancala replay transformer.
//
// Populates `players[]` so the side-panel can show each mover's thoughts
// and pit choice, and pre-parses the observation JSON into `boardState`.
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
import type { MancalaObservation } from '../types';

interface MancalaPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  forfeited: boolean;
  forfeitLastAttempt: string | null;
}

export interface MancalaStep {
  step: number;
  players: MancalaPlayer[];
  boardState: MancalaObservation | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): MancalaObservation | null {
  const raw = step?.[0]?.observation?.observationString;
  if (typeof raw !== 'string') return null;
  try {
    return JSON.parse(raw) as MancalaObservation;
  } catch {
    return null;
  }
}

const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const mancalaTransformer = (environment: any): MancalaStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): MancalaStep => {
    const forfeit = detectForfeit(step);

    const players: MancalaPlayer[] = step.map((p, i): MancalaPlayer => {
      const isForfeiter = forfeit?.index === i;
      return {
        id: i,
        name: teamNames[i] || `Player ${i + 1}`,
        thumbnail: '',
        // Forfeiter submits -1 but should still be treated as acting so
        // their thoughts / last attempt render in the sidebar.
        isTurn: isRealMove(p.action?.submission) || isForfeiter,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    const boardState = parseBoardState(step);
    const observationTerminal = !!step[0]?.observation?.isTerminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    return {
      step: index,
      players,
      boardState,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    };
  });
};
