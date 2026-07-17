// Bargaining replay transformer.
//
// Private-information game: each seat's observationString is a per-seat JSON
// view (contains that seat's `my_values`), and `is_terminal` lives inside
// that per-seat obs rather than on `observation.isTerminal`.
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
import { BargainingObs, BargainingPlayer, BargainingStep } from './bargainingReplayTypes';

function parseObs(raw?: string): BargainingObs | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as BargainingObs;
  } catch {
    return null;
  }
}

// Submission is -1 (or null) on setup/inactive turns; treat anything else as a real move.
const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const bargainingTransformer = (environment: any): BargainingStep[] => {
  const agents: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): BargainingStep => {
    const forfeit = detectForfeit(step);

    const obs0 = parseObs(step?.[0]?.observation?.observationString);
    const obs1 = parseObs(step?.[1]?.observation?.observationString);
    const obs = obs0 ?? obs1;

    const players: BargainingPlayer[] = step.map((p, pi): BargainingPlayer => {
      const isForfeiter = forfeit?.index === pi;
      return {
        id: pi,
        name: agents[pi] || `Player ${pi + 1}`,
        thumbnail: '',
        // Forfeiter submits -1 but should still be treated as acting so
        // their thoughts / last attempt render in the sidebar.
        isTurn: isRealMove(p.action?.submission) || isForfeiter,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? null,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    // OpenSpiel's is_terminal lives per-seat inside the observation JSON
    // here; either seat's flag is authoritative when set. A forfeit ends
    // the episode even if OpenSpiel's own state isn't terminal.
    const observationTerminal = !!obs?.is_terminal;
    const isTerminal = observationTerminal || forfeit !== null;

    return {
      step: index,
      players,
      observations: [obs0, obs1],
      obs,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, agents) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, agents) : null,
    };
  });
};
