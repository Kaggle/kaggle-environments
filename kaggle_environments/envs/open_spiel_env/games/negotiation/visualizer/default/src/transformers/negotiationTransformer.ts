// Negotiation replay transformer.
//
// Populates `players[]` so the sidebar can show each mover's thoughts and
// their proposal/utterance action. Negotiation is a private-information
// game — each seat's observation shows different utilities, so `boardState`
// keeps both players' pre-parsed observations side by side.

import {
  detectForfeit,
  buildForfeitReason,
  deriveWinnerFromRewards,
  parseThoughts,
  OpenSpielRawPlayer,
} from '@kaggle-environments/core';

interface NegotiationPlayer {
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

export interface NegotiationObs {
  current_player: number;
  viewing_player: number;
  turn_type: 'proposal' | 'utterance' | null;
  max_steps: number;
  item_pool: number[];
  my_utilities: number[];
  proposals: Array<{ player: number; items?: number[]; accept: boolean }>;
  utterances: Array<{ player: number; symbols: number[] }>;
  most_recent_proposal: number[] | null;
  most_recent_utterance: number[] | null;
  agreement_reached: boolean;
  is_terminal: boolean;
  winner: number | 'draw' | null;
  rewards: number[] | null;
  params: {
    num_items: number;
    num_symbols: number;
    utterance_dim: number;
    enable_utterances: boolean;
  };
}

export interface NegotiationStep {
  step: number;
  players: NegotiationPlayer[];
  // Per-seat observations — both are needed because utilities are private.
  boardState: {
    perPlayer: [NegotiationObs | null, NegotiationObs | null];
  } | null;
  isTerminal: boolean;
  winner: string | null;
  forfeitReason: string | null;
}

function parseObs(raw?: string): NegotiationObs | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as NegotiationObs;
  } catch {
    return null;
  }
}

const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const negotiationTransformer = (environment: any): NegotiationStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): NegotiationStep => {
    const forfeit = detectForfeit(step);

    const players: NegotiationPlayer[] = step.map((p, i): NegotiationPlayer => {
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

    const obs0 = parseObs(step?.[0]?.observation?.observationString);
    const obs1 = parseObs(step?.[1]?.observation?.observationString);

    // OpenSpiel's is_terminal lives per-seat inside the observation JSON
    // here; either seat's flag is authoritative when set.
    const observationTerminal = !!(obs0?.is_terminal || obs1?.is_terminal);
    const isTerminal = observationTerminal || forfeit !== null;

    return {
      step: index,
      players,
      boardState: obs0 || obs1 ? { perPlayer: [obs0, obs1] } : null,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    };
  });
};
