// Gin Rummy replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs so the right-hand
// Game Log can render each agent's action label and thoughts. The renderer
// itself still consumes the raw step data via `mergedObservation()`.
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

interface GinRummyPlayer {
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

export interface GinRummyStep {
  step: number;
  players: GinRummyPlayer[];
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
  rawStep: OpenSpielRawPlayer[];
}

const SUIT_GLYPH: Record<string, string> = { s: '♠', c: '♣', d: '♦', h: '♥' };

function actionLabel(submission: number | undefined | null): string {
  if (submission === undefined || submission === null || submission < 0) return '';
  if (submission === 52) return 'Draw upcard';
  if (submission === 53) return 'Draw stock';
  if (submission === 54) return 'Pass';
  if (submission === 55) return 'Knock';
  if (submission >= 56) return `Meld (action ${submission})`;
  // Single-card action 0-51: derive the card from OpenSpiel's canonical
  // ordering (rank-major within suit s, c, d, h).
  const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'];
  const suits = ['s', 'c', 'd', 'h'];
  const suit = suits[Math.floor(submission / 13)];
  const rank = ranks[submission % 13];
  return `${rank}${SUIT_GLYPH[suit] ?? suit}`;
}

export const ginRummyTransformer = (environment: any): GinRummyStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: GinRummyStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: GinRummyPlayer[] = step.map((p, i): GinRummyPlayer => {
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
        actionDisplayText: p.action?.actionString ?? actionLabel(submission),
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    // Gin Rummy has chance / dealing steps where no player acts; the
    // renderer consumes every step (via rawStep) so we don't filter here,
    // matching the pre-refactor behavior.
    out.push({
      step: index,
      players,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
      rawStep: step,
    });
  });

  return out;
};
