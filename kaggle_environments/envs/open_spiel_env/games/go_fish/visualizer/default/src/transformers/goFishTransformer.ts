// Go Fish replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs so the right-hand
// Game Log can render each agent's action label and thoughts. The renderer
// itself still consumes the raw step data via the observation JSON.
//
// In OpenSpiel go_fish every *player* action is an "Ask" (dealing and drawing
// from the pool are chance nodes handled internally). An ask action is encoded
// as `target * ranks + rank`, so we decode it into a human-readable label like
// "Ask Player 2 for K".
//
// Forfeit handling (illegal-move / TIMEOUT / ERROR) is delegated to the shared
// helpers in @kaggle-environments/core so that every OpenSpiel game labels
// early terminations the same way.

import {
  detectForfeit,
  buildForfeitReason,
  deriveWinnerFromRewards,
  parseThoughts,
  OpenSpielRawPlayer,
} from '@kaggle-environments/core';

interface GoFishPlayer {
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

export interface GoFishStep {
  step: number;
  players: GoFishPlayer[];
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
  rawStep: OpenSpielRawPlayer[];
}

const STANDARD_RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];

function rankLabel(rankIndex: number, numRanks: number): string {
  if (numRanks === 13 && rankIndex >= 0 && rankIndex < 13) return STANDARD_RANKS[rankIndex];
  return String.fromCharCode('a'.charCodeAt(0) + rankIndex);
}

function actionLabel(submission: number | undefined | null, numRanks: number): string {
  if (submission === undefined || submission === null || submission < 0) return '';
  const target = Math.floor(submission / numRanks);
  const rank = submission % numRanks;
  return `Ask P${target + 1} for ${rankLabel(rank, numRanks)}`;
}

export const goFishTransformer = (environment: any): GoFishStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const numRanks: number = environment?.configuration?.openSpielGameParameters?.ranks ?? 13;
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: GoFishStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: GoFishPlayer[] = step.map((p, i): GoFishPlayer => {
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
        // Prefer our decoded label ("Ask P2 for K") over OpenSpiel's cryptic
        // raw action string ("1c"). Fall back to actionString for forfeits,
        // where submission is -1 and holds the last illegal attempt text.
        actionDisplayText: actionLabel(submission, numRanks) || (p.action?.actionString ?? ''),
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

    // Go Fish has chance / dealing steps where no player acts; the renderer
    // consumes every step (via rawStep) so we don't filter here.
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
