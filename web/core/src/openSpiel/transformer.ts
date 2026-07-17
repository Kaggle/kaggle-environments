// Generic transformer helpers for OpenSpiel game visualizers -- pulling
// model thoughts out of harness output, deriving a winner from reward
// signs on a terminal step, etc.

import { OpenSpielRawAction, OpenSpielRawPlayer } from './types';

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
export function parseThoughts(action?: OpenSpielRawAction): string {
  if (action?.thoughts) return action.thoughts;
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      // fall through
    }
  }
  return '';
}

// Reward-sign winner derivation. Works for any 2-player terminal, and is
// the fallback path when a forfeit ends the game before OpenSpiel's own
// winner field is set. Returns "Draw" on equal rewards.
export function deriveWinnerFromRewards(
  step: OpenSpielRawPlayer[],
  teamNames: string[],
  winnerSuffix = 'wins!'
): string | null {
  if (!Array.isArray(step) || step.length < 2) return null;
  const r0 = step[0].reward ?? 0;
  const r1 = step[1].reward ?? 0;
  if (r0 === r1) return 'Draw';
  const winnerName = r0 > r1 ? teamNames[0] : teamNames[1];
  return `${winnerName} ${winnerSuffix}`;
}
