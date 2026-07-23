// Bridge Arena replay transformer.
//
// Populates `players[]` (4 seats) so the side-panel sidebar can show each
// mover's thoughts and call/play. Pre-parses the shared observation JSON
// (whichever player has it populated) into `boardState`.
//
// Bridge is a 4-player team game (seats 0/1 vs seats 2/3), so the standard
// 2-player `buildForfeitReason` helper (which does `teamNames[1 - i]`) does
// not apply -- we build the forfeit reason in-place.

import { detectForfeit, FORFEIT_REASONS, parseThoughts, OpenSpielRawPlayer } from '@kaggle-environments/core';

interface BridgePlayer {
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

export interface BridgeBoardState {
  phase?: string;
  is_terminal?: boolean;
  dealer_table_position?: string;
  dealer_player_id?: number;
  current_player_id?: number | null;
  current_table_position?: string | null;
  current_team_id?: number | null;
  auction?: unknown[];
  plays?: unknown[];
  table_seating?: Record<string, string>;
  teams?: Record<string, number[]>;
  returns?: number[];
  team_totals?: number[];
  winning_team?: number | string;
}

export interface BridgeStep {
  step: number;
  players: BridgePlayer[];
  boardState: BridgeBoardState | null;
  isTerminal: boolean;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash).
  forfeitReason: string | null;
}

// Bridge observations are per-player (hidden information), but the renderer
// only uses the shared/public parts, and takes them from whichever seat has
// a non-empty observationString.
function parseBoardState(step: OpenSpielRawPlayer[]): BridgeBoardState | null {
  for (const p of step) {
    const raw = p?.observation?.observationString;
    if (!raw) continue;
    try {
      return JSON.parse(raw) as BridgeBoardState;
    } catch {
      // try the next seat
    }
  }
  return null;
}

const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

// Bridge is 2v2 (seats 0,1 = team A; seats 2,3 = team B), so the winning
// team is the opposing team of the forfeiter's seat.
function buildBridgeForfeitReason(forfeit: { index: number; reasonKey: string }, teamNames: string[]): string {
  const loser = teamNames[forfeit.index] ?? `Player ${forfeit.index}`;
  const loserTeamLabel = forfeit.index < 2 ? 'Team A' : 'Team B';
  const winnerTeamLabel = forfeit.index < 2 ? 'Team B' : 'Team A';
  const reason = FORFEIT_REASONS[forfeit.reasonKey] ?? 'forfeited';
  return `${loser} (${loserTeamLabel}) ${reason}. ${winnerTeamLabel} wins by default.`;
}

export const bridgeArenaTransformer = (environment: any): BridgeStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): BridgeStep => {
    const forfeit = detectForfeit(step);

    const players: BridgePlayer[] = step.map((p, i): BridgePlayer => {
      const isForfeiter = forfeit?.index === i;
      return {
        id: i,
        name: teamNames[i] || `Player ${i}`,
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
    // Bridge's `is_terminal` lives inside the per-seat obs JSON (like the
    // other private-information games). A forfeit ends the episode even if
    // OpenSpiel's own state isn't terminal.
    const observationTerminal = !!boardState?.is_terminal;
    const isTerminal = observationTerminal || forfeit !== null;

    return {
      step: index,
      players,
      boardState,
      isTerminal,
      forfeitReason: forfeit ? buildBridgeForfeitReason(forfeit, teamNames) : null,
    };
  });
};
