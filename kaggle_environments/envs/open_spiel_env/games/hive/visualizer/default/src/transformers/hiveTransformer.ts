// Hive replay transformer.
//
// Builds the per-step `players` array that the side-panel UI needs and parses
// the proxy's JSON observation into a typed board state.
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

interface HivePlayer {
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

export type HivePieces = Record<string, [number, number, number]>;

export interface HiveBoardState {
  game_type: string;
  expansions?: { mosquito?: boolean; ladybug?: boolean; pillbug?: boolean };
  board_radius?: number;
  status: string;
  turn?: string;
  current_player: string;
  move_number: number;
  moves: string[];
  last_move: string | null;
  pieces: HivePieces;
  is_terminal: boolean;
  winner: string | null;
  uhp?: string;
}

export interface HiveStep {
  step: number;
  players: HivePlayer[];
  boardState: HiveBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): HiveBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as HiveBoardState;
  } catch {
    return null;
  }
}

export const hiveTransformer = (environment: any): HiveStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['White', 'Black'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: HiveStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: HivePlayer[] = step.map((p, i): HivePlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender has submission === -1 but should still be
      // treated as "acting" so the step is retained and their thoughts /
      // last-attempt render in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'White' : 'Black'),
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

    // Skip the env's setup step where both players submit -1 and neither
    // forfeited.
    if (!players.some((pl) => pl.isTurn)) return;

    const boardState = parseBoardState(step);
    const observationTerminal = !!step[0]?.observation?.isTerminal || !!boardState?.is_terminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    out.push({
      step: index,
      players,
      boardState,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    });
  });

  return out;
};
