// Snake replay transformer.
//
// Populates `players[]` so the side-panel sidebar can render each mover's
// thoughts / action label, and pre-parses the observation JSON into
// `boardState` for the renderer.
//
// Snake supports 1-4 players and terminal-state / winner information comes
// from the observation JSON (`boardState.is_terminal` / `boardState.winner`),
// so we don't emit a reward-derived winner at the step level. We do emit a
// step-level `forfeitReason` so renderers can label the offender with the
// actual detected reason (TIMEOUT / INVALID / ERROR) rather than guessing.

import { detectForfeit, FORFEIT_REASONS, parseThoughts, OpenSpielRawPlayer } from '@kaggle-environments/core';

interface SnakePlayer {
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

export interface SnakeBoardState {
  board: string[][];
  num_rows: number;
  num_columns: number;
  num_players: number;
  foods?: [number, number][];
  food?: [number, number] | null;
  snakes: {
    player: number;
    body: [number, number][];
    alive: boolean;
    score: number;
  }[];
  scores: number[];
  is_alive: boolean[];
  current_player: number;
  pending_this_turn: number[];
  turn: number;
  is_terminal: boolean;
  winner: number | string | null;
  game_over_reason: string | null;
}

export interface SnakeStep {
  step: number;
  players: SnakePlayer[];
  boardState: SnakeBoardState | null;
  // Non-null when a player forfeited on this step. Cooperative game: no
  // "wins by default" clause -- just names the offender and the reason.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): SnakeBoardState | null {
  const raw = step?.[0]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as SnakeBoardState;
  } catch {
    return null;
  }
}

const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const snakeTransformer = (environment: any): SnakeStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): SnakeStep => {
    const forfeit = detectForfeit(step);

    const players: SnakePlayer[] = step.map((p, i): SnakePlayer => {
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender submits -1 but should still be treated as
      // "acting" so their thoughts / last attempt render in the sidebar.
      const isTurn = isRealMove(p.action?.submission) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] || `Player ${i}`,
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    let forfeitReason: string | null = null;
    if (forfeit) {
      const loser = teamNames[forfeit.index] || `Player ${forfeit.index}`;
      const reason = FORFEIT_REASONS[forfeit.reasonKey] ?? 'forfeited';
      forfeitReason = `${loser} ${reason}.`;
    }

    return {
      step: index,
      players,
      boardState: parseBoardState(step),
      forfeitReason,
    };
  });
};
