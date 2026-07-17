// Python Ant Foraging replay transformer.
//
// Populates `players[]` so the sidebar shows each mover's thoughts, and
// pre-parses the observation JSON into `boardState`.
//
// Ant foraging is a cooperative multi-agent game — all ants share the same
// score, so a reward-derived "winner" isn't meaningful. Terminal state comes
// from `boardState.is_terminal`. We still detect forfeits so the per-seat
// `forfeited` flag is populated for the sidebar, but skip winner/forfeitReason
// at the step level.

import { detectForfeit, parseThoughts, OpenSpielRawPlayer } from '@kaggle-environments/core';

interface AntPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  // Preserved so the renderer's `getLastAction` can look up the semantic
  // name via boardState.action_names[submission] like before.
  submission: number | null;
  forfeited: boolean;
  forfeitLastAttempt: string | null;
}

export interface AntBoardState {
  grid: string[][];
  grid_size: number;
  num_ants: number;
  num_food: number;
  nest_position: [number, number];
  food_positions: [number, number][];
  ant_positions: [number, number][];
  carrying_food: boolean[];
  pheromone_to_food: number[][];
  pheromone_to_nest: number[][];
  food_collected: number;
  score: number;
  turn: number;
  max_turns: number;
  current_player: number;
  legal_actions: number[];
  action_names: Record<string, string>;
  is_terminal: boolean;
}

export interface AntStep {
  step: number;
  players: AntPlayer[];
  boardState: AntBoardState | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): AntBoardState | null {
  const raw = step?.[0]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AntBoardState;
  } catch {
    return null;
  }
}

const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const pythonAntForagingTransformer = (environment: any): AntStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): AntStep => {
    const forfeit = detectForfeit(step);

    const players: AntPlayer[] = step.map((p, i): AntPlayer => {
      const sub = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender submits -1 but should still be treated as
      // "acting" so their thoughts / last attempt render in the sidebar.
      const isTurn = isRealMove(sub) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] || `Ant ${i}`,
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        submission: typeof sub === 'number' ? sub : null,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    return {
      step: index,
      players,
      boardState: parseBoardState(step),
    };
  });
};
