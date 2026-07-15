// Mancala replay transformer.
//
// Populates `players[]` so the side-panel can show each mover's thoughts
// and pit choice, and pre-parses the observation JSON into `boardState`.

import type { MancalaObservation } from '../types';

interface MancalaAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  generate_returns?: string[] | null;
}

interface MancalaReplayPlayer {
  action?: MancalaAction;
  reward: number;
  observation: { observationString?: string };
  status?: string;
}

interface MancalaPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
}

export interface MancalaStep {
  step: number;
  players: MancalaPlayer[];
  boardState: MancalaObservation | null;
}

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
function parseThoughts(action?: MancalaAction): string {
  if (action?.thoughts) return action.thoughts;
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) return parsed.main_response_and_thoughts;
    } catch {
      // fall through
    }
  }
  return '';
}

function parseBoardState(step: MancalaReplayPlayer[]): MancalaObservation | null {
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
  const rawSteps: MancalaReplayPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): MancalaStep => {
    const players: MancalaPlayer[] = step.map(
      (p, i): MancalaPlayer => ({
        id: i,
        name: teamNames[i] || `Player ${i + 1}`,
        thumbnail: '',
        isTurn: isRealMove(p.action?.submission),
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
      })
    );

    return {
      step: index,
      players,
      boardState: parseBoardState(step),
    };
  });
};
