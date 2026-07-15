// Python Ant Foraging replay transformer.
//
// Populates `players[]` so the sidebar shows each mover's thoughts, and
// pre-parses the observation JSON into `boardState`.

interface AntAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  generate_returns?: string[] | null;
}

interface AntReplayPlayer {
  action?: AntAction;
  reward: number;
  observation: { observationString?: string };
  status?: string;
}

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

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
function parseThoughts(action?: AntAction): string {
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

function parseBoardState(step: AntReplayPlayer[]): AntBoardState | null {
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
  const rawSteps: AntReplayPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): AntStep => {
    const players: AntPlayer[] = step.map((p, i): AntPlayer => {
      const sub = p.action?.submission;
      return {
        id: i,
        name: teamNames[i] || `Ant ${i}`,
        thumbnail: '',
        isTurn: isRealMove(sub),
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        submission: typeof sub === 'number' ? sub : null,
      };
    });

    return {
      step: index,
      players,
      boardState: parseBoardState(step),
    };
  });
};
