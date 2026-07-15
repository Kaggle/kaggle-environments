// Snake replay transformer.
//
// Populates `players[]` so the side-panel sidebar can render each mover's
// thoughts / action label, and pre-parses the observation JSON into
// `boardState` for the renderer.

interface SnakeAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  generate_returns?: string[] | null;
}

interface SnakeReplayPlayer {
  action?: SnakeAction;
  reward: number;
  observation: { observationString?: string };
  status?: string;
}

interface SnakePlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
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
}

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
function parseThoughts(action?: SnakeAction): string {
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

function parseBoardState(step: SnakeReplayPlayer[]): SnakeBoardState | null {
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
  const rawSteps: SnakeReplayPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): SnakeStep => {
    const players: SnakePlayer[] = step.map(
      (p, i): SnakePlayer => ({
        id: i,
        name: teamNames[i] || `Player ${i}`,
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
