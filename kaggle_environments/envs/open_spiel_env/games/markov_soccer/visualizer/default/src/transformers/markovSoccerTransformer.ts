// Markov Soccer replay transformer.
//
// Populates `players[]` so the side-panel can show each mover's thoughts
// and last action, and pre-parses the observation JSON into `boardState`.
// Markov Soccer is a simultaneous-move game — both seats submit a real
// action every round, so both players are `isTurn: true` on played steps.

interface SoccerAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  generate_returns?: string[] | null;
}

interface SoccerReplayPlayer {
  action?: SoccerAction;
  info?: { actionSubmittedToString?: string | null };
  reward: number;
  observation: { observationString?: string };
  status?: string;
}

interface SoccerPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
}

type Pos = [number, number] | null;

export interface SoccerBoardState {
  board: string[][];
  current_player: string;
  is_terminal: boolean;
  winner: 'A' | 'B' | 'draw' | null;
  player_a_pos: Pos;
  player_b_pos: Pos;
  ball_pos: Pos;
  ball_owner: 'A' | 'B' | null;
}

export interface SoccerStep {
  step: number;
  players: SoccerPlayer[];
  boardState: SoccerBoardState | null;
}

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
function parseThoughts(action?: SoccerAction): string {
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

function parseBoardState(step: SoccerReplayPlayer[]): SoccerBoardState | null {
  const raw = step?.[0]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as SoccerBoardState;
  } catch {
    return null;
  }
}

const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const markovSoccerTransformer = (environment: any): SoccerStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: SoccerReplayPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): SoccerStep => {
    const players: SoccerPlayer[] = step.map(
      (p, i): SoccerPlayer => ({
        id: i,
        name: teamNames[i] || (i === 0 ? 'Player A' : 'Player B'),
        thumbnail: '',
        isTurn: isRealMove(p.action?.submission),
        // info.actionSubmittedToString is what the renderer historically
        // displayed. Fall back to action.actionString when info is absent.
        actionDisplayText: p.info?.actionSubmittedToString ?? p.action?.actionString ?? '',
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
