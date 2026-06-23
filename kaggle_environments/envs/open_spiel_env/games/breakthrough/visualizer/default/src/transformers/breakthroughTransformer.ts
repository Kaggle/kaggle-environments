// Breakthrough replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs. Breakthrough
// is a sequential game: only the active player produces a real action in any
// given step; the inactive player just submits -1.
//
// We surface the LLM's `thoughts` (and `generate_returns`, when present, in
// the Go-harness shape) so the right-hand Game Log can render reasoning.

interface BreakthroughAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface BreakthroughObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface BreakthroughReplayPlayer {
  action?: BreakthroughAction;
  reward: number;
  observation: BreakthroughObservation;
  status?: string;
}

interface BreakthroughPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export type BreakthroughCell = 'b' | 'w' | '.';

export interface BreakthroughBoardState {
  board: BreakthroughCell[][];
  rows: number;
  columns: number;
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  last_move: string | null;
  move_number: number;
  pieces: { b: number; w: number };
  params: { rows: number; columns: number };
}

export interface BreakthroughStep {
  step: number;
  players: BreakthroughPlayer[];
  boardState: BreakthroughBoardState | null;
  isTerminal: boolean;
}

function parseThoughts(action?: BreakthroughAction): string {
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      // fall through to action.thoughts
    }
  }
  return action?.thoughts ?? '';
}

function parseBoardState(step: BreakthroughReplayPlayer[]): BreakthroughBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as BreakthroughBoardState;
  } catch {
    return null;
  }
}

export const breakthroughTransformer = (environment: any): BreakthroughStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: BreakthroughReplayPlayer[][] = environment?.steps ?? [];
  const out: BreakthroughStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: BreakthroughPlayer[] = step.map((p, i): BreakthroughPlayer => {
      const submission = p.action?.submission;
      const isTurn = submission !== undefined && submission !== -1;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Player 1' : 'Player 2'),
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
      };
    });

    // Skip the env's setup step where both players submit -1.
    if (!players.some((pl) => pl.isTurn)) return;

    const isTerminal = !!step[0]?.observation?.isTerminal;
    out.push({
      step: index,
      players,
      boardState: parseBoardState(step),
      isTerminal,
    });
  });

  return out;
};
