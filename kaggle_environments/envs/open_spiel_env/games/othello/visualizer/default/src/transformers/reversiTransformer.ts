// Reversi (open_spiel_othello) replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs. Reversi is a
// sequential game: only the active player produces a real action in any given
// step; the inactive player just submits -1. We surface the LLM's `thoughts`
// (and `generate_returns`, when present) so the right-hand Game Log can
// render reasoning.

interface ReversiAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface ReversiObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface ReversiReplayPlayer {
  action?: ReversiAction;
  reward: number;
  observation: ReversiObservation;
  status?: string;
}

interface ReversiPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export type ReversiCell = 'x' | 'o' | '';

export interface ReversiBoardState {
  board: ReversiCell[][];
  rows: number;
  columns: number;
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  disks: { x: number; o: number };
  last_move: string | null;
  move_history: string[];
  move_number: number;
  must_pass: boolean;
}

export interface ReversiStep {
  step: number;
  players: ReversiPlayer[];
  boardState: ReversiBoardState | null;
  isTerminal: boolean;
}

function parseThoughts(action?: ReversiAction): string {
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

function parseBoardState(step: ReversiReplayPlayer[]): ReversiBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as ReversiBoardState;
  } catch {
    return null;
  }
}

export const reversiTransformer = (environment: any): ReversiStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: ReversiReplayPlayer[][] = environment?.steps ?? [];
  const out: ReversiStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: ReversiPlayer[] = step.map((p, i): ReversiPlayer => {
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
