// Shogi replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs and parses the
// proxy's JSON observation into a typed board state. Shogi is sequential:
// only the active player submits a real action; the other side submits -1.

interface ShogiAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface ShogiObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface ShogiReplayPlayer {
  action?: ShogiAction;
  reward: number;
  observation: ShogiObservation;
  status?: string;
}

interface ShogiPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

// Each cell is either '.', a single piece letter (uppercase = Sente / Black,
// lowercase = Gote / White), or a 2-char string starting with '+' marking a
// promoted piece (e.g. '+P', '+r').
export type ShogiCell = string;

export interface ShogiHandCounts {
  [piece: string]: number;
}

export interface ShogiBoardState {
  board: ShogiCell[][];
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  captured: { b: ShogiHandCounts; w: ShogiHandCounts };
  move_number: number;
  last_move: string | null;
  move_history: string[];
  sfen: string;
}

export interface ShogiStep {
  step: number;
  players: ShogiPlayer[];
  boardState: ShogiBoardState | null;
  isTerminal: boolean;
}

function parseThoughts(action?: ShogiAction): string {
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

function parseBoardState(step: ShogiReplayPlayer[]): ShogiBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as ShogiBoardState;
  } catch {
    return null;
  }
}

export const shogiTransformer = (environment: any): ShogiStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: ShogiReplayPlayer[][] = environment?.steps ?? [];
  const out: ShogiStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: ShogiPlayer[] = step.map((p, i): ShogiPlayer => {
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
    if (!players.some((pl) => pl.isTurn) && index > 0) return;

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
