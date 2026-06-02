// Backgammon replay transformer.
//
// Converts raw OpenSpiel step arrays into the structured form the side-panel
// UI expects (one `players` entry per seat, plus a parsed `boardState`).
// Backgammon is sequential: only the active player submits a real action;
// the inactive player sends -1.

interface BackgammonAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface BackgammonObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface BackgammonReplayPlayer {
  action?: BackgammonAction;
  reward: number;
  observation: BackgammonObservation;
  status?: string;
}

interface BackgammonPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export interface BackgammonPoint {
  player: 'x' | 'o';
  count: number;
}

export interface BackgammonDie {
  value: number;
  used: boolean;
}

export interface BackgammonBoardState {
  board: (BackgammonPoint | null)[];
  bar: { x: number; o: number };
  off: { x: number; o: number };
  dice: BackgammonDie[];
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  move_number: number;
}

export interface BackgammonStep {
  step: number;
  players: BackgammonPlayer[];
  boardState: BackgammonBoardState | null;
  isTerminal: boolean;
  winner: string | null;
}

function parseThoughts(action?: BackgammonAction): string {
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      // fall through
    }
  }
  return action?.thoughts ?? '';
}

function parseBoardState(step: BackgammonReplayPlayer[]): BackgammonBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as BackgammonBoardState;
  } catch {
    return null;
  }
}

function deriveWinner(step: BackgammonReplayPlayer[], teamNames: string[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward ?? 0;
  const r1 = step[1].reward ?? 0;
  if (r0 === r1) return 'Draw';
  return r0 > r1 ? `${teamNames[0]} wins!` : `${teamNames[1]} wins!`;
}

export const backgammonTransformer = (environment: any): BackgammonStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: BackgammonReplayPlayer[][] = environment?.steps ?? [];
  const out: BackgammonStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: BackgammonPlayer[] = step.map((p, i): BackgammonPlayer => {
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
      winner: isTerminal ? deriveWinner(step, teamNames) : null,
    });
  });

  return out;
};
