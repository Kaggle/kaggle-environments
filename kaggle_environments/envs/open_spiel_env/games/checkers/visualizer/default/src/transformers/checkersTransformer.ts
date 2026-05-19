// Checkers replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs. Checkers is a
// sequential game: only the active player produces a real action in any
// given step; the inactive player submits -1.

interface CheckersAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface CheckersObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface CheckersReplayPlayer {
  action?: CheckersAction;
  reward: number;
  observation: CheckersObservation;
  status?: string;
}

interface CheckersPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export type CheckersCell = 'o' | '+' | 'O' | '*' | '.';

export interface CheckersBoardState {
  board: CheckersCell[][];
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  last_move: string | null;
  move_number: number;
  piece_counts: { o: number; '+': number; O: number; '*': number };
}

export interface CheckersStep {
  step: number;
  players: CheckersPlayer[];
  boardState: CheckersBoardState | null;
  isTerminal: boolean;
  winner: string | null;
}

function parseThoughts(action?: CheckersAction): string {
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

function parseBoardState(step: CheckersReplayPlayer[]): CheckersBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as CheckersBoardState;
  } catch {
    return null;
  }
}

function deriveWinner(step: CheckersReplayPlayer[], teamNames: string[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward ?? 0;
  const r1 = step[1].reward ?? 0;
  if (r0 === r1) return 'Draw';
  return r0 > r1 ? `${teamNames[0]} wins!` : `${teamNames[1]} wins!`;
}

export const checkersTransformer = (environment: any): CheckersStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: CheckersReplayPlayer[][] = environment?.steps ?? [];
  const out: CheckersStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: CheckersPlayer[] = step.map((p, i): CheckersPlayer => {
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
