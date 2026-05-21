// Havannah replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs. Havannah is a
// sequential game: only the active player produces a real action in any given
// step; the inactive player just submits -1.

interface HavannahAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface HavannahObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface HavannahReplayPlayer {
  action?: HavannahAction;
  reward: number;
  observation: HavannahObservation;
  status?: string;
}

interface HavannahPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export type HavannahCell = 'x' | 'o' | null;

export interface HavannahBoardState {
  board: HavannahCell[][];
  board_size: number;
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  last_move: string | null;
  move_number: number;
}

export interface HavannahStep {
  step: number;
  players: HavannahPlayer[];
  boardState: HavannahBoardState | null;
  isTerminal: boolean;
  winner: string | null;
}

function parseThoughts(action?: HavannahAction): string {
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

function parseBoardState(step: HavannahReplayPlayer[]): HavannahBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as HavannahBoardState;
  } catch {
    return null;
  }
}

function deriveWinner(step: HavannahReplayPlayer[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward;
  const r1 = step[1].reward;
  if (r0 === r1) return 'Draw';
  return r0 > r1 ? 'Player X wins!' : 'Player O wins!';
}

export const havannahTransformer = (environment: any): HavannahStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player X', 'Player O'];
  const rawSteps: HavannahReplayPlayer[][] = environment?.steps ?? [];
  const out: HavannahStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: HavannahPlayer[] = step.map((p, i): HavannahPlayer => {
      const submission = p.action?.submission;
      const isTurn = submission !== undefined && submission !== -1;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Player X' : 'Player O'),
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
      winner: isTerminal ? deriveWinner(step) : null,
    });
  });

  return out;
};
