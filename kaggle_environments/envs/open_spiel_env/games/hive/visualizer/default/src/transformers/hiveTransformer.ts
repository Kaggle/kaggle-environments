// Hive replay transformer.
//
// Builds the per-step `players` array that the side-panel UI needs and parses
// the proxy's JSON observation into a typed board state.

interface HiveAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface HiveObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface HiveReplayPlayer {
  action?: HiveAction;
  reward: number;
  observation: HiveObservation;
  status?: string;
}

interface HivePlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export type HivePieces = Record<string, [number, number, number]>;

export interface HiveBoardState {
  game_type: string;
  expansions?: { mosquito?: boolean; ladybug?: boolean; pillbug?: boolean };
  board_radius?: number;
  status: string;
  turn?: string;
  current_player: string;
  move_number: number;
  moves: string[];
  last_move: string | null;
  pieces: HivePieces;
  is_terminal: boolean;
  winner: string | null;
  uhp?: string;
}

export interface HiveStep {
  step: number;
  players: HivePlayer[];
  boardState: HiveBoardState | null;
  isTerminal: boolean;
  winner: string | null;
}

function parseThoughts(action?: HiveAction): string {
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

function parseBoardState(step: HiveReplayPlayer[]): HiveBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as HiveBoardState;
  } catch {
    return null;
  }
}

function deriveWinner(board: HiveBoardState | null, step: HiveReplayPlayer[]): string | null {
  if (board?.winner === 'white') return 'White wins!';
  if (board?.winner === 'black') return 'Black wins!';
  if (board?.winner === 'draw') return 'Draw';
  if (step.length < 2) return null;
  const r0 = step[0].reward;
  const r1 = step[1].reward;
  if (r0 === r1) return 'Draw';
  return r0 > r1 ? 'White wins!' : 'Black wins!';
}

export const hiveTransformer = (environment: any): HiveStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['White', 'Black'];
  const rawSteps: HiveReplayPlayer[][] = environment?.steps ?? [];
  const out: HiveStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: HivePlayer[] = step.map((p, i): HivePlayer => {
      const submission = p.action?.submission;
      const isTurn = submission !== undefined && submission !== -1;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'White' : 'Black'),
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

    const boardState = parseBoardState(step);
    const isTerminal = !!step[0]?.observation?.isTerminal || !!boardState?.is_terminal;
    out.push({
      step: index,
      players,
      boardState,
      isTerminal,
      winner: isTerminal ? deriveWinner(boardState, step) : null,
    });
  });

  return out;
};
