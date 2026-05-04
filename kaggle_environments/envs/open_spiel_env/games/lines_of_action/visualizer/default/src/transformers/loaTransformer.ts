// Lines of Action replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs so the
// right-hand Game Log can render each agent's reasoning. Mirrors the
// pattern used by the dark_hex / amazons / Y transformers.

interface LoaAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface LoaObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface LoaReplayPlayer {
  action?: LoaAction;
  reward: number;
  observation: LoaObservation;
  status?: string;
}

interface LoaPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export type LoaCell = '.' | 'x' | 'o';

export interface LoaBoardState {
  board: LoaCell[][];
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  move_number: number;
  last_move: string | null;
}

export interface LoaMove {
  fromRow: number;
  fromCol: number;
  toRow: number;
  toCol: number;
  capture: boolean;
}

export interface LoaStep {
  step: number;
  players: LoaPlayer[];
  boardState: LoaBoardState | null;
  lastMove: LoaMove | null;
  lastActor: number | null;
  isTerminal: boolean;
  winner: string | null;
}

function parseThoughts(action?: LoaAction): string {
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

function parseBoardState(step: LoaReplayPlayer[]): LoaBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as LoaBoardState;
  } catch {
    return null;
  }
}

function parseMoveString(s: string | null | undefined): LoaMove | null {
  if (!s || s.length < 5) return null;
  // Format: "<col><row><x|->/<col><row>" e.g. "b1-h1" or "c3xa3".
  const fromCol = s.charCodeAt(0) - 'a'.charCodeAt(0);
  const fromRow = parseInt(s[1], 10) - 1;
  const sep = s[2];
  const toCol = s.charCodeAt(3) - 'a'.charCodeAt(0);
  const toRow = parseInt(s[4], 10) - 1;
  if (
    fromCol < 0 ||
    fromCol > 7 ||
    toCol < 0 ||
    toCol > 7 ||
    isNaN(fromRow) ||
    isNaN(toRow) ||
    (sep !== '-' && sep !== 'x')
  ) {
    return null;
  }
  return { fromRow, fromCol, toRow, toCol, capture: sep === 'x' };
}

function deriveWinner(boardState: LoaBoardState | null, teamNames: string[]): string | null {
  if (!boardState) return null;
  if (boardState.winner === 'x') return `${teamNames[0]} (X) wins!`;
  if (boardState.winner === 'o') return `${teamNames[1]} (O) wins!`;
  return 'Draw';
}

export const loaTransformer = (environment: any): LoaStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player X', 'Player O'];
  const rawSteps: LoaReplayPlayer[][] = environment?.steps ?? [];
  const out: LoaStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: LoaPlayer[] = step.map((p, i): LoaPlayer => {
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

    const boardState = parseBoardState(step);
    const lastMove = parseMoveString(boardState?.last_move);
    // Move parity: black (x, player 0) moves on odd move_number, after which it's white's turn.
    const lastActor = boardState && boardState.move_number > 0 ? (boardState.move_number - 1) % 2 : null;
    const isTerminal = !!step[0]?.observation?.isTerminal;
    out.push({
      step: index,
      players,
      boardState,
      lastMove,
      lastActor,
      isTerminal,
      winner: isTerminal ? deriveWinner(boardState, teamNames) : null,
    });
  });

  return out;
};
