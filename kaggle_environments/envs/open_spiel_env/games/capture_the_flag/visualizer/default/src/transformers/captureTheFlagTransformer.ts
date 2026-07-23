// Capture the Flag replay transformer.
//
// CTF is a simultaneous-move game: both players submit an action every step,
// and the framework then resolves a chance node for move-order initiative.
// After every "real" step both players have submission != -1.

const FORFEIT_STATUSES = new Set(['TIMEOUT', 'ERROR', 'INVALID']);
const FORFEIT_REASONS: Record<string, string> = {
  TIMEOUT: 'ran out of time',
  INVALID: 'submitted an illegal move',
  ERROR: 'failed to produce valid input',
};

interface RawAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface RawObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface RawPlayer {
  action?: RawAction;
  observation: RawObservation;
  reward: number;
  status?: string;
}

export interface CtfPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
  forfeited: boolean;
  forfeitLastAttempt: string | null;
}

export type CtfCell = '.' | '*' | 'A' | 'B' | 'a' | 'b';

export interface CtfBoardState {
  board: CtfCell[][];
  num_rows: number;
  num_cols: number;
  a_base: [number, number];
  b_base: [number, number];
  obstacles: [number, number][];
  a_pos: [number, number];
  b_pos: [number, number];
  flag_a_pos: [number, number];
  flag_b_pos: [number, number];
  carrier_a: number | null;
  carrier_b: number | null;
  score: [number, number];
  score_limit: number;
  horizon: number;
  move_number: number;
  current_player: string | number;
  action_names: string[];
  is_terminal: boolean;
  winner: number | 'draw' | null;
}

export interface CtfStep {
  step: number;
  players: CtfPlayer[];
  boardState: CtfBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  forfeitReason: string | null;
}

function parseThoughts(action?: RawAction): string {
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

function parseBoardState(step: RawPlayer[]): CtfBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as CtfBoardState;
  } catch {
    return null;
  }
}

// Detect a forfeit and its reason category. Two signals:
//   1. top-level player.status in FORFEIT_STATUSES
//   2. action.submission === -1 with a non-null action.status
//      (illegalMoveForfeit path: env normalizes top-level statuses to DONE)
function detectForfeit(step: RawPlayer[]): { index: number; reasonKey: string } | null {
  if (step.length < 2) return null;

  const byStatus = step.map((p, i) => ({ p, i })).filter(({ p }) => p.status && FORFEIT_STATUSES.has(p.status));
  if (byStatus.length === 1) return { index: byStatus[0].i, reasonKey: byStatus[0].p.status! };
  if (byStatus.length > 1) return null;

  const byAction = step.map((p, i) => ({ p, i })).filter(({ p }) => p.action?.submission === -1 && !!p.action?.status);
  if (byAction.length === 1) return { index: byAction[0].i, reasonKey: 'INVALID' };
  return null;
}

function deriveWinner(step: RawPlayer[], teamNames: string[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward ?? 0;
  const r1 = step[1].reward ?? 0;
  if (r0 === r1) return 'Draw';
  return r0 > r1 ? `${teamNames[0]} wins!` : `${teamNames[1]} wins!`;
}

export const captureTheFlagTransformer = (environment: any): CtfStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player A', 'Player B'];
  const rawSteps: RawPlayer[][] = environment?.steps ?? [];
  const out: CtfStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: CtfPlayer[] = step.map((p, i): CtfPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      const isTurn = (submission !== undefined && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? `Player ${String.fromCharCode(65 + i)}`,
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    // Drop the env's setup step where both players submit -1.
    if (!players.some((pl) => pl.isTurn)) return;

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    const isTerminal = observationTerminal || forfeit !== null;

    let forfeitReason: string | null = null;
    if (forfeit) {
      const loser = teamNames[forfeit.index] ?? `Player ${String.fromCharCode(65 + forfeit.index)}`;
      const winnerIdx = 1 - forfeit.index;
      const winner = teamNames[winnerIdx] ?? `Player ${String.fromCharCode(65 + winnerIdx)}`;
      forfeitReason = `${loser} ${FORFEIT_REASONS[forfeit.reasonKey] ?? 'forfeited'}. ${winner} wins by default.`;
    }

    out.push({
      step: index,
      players,
      boardState: parseBoardState(step),
      isTerminal,
      winner: isTerminal ? deriveWinner(step, teamNames) : null,
      forfeitReason,
    });
  });

  return out;
};
