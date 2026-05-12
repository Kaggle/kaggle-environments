// Amazons replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs. Each Amazons
// turn is three sub-actions (from / to / shoot), and only the active player
// produces an action in any given step — the inactive player just waits.
//
// We surface the LLM's `thoughts` (and `generate_returns`, when present, in
// the Go-harness shape) so the right-hand Game Log can render reasoning.

interface AmazonsAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface AmazonsObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface AmazonsReplayPlayer {
  action?: AmazonsAction;
  reward: number;
  observation: AmazonsObservation;
  status?: string;
}

interface AmazonsPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export type AmazonsCell = 'X' | 'O' | '#' | '.';

export interface AmazonsBoardState {
  board: AmazonsCell[][];
  board_size: number;
  current_player: string;
  phase: 'from' | 'to' | 'shoot' | null;
  move_number: number;
  is_terminal: boolean;
  winner: string | null;
}

export interface AmazonsStep {
  step: number;
  players: AmazonsPlayer[];
  boardState: AmazonsBoardState | null;
  isTerminal: boolean;
  winner: string | null;
}

function parseThoughts(action?: AmazonsAction): string {
  // Prefer the Go-harness ``main_response_and_thoughts`` payload when
  // generate_returns is populated; fall through to ``action.thoughts`` if
  // generate_returns is missing, fails to parse, or doesn't carry the field.
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

function parseBoardState(step: AmazonsReplayPlayer[]): AmazonsBoardState | null {
  // Both agents see the same observationString; pick whichever is populated.
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AmazonsBoardState;
  } catch {
    return null;
  }
}

function deriveWinner(step: AmazonsReplayPlayer[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward;
  const r1 = step[1].reward;
  if (r0 === r1) return 'Draw';
  return r0 > r1 ? 'Black (X) wins!' : 'White (O) wins!';
}

export const amazonsTransformer = (environment: any): AmazonsStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Black (X)', 'White (O)'];
  const rawSteps: AmazonsReplayPlayer[][] = environment?.steps ?? [];
  const out: AmazonsStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: AmazonsPlayer[] = step.map((p, i): AmazonsPlayer => {
      const submission = p.action?.submission;
      // Match Go: an action with submission undefined or -1 is a no-op
      // (the env's setup step or the inactive player).
      const isTurn = submission !== undefined && submission !== -1;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Black (X)' : 'White (O)'),
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
      };
    });

    // Skip steps where neither player acted (e.g. the env's setup step where
    // both submit -1). The side-panel uses isTurn to pick the active player.
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
