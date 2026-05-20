// Dots and Boxes replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs so the
// right-hand Game Log can render each agent's reasoning, and parses the
// proxy's JSON observationString into a typed board state for the renderer.

interface DotsAndBoxesAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface DotsAndBoxesObservationEnvelope {
  observationString?: string;
  isTerminal?: boolean;
}

interface DotsAndBoxesReplayPlayer {
  action?: DotsAndBoxesAction;
  reward: number;
  observation: DotsAndBoxesObservationEnvelope;
  status?: string;
}

interface DotsAndBoxesPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export interface DotsAndBoxesLastAction {
  orientation: 'h' | 'v';
  row: number;
  col: number;
  player: '1' | '2' | '';
}

export interface DotsAndBoxesBoardState {
  num_rows: number;
  num_cols: number;
  h_lines: number[][];
  v_lines: number[][];
  boxes: number[][];
  scores: [number, number];
  current_player: '1' | '2' | '';
  is_terminal: boolean;
  winner: '1' | '2' | 'draw' | null;
  last_action: DotsAndBoxesLastAction | null;
}

export interface DotsAndBoxesStep {
  step: number;
  players: DotsAndBoxesPlayer[];
  boardState: DotsAndBoxesBoardState | null;
  isTerminal: boolean;
  winner: string | null;
}

function parseThoughts(action?: DotsAndBoxesAction): string {
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

function parseBoardState(step: DotsAndBoxesReplayPlayer[]): DotsAndBoxesBoardState | null {
  // Both agents see the same observationString; pick whichever is populated.
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as DotsAndBoxesBoardState;
  } catch {
    return null;
  }
}

function deriveWinner(step: DotsAndBoxesReplayPlayer[], teamNames: string[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward;
  const r1 = step[1].reward;
  if (r0 === r1) return 'Draw';
  const winnerIdx = r0 > r1 ? 0 : 1;
  const code = winnerIdx === 0 ? '1' : '2';
  return `${teamNames[winnerIdx]} (${code}) wins!`;
}

export const dotsAndBoxesTransformer = (environment: any): DotsAndBoxesStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: DotsAndBoxesReplayPlayer[][] = environment?.steps ?? [];
  const out: DotsAndBoxesStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: DotsAndBoxesPlayer[] = step.map((p, i): DotsAndBoxesPlayer => {
      const submission = p.action?.submission;
      // Match the other transformers: submission undefined or -1 means the
      // player did not act this step (setup step or inactive player).
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

    // Skip steps where neither player acted (the env's setup step where both
    // submit -1). The side-panel uses isTurn to pick the active player.
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
