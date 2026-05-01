// Dark Hex replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs so the
// right-hand Game Log can render each agent's reasoning. Dark Hex is
// imperfect-information, so each player sees their own board view --
// we surface both views as `boardX` / `boardO` for the renderer.

interface DarkHexAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface DarkHexObservationEnvelope {
  observationString?: string;
  isTerminal?: boolean;
}

interface DarkHexReplayPlayer {
  action?: DarkHexAction;
  reward: number;
  observation: DarkHexObservationEnvelope;
  status?: string;
}

interface DarkHexPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export interface DarkHexBoardState {
  board: string[][];
  current_player: string;
  is_terminal: boolean;
  winner: string | null;
  num_rows: number;
  num_cols: number;
}

export interface DarkHexStep {
  step: number;
  players: DarkHexPlayer[];
  boardX: DarkHexBoardState | null;
  boardO: DarkHexBoardState | null;
  lastAction: number | null;
  lastActor: number | null;
  isTerminal: boolean;
  winner: string | null;
}

function parseThoughts(action?: DarkHexAction): string {
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

function parseBoardState(player: DarkHexReplayPlayer | undefined): DarkHexBoardState | null {
  const raw = player?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as DarkHexBoardState;
  } catch {
    return null;
  }
}

function deriveWinner(step: DarkHexReplayPlayer[], teamNames: string[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward;
  const r1 = step[1].reward;
  if (r0 === r1) return 'Draw';
  const winnerIdx = r0 > r1 ? 0 : 1;
  const code = winnerIdx === 0 ? 'X' : 'O';
  return `${teamNames[winnerIdx]} (${code}) wins!`;
}

export const darkHexTransformer = (environment: any): DarkHexStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player X', 'Player O'];
  const rawSteps: DarkHexReplayPlayer[][] = environment?.steps ?? [];
  const out: DarkHexStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: DarkHexPlayer[] = step.map((p, i): DarkHexPlayer => {
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

    // Skip steps where neither player acted (e.g. the env's setup step).
    if (!players.some((pl) => pl.isTurn)) return;

    let lastAction: number | null = null;
    let lastActor: number | null = null;
    for (let i = 0; i < step.length; i++) {
      const sub = step[i]?.action?.submission;
      if (typeof sub === 'number' && sub >= 0) {
        lastAction = sub;
        lastActor = i;
        break;
      }
    }

    const isTerminal = !!step[0]?.observation?.isTerminal;
    out.push({
      step: index,
      players,
      boardX: parseBoardState(step[0]),
      boardO: parseBoardState(step[1]),
      lastAction,
      lastActor,
      isTerminal,
      winner: isTerminal ? deriveWinner(step, teamNames) : null,
    });
  });

  return out;
};
