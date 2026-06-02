// Coin Game replay transformer.
//
// Builds the per-step `players` array the side-panel UI expects so the
// right-hand Game Log can render each agent's reasoning, mirroring the
// dark_hex / Y / Lines of Action transformers.

interface CoinAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface CoinObservation {
  observationString?: string;
  isTerminal?: boolean;
  currentPlayer?: number;
}

interface CoinReplayPlayer {
  action?: CoinAction;
  reward: number;
  observation: CoinObservation;
  status?: string;
}

interface CoinPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export type CoinCell = string;

export interface CoinBoardState {
  phase: string;
  board: CoinCell[][];
  num_rows: number;
  num_columns: number;
  coin_colors: string[];
  player_positions: Record<string, [number, number] | null>;
  coins_collected: Record<string, Record<string, number>>;
  current_player: number;
  move_number: number;
  moves_remaining: number;
  episode_length: number;
  is_terminal: boolean;
  winner: number | string | null;
  last_action: string | null;
  your_preference?: string;
  your_player_id?: number;
  preferences?: Record<string, string>;
  returns?: number[];
}

export interface CoinStep {
  step: number;
  players: CoinPlayer[];
  boardState: CoinBoardState | null;
  // Per-player private observation (carries each player's preference).
  privateObs: (CoinBoardState | null)[];
  lastActor: number | null;
  lastAction: string | null;
  isTerminal: boolean;
  winner: number | string | null;
}

function parseThoughts(action?: CoinAction): string {
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

function parsePerPlayer(step: CoinReplayPlayer[]): (CoinBoardState | null)[] {
  return step.map((p) => {
    const raw = p?.observation?.observationString;
    if (!raw) return null;
    try {
      return JSON.parse(raw) as CoinBoardState;
    } catch {
      return null;
    }
  });
}

export const coinGameTransformer = (environment: any): CoinStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 0', 'Player 1'];
  const rawSteps: CoinReplayPlayer[][] = environment?.steps ?? [];
  const out: CoinStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: CoinPlayer[] = step.map((p, i): CoinPlayer => {
      const submission = p.action?.submission;
      const isTurn = submission !== undefined && submission !== -1;
      return {
        id: i,
        name: teamNames[i] ?? `Player ${i}`,
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

    const privateObs = parsePerPlayer(step);
    // Pick whichever per-player JSON is non-null as the canonical view; both
    // share board state, only the preference differs.
    const boardState = privateObs.find((o) => o !== null) ?? null;
    const lastActor = players.findIndex((pl) => pl.isTurn);
    const lastAction = lastActor >= 0 ? players[lastActor].actionDisplayText : null;

    const isTerminal = !!step[0]?.observation?.isTerminal || !!boardState?.is_terminal;
    out.push({
      step: index,
      players,
      boardState,
      privateObs,
      lastActor: lastActor >= 0 ? lastActor : null,
      lastAction,
      isTerminal,
      winner: isTerminal ? (boardState?.winner ?? null) : null,
    });
  });

  return out;
};
