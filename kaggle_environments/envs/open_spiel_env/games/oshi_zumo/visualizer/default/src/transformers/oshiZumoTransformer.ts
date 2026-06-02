// Oshi-Zumo replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs. Oshi-Zumo is a
// simultaneous-move game: each non-setup step has both players acting, so we
// surface both players' bids, rewards, and reasoning to the right-hand Game
// Log. Mirrors the amazons transformer pattern.

interface OshiZumoAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface OshiZumoObservation {
  observationString?: string;
  isTerminal?: boolean;
}

interface OshiZumoReplayPlayer {
  action?: OshiZumoAction;
  reward: number;
  observation: OshiZumoObservation;
  status?: string;
}

interface OshiZumoPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
  bid: number | null;
}

export interface OshiZumoBoardState {
  field: string;
  field_size: number;
  wrestler_position: number;
  coins: [number, number];
  current_player: number | string;
  move_number: number;
  is_terminal: boolean;
  winner: number | string | null;
  params: {
    alesia: boolean;
    starting_coins: number;
    size: number;
    horizon: number;
    min_bid: number;
  };
}

export interface OshiZumoStep {
  step: number;
  players: OshiZumoPlayer[];
  boardState: OshiZumoBoardState | null;
  bids: [number | null, number | null];
  isTerminal: boolean;
  winner: string | null;
}

function parseThoughts(action?: OshiZumoAction): string {
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

function parseBoardState(step: OshiZumoReplayPlayer[]): OshiZumoBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as OshiZumoBoardState;
  } catch {
    return null;
  }
}

function deriveWinner(step: OshiZumoReplayPlayer[], teamNames: string[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward ?? 0;
  const r1 = step[1].reward ?? 0;
  if (r0 === r1) return 'Draw';
  return r0 > r1 ? `${teamNames[0]} wins!` : `${teamNames[1]} wins!`;
}

export const oshiZumoTransformer = (environment: any): OshiZumoStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: OshiZumoReplayPlayer[][] = environment?.steps ?? [];
  const out: OshiZumoStep[] = [];

  rawSteps.forEach((step, index) => {
    const bids: [number | null, number | null] = [null, null];
    const playerInfo = step.map((p, i) => {
      const submission = p.action?.submission;
      // Setup step / inactive: submission is -1 or undefined.
      const acted = submission !== undefined && submission !== -1;
      const bid = acted ? (submission as number) : null;
      bids[i] = bid;
      return {
        id: i,
        name: teamNames[i] ?? `Player ${i + 1}`,
        acted,
        bid,
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        actionString: p.action?.actionString ?? '',
      };
    });

    // Skip the setup step (neither player acted).
    if (!playerInfo.some((pl) => pl.acted)) return;

    const isTerminal = !!step[0]?.observation?.isTerminal;
    const boardState = parseBoardState(step);
    const winner = isTerminal ? deriveWinner(step, teamNames) : null;

    // Emit one log entry per player who acted, so the side-panel Game Log
    // surfaces both players' bids and reasoning. Each entry shares the same
    // post-bid board state; only the active-player flag differs.
    playerInfo.forEach((info, focusIdx) => {
      if (!info.acted) return;
      const players: OshiZumoPlayer[] = playerInfo.map((pi) => ({
        id: pi.id,
        name: pi.name,
        thumbnail: '',
        isTurn: pi.id === focusIdx,
        actionDisplayText: pi.bid !== null ? `Bid: ${pi.bid}` : pi.actionString,
        thoughts: pi.thoughts,
        reward: pi.reward,
        generateReturns: pi.generateReturns,
        bid: pi.bid,
      }));
      out.push({
        step: index,
        players,
        boardState,
        bids,
        isTerminal,
        winner,
      });
    });
  });

  return out;
};
