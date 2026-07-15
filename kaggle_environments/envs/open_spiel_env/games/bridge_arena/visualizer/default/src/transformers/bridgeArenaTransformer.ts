// Bridge Arena replay transformer.
//
// Populates `players[]` (4 seats) so the side-panel sidebar can show each
// mover's thoughts and call/play. Pre-parses the shared observation JSON
// (whichever player has it populated) into `boardState`.

interface BridgeAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  generate_returns?: string[] | null;
}

interface BridgeReplayPlayer {
  action?: BridgeAction;
  reward: number;
  observation: { observationString?: string };
  status?: string;
}

interface BridgePlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
}

export interface BridgeBoardState {
  phase?: string;
  is_terminal?: boolean;
  dealer_table_position?: string;
  dealer_player_id?: number;
  current_player_id?: number | null;
  current_table_position?: string | null;
  current_team_id?: number | null;
  auction?: unknown[];
  plays?: unknown[];
  table_seating?: Record<string, string>;
  teams?: Record<string, number[]>;
  returns?: number[];
  team_totals?: number[];
  winning_team?: number | string;
}

export interface BridgeStep {
  step: number;
  players: BridgePlayer[];
  boardState: BridgeBoardState | null;
}

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
function parseThoughts(action?: BridgeAction): string {
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

// Bridge observations are per-player (hidden information), but the renderer
// only uses the shared/public parts, and takes them from whichever seat has
// a non-empty observationString.
function parseBoardState(step: BridgeReplayPlayer[]): BridgeBoardState | null {
  for (const p of step) {
    const raw = p?.observation?.observationString;
    if (!raw) continue;
    try {
      return JSON.parse(raw) as BridgeBoardState;
    } catch {
      // try the next seat
    }
  }
  return null;
}

const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const bridgeArenaTransformer = (environment: any): BridgeStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: BridgeReplayPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): BridgeStep => {
    const players: BridgePlayer[] = step.map(
      (p, i): BridgePlayer => ({
        id: i,
        name: teamNames[i] || `Player ${i}`,
        thumbnail: '',
        isTurn: isRealMove(p.action?.submission),
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
      })
    );

    return {
      step: index,
      players,
      boardState: parseBoardState(step),
    };
  });
};
