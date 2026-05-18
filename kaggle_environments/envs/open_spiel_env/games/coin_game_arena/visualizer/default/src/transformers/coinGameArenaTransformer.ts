// Coin Game Arena replay transformer.
//
// Each step in the raw replay has 4 player slots (2 per team). Each
// slot's observationString is a per-player JSON view containing only
// that player's team's board. We combine the per-player observations
// to recover both boards (board A from player 0/1, board B from player
// 2/3). At terminal, every player observation reveals both boards via
// the ``boards`` array, plus the per-player ``preferences``.

interface ArenaAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface ArenaObservation {
  observationString?: string;
  isTerminal?: boolean;
  currentPlayer?: number;
}

interface ArenaReplayPlayer {
  action?: ArenaAction;
  reward: number;
  observation: ArenaObservation;
  status?: string;
}

export interface ArenaBoardView {
  team_id: number;
  board: string[][];
  num_rows: number;
  num_columns: number;
  coin_colors: string[];
  player_positions: Record<string, [number, number] | null>;
  coins_collected: Record<string, Record<string, number>>;
  move_history?: { seat: number; player_id: number; action: string }[];
}

interface ArenaPerPlayerObs {
  phase: string;
  move_number: number;
  moves_remaining: number;
  episode_length: number;
  active_seat: number | null;
  num_teams: number;
  players_per_team: number;
  is_terminal: boolean;
  your_player_id?: number;
  your_team_id?: number;
  your_seat?: number;
  your_preference?: string;
  your_turn?: boolean;
  board?: ArenaBoardView;
  boards?: ArenaBoardView[];
  preferences?: Record<string, string>;
  returns?: number[];
  team_totals?: number[];
  winning_team?: number | string | null;
}

export interface ArenaPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
  teamId: number;
  seat: number;
}

export interface ArenaStep {
  step: number;
  players: ArenaPlayer[];
  boards: (ArenaBoardView | null)[]; // index 0 = team A, 1 = team B
  privateObs: (ArenaPerPlayerObs | null)[];
  preferences: (string | null)[]; // length = num players (4)
  moveNumber: number | null;
  movesRemaining: number | null;
  episodeLength: number | null;
  activeSeat: number | null;
  isTerminal: boolean;
  teamTotals: number[] | null;
  returns: number[] | null;
  winningTeam: number | string | null;
  // Last action played per board, for arrows / highlights.
  lastActionPerBoard: { actor: number; action: string }[];
}

const PLAYERS_PER_TEAM = 2;

function parseThoughts(action?: ArenaAction): string {
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      /* fall through */
    }
  }
  return action?.thoughts ?? '';
}

function parsePerPlayer(step: ArenaReplayPlayer[]): (ArenaPerPlayerObs | null)[] {
  return step.map((p) => {
    const raw = p?.observation?.observationString;
    if (!raw) return null;
    try {
      return JSON.parse(raw) as ArenaPerPlayerObs;
    } catch {
      return null;
    }
  });
}

function pickBoardForTeam(privateObs: (ArenaPerPlayerObs | null)[], teamId: number): ArenaBoardView | null {
  // Prefer terminal full reveal if present on any obs.
  for (const obs of privateObs) {
    if (obs?.boards && obs.boards[teamId]) return obs.boards[teamId];
  }
  // Otherwise use whichever player on that team has a per-player view.
  for (let pid = teamId * PLAYERS_PER_TEAM; pid < (teamId + 1) * PLAYERS_PER_TEAM; pid++) {
    const obs = privateObs[pid];
    if (obs?.board && obs.board.team_id === teamId) return obs.board;
  }
  return null;
}

export const coinGameArenaTransformer = (environment: any): ArenaStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 0', 'Player 1', 'Player 2', 'Player 3'];
  const rawSteps: ArenaReplayPlayer[][] = environment?.steps ?? [];
  const out: ArenaStep[] = [];

  // Cumulative per-player preferences (each obs only reveals the
  // viewer's own preference until terminal).
  const cumulativePrefs: (string | null)[] = [null, null, null, null];

  rawSteps.forEach((step, index) => {
    const privateObs = parsePerPlayer(step);
    privateObs.forEach((obs, pid) => {
      if (obs?.your_preference) cumulativePrefs[pid] = obs.your_preference;
      if (obs?.preferences) {
        for (const [k, v] of Object.entries(obs.preferences)) {
          cumulativePrefs[Number(k)] = v;
        }
      }
    });

    const players: ArenaPlayer[] = step.map((p, i): ArenaPlayer => {
      const submission = p.action?.submission;
      const isTurn = submission !== undefined && submission !== null && submission !== -1;
      return {
        id: i,
        name: teamNames[i] ?? `Player ${i}`,
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        teamId: Math.floor(i / PLAYERS_PER_TEAM),
        seat: i % PLAYERS_PER_TEAM,
      };
    });

    // Skip the env's setup step where everyone submits -1.
    if (!players.some((pl) => pl.isTurn) && index > 0) return;

    const boards: (ArenaBoardView | null)[] = [pickBoardForTeam(privateObs, 0), pickBoardForTeam(privateObs, 1)];

    const lastActionPerBoard: { actor: number; action: string }[] = [];
    for (const pl of players) {
      if (pl.isTurn) {
        lastActionPerBoard[pl.teamId] = {
          actor: pl.id,
          action: pl.actionDisplayText,
        };
      }
    }

    // Sample one obs to read shared state (move_number, etc.).
    const sample = privateObs.find((o) => o !== null) ?? null;
    const isTerminal = !!step[0]?.observation?.isTerminal || !!sample?.is_terminal;

    out.push({
      step: index,
      players,
      boards,
      privateObs,
      preferences: [...cumulativePrefs],
      moveNumber: sample?.move_number ?? null,
      movesRemaining: sample?.moves_remaining ?? null,
      episodeLength: sample?.episode_length ?? null,
      activeSeat: sample?.active_seat ?? null,
      isTerminal,
      teamTotals: sample?.team_totals ?? null,
      returns: sample?.returns ?? null,
      winningTeam: sample?.winning_team ?? null,
      lastActionPerBoard,
    });
  });

  return out;
};
