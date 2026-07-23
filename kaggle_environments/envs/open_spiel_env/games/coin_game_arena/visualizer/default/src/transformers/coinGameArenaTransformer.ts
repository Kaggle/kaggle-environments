// Coin Game Arena replay transformer.
//
// Each step in the raw replay has 4 player slots (2 per team). Each
// slot's observationString is a per-player JSON view containing only
// that player's team's board. We combine the per-player observations
// to recover both boards (board A from player 0/1, board B from player
// 2/3). At terminal, every player observation reveals both boards via
// the ``boards`` array, plus the per-player ``preferences``.
//
// Forfeit handling (illegal-move / TIMEOUT / ERROR) uses the shared
// detectForfeit helper for per-seat labeling. Winner / reason strings
// remain arena-specific (team-aware) because the two-player helpers in
// core would report the wrong team on a 4-seat game.

import { detectForfeit, FORFEIT_REASONS, parseThoughts, OpenSpielRawPlayer } from '@kaggle-environments/core';

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
  forfeited: boolean;
  forfeitLastAttempt: string | null;
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
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). Team-
  // aware here: names the offending team as the loser, not just the seat.
  forfeitReason: string | null;
}

const PLAYERS_PER_TEAM = 2;

function parsePerPlayer(step: OpenSpielRawPlayer[]): (ArenaPerPlayerObs | null)[] {
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

function buildArenaForfeitReason(forfeitSeat: number, reasonKey: string, teamNames: string[]): string {
  const loser = teamNames[forfeitSeat] ?? `Player ${forfeitSeat}`;
  const loserTeam = Math.floor(forfeitSeat / PLAYERS_PER_TEAM);
  const winningTeam = 1 - loserTeam;
  // Name the winning team by its two seats when we have names for both.
  const winnerSeatA = winningTeam * PLAYERS_PER_TEAM;
  const winnerSeatB = winnerSeatA + 1;
  const winnerA = teamNames[winnerSeatA] ?? `Player ${winnerSeatA}`;
  const winnerB = teamNames[winnerSeatB] ?? `Player ${winnerSeatB}`;
  const reason = FORFEIT_REASONS[reasonKey] ?? 'forfeited';
  return `${loser} ${reason}. Team ${winningTeam} (${winnerA} & ${winnerB}) wins by default.`;
}

export const coinGameArenaTransformer = (environment: any): ArenaStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 0', 'Player 1', 'Player 2', 'Player 3'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: ArenaStep[] = [];

  // Cumulative per-player preferences (each obs only reveals the
  // viewer's own preference until terminal).
  const cumulativePrefs: (string | null)[] = [null, null, null, null];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

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
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender has submission === -1 but should still be
      // treated as "acting" so the step is retained and their thoughts /
      // last-attempt render in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
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
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    // Skip the env's setup step where everyone submits -1 and no one
    // forfeited.
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
    const observationTerminal = !!step[0]?.observation?.isTerminal || !!sample?.is_terminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    // On a natural terminal use the arena's own team winner; on forfeit
    // fall back to "the other team wins by default".
    let winningTeam: number | string | null = sample?.winning_team ?? null;
    if (forfeit && (winningTeam === null || winningTeam === undefined)) {
      winningTeam = 1 - Math.floor(forfeit.index / PLAYERS_PER_TEAM);
    }

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
      winningTeam,
      lastActionPerBoard,
      forfeitReason: forfeit ? buildArenaForfeitReason(forfeit.index, forfeit.reasonKey, teamNames) : null,
    });
  });

  return out;
};
