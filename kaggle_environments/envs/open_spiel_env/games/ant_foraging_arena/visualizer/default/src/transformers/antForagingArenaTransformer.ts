// Ant Foraging Arena replay transformer.
//
// Each step in the raw replay has 4 player slots (2 per team). Each
// slot's observationString is a per-player JSON view containing only
// that player's team's board. We combine the per-player observations
// to recover both boards (board A from player 0/1, board B from player
// 2/3). At terminal, every player observation reveals both boards via
// the ``boards`` array.
//
// Forfeit handling (illegal-move / TIMEOUT / ERROR): reuse the shared
// detector, but Ant Foraging is 2v2 (seats 0,1 = team A; seats 2,3 =
// team B) so we build the forfeit reason in-place instead of using the
// 2-player `buildForfeitReason` helper.

import { detectForfeit, FORFEIT_REASONS, parseThoughts, OpenSpielRawPlayer } from '@kaggle-environments/core';

export interface AntBoardView {
  team_id: number;
  grid: string[][];
  grid_size: number;
  num_ants: number;
  num_food: number;
  nest_position: [number, number];
  food_positions: [number, number][];
  ant_positions: Record<string, [number, number] | null>;
  carrying_food: Record<string, boolean>;
  pheromone_to_food: number[][];
  pheromone_to_nest: number[][];
  food_collected: number;
  move_history?: { seat: number; player_id: number; action: string }[];
}

interface ArenaPerPlayerObs {
  phase: string;
  move_number: number;
  moves_remaining: number;
  max_turns: number;
  active_seat: number | null;
  num_teams: number;
  players_per_team: number;
  is_terminal: boolean;
  your_player_id?: number;
  your_team_id?: number;
  your_seat?: number;
  your_turn?: boolean;
  board?: AntBoardView;
  boards?: AntBoardView[];
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
  boards: (AntBoardView | null)[]; // index 0 = team A, 1 = team B
  privateObs: (ArenaPerPlayerObs | null)[];
  moveNumber: number | null;
  movesRemaining: number | null;
  maxTurns: number | null;
  activeSeat: number | null;
  isTerminal: boolean;
  teamTotals: number[] | null;
  returns: number[] | null;
  winningTeam: number | string | null;
  // Last action played per board, for arrows / highlights.
  lastActionPerBoard: { actor: number; action: string }[];
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash).
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

function pickBoardForTeam(privateObs: (ArenaPerPlayerObs | null)[], teamId: number): AntBoardView | null {
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

// Ant Foraging is 2v2 (seats 0,1 = team A; seats 2,3 = team B), so we can't
// use the shared 2-player `buildForfeitReason`. Build the reason inline.
function buildArenaForfeitReason(forfeit: { index: number; reasonKey: string }, teamNames: string[]): string {
  const loser = teamNames[forfeit.index] ?? `Player ${forfeit.index}`;
  const loserTeamLabel = forfeit.index < PLAYERS_PER_TEAM ? 'Team A' : 'Team B';
  const winnerTeamLabel = forfeit.index < PLAYERS_PER_TEAM ? 'Team B' : 'Team A';
  const reason = FORFEIT_REASONS[forfeit.reasonKey] ?? 'forfeited';
  return `${loser} (${loserTeamLabel}) ${reason}. ${winnerTeamLabel} wins by default.`;
}

export const antForagingArenaTransformer = (environment: any): ArenaStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 0', 'Player 1', 'Player 2', 'Player 3'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: ArenaStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);
    const privateObs = parsePerPlayer(step);

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

    // Skip the env's setup step where everyone submits -1.
    if (!players.some((pl) => pl.isTurn) && index > 0) return;

    const boards: (AntBoardView | null)[] = [pickBoardForTeam(privateObs, 0), pickBoardForTeam(privateObs, 1)];

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
    const isTerminal = observationTerminal || forfeit !== null;

    out.push({
      step: index,
      players,
      boards,
      privateObs,
      moveNumber: sample?.move_number ?? null,
      movesRemaining: sample?.moves_remaining ?? null,
      maxTurns: sample?.max_turns ?? null,
      activeSeat: sample?.active_seat ?? null,
      isTerminal,
      teamTotals: sample?.team_totals ?? null,
      returns: sample?.returns ?? null,
      winningTeam: sample?.winning_team ?? null,
      lastActionPerBoard,
      forfeitReason: forfeit ? buildArenaForfeitReason(forfeit, teamNames) : null,
    });
  });

  return out;
};
