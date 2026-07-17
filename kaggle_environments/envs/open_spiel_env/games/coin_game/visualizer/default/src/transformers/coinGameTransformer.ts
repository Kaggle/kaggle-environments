// Coin Game replay transformer.
//
// Builds the per-step `players` array the side-panel UI expects so the
// right-hand Game Log can render each agent's reasoning, mirroring the
// dark_hex / Y / Lines of Action transformers.
//
// Forfeit handling (illegal-move / TIMEOUT / ERROR) is delegated to the
// shared helpers in @kaggle-environments/core so that every OpenSpiel game
// labels early terminations the same way.

import { detectForfeit, buildForfeitReason, parseThoughts, OpenSpielRawPlayer } from '@kaggle-environments/core';

interface CoinPlayer {
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
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parsePerPlayer(step: OpenSpielRawPlayer[]): (CoinBoardState | null)[] {
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
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: CoinStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: CoinPlayer[] = step.map((p, i): CoinPlayer => {
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
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
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

    const observationTerminal = !!step[0]?.observation?.isTerminal || !!boardState?.is_terminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    // On a natural terminal, expose the game's own winner (a numeric id
    // that the renderer indexes into `playerNames`). On a forfeit, leave
    // it null: the renderer has a dedicated forfeit-fallback branch that
    // derives the winner from `forfeiterIdx`, and emitting a string here
    // would break the sibling `Number(stepWinner)` code path.
    const winner: number | string | null = isTerminal && !forfeit ? (boardState?.winner ?? null) : null;

    out.push({
      step: index,
      players,
      boardState,
      privateObs,
      lastActor: lastActor >= 0 ? lastActor : null,
      lastAction,
      isTerminal,
      winner,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    });
  });

  return out;
};
