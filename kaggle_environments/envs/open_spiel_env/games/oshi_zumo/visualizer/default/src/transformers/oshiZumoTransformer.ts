// Oshi-Zumo replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs. Oshi-Zumo is a
// simultaneous-move game: each non-setup step has both players acting, so we
// surface both players' bids, rewards, and reasoning to the right-hand Game
// Log. Mirrors the amazons transformer pattern.
//
// Forfeit handling (illegal-move / TIMEOUT / ERROR) is delegated to the
// shared helpers in @kaggle-environments/core so that every OpenSpiel game
// labels early terminations the same way.

import {
  detectForfeit,
  buildForfeitReason,
  deriveWinnerFromRewards,
  parseThoughts,
  OpenSpielRawPlayer,
} from '@kaggle-environments/core';

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
  forfeited: boolean;
  forfeitLastAttempt: string | null;
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
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): OshiZumoBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as OshiZumoBoardState;
  } catch {
    return null;
  }
}

export const oshiZumoTransformer = (environment: any): OshiZumoStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: OshiZumoStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);
    const bids: [number | null, number | null] = [null, null];

    const playerInfo = step.map((p, i) => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // Setup step / inactive: submission is -1 or undefined.
      const acted = submission !== undefined && submission !== null && submission !== -1;
      const bid = acted ? (submission as number) : null;
      bids[i] = bid;
      return {
        id: i,
        name: teamNames[i] ?? `Player ${i + 1}`,
        acted,
        isForfeiter,
        // Forfeiter submits -1 but should still get a log entry so their
        // thoughts / last attempt render in the sidebar.
        emit: acted || isForfeiter,
        bid,
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        actionString: p.action?.actionString ?? '',
      };
    });

    // Skip the setup step (neither player acted and no forfeit).
    if (!playerInfo.some((pl) => pl.emit)) return;

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    const isTerminal = observationTerminal || forfeit !== null;
    const boardState = parseBoardState(step);
    const winner = isTerminal ? deriveWinnerFromRewards(step, teamNames) : null;
    const forfeitReason = forfeit ? buildForfeitReason(forfeit, teamNames) : null;

    // Emit one log entry per player who acted (or forfeited), so the
    // side-panel Game Log surfaces both players' bids and reasoning. Each
    // entry shares the same post-bid board state; only the active-player
    // flag differs.
    playerInfo.forEach((info, focusIdx) => {
      if (!info.emit) return;
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
        forfeited: pi.isForfeiter,
        forfeitLastAttempt: pi.isForfeiter ? pi.actionString || null : null,
      }));
      out.push({
        step: index,
        players,
        boardState,
        bids,
        isTerminal,
        winner,
        forfeitReason,
      });
    });
  });

  return out;
};
