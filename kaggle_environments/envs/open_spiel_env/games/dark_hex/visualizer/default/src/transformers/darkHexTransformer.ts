// Dark Hex replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs so the
// right-hand Game Log can render each agent's reasoning. Dark Hex is
// imperfect-information, so each player sees their own board view --
// we surface both views as `boardX` / `boardO` for the renderer.
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

interface DarkHexPlayer {
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
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(player: OpenSpielRawPlayer | undefined): DarkHexBoardState | null {
  const raw = player?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as DarkHexBoardState;
  } catch {
    return null;
  }
}

export const darkHexTransformer = (environment: any): DarkHexStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player X', 'Player O'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: DarkHexStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: DarkHexPlayer[] = step.map((p, i): DarkHexPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender has submission === -1 but should still be
      // treated as "acting" so the step is retained and their thoughts /
      // last-attempt render in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Player X' : 'Player O'),
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

    const boardX = parseBoardState(step[0]);
    const boardO = parseBoardState(step[1]);
    // Dark Hex is private-info; the proxy emits `is_terminal` inside each
    // per-seat observation JSON in addition to the envelope-level flag.
    const observationTerminal = !!step[0]?.observation?.isTerminal || !!boardX?.is_terminal || !!boardO?.is_terminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    out.push({
      step: index,
      players,
      boardX,
      boardO,
      lastAction,
      lastActor,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    });
  });

  return out;
};
