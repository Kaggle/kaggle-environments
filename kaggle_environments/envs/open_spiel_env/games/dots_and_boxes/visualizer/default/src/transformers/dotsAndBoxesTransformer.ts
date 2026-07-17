// Dots and Boxes replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs so the
// right-hand Game Log can render each agent's reasoning, and parses the
// proxy's JSON observationString into a typed board state for the renderer.
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

interface DotsAndBoxesPlayer {
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

export interface DotsAndBoxesLastAction {
  orientation: 'h' | 'v';
  row: number;
  col: number;
  player: '1' | '2' | '';
}

export interface DotsAndBoxesBoardState {
  num_rows: number;
  num_cols: number;
  h_lines: number[][];
  v_lines: number[][];
  boxes: number[][];
  scores: [number, number];
  current_player: '1' | '2' | '';
  is_terminal: boolean;
  winner: '1' | '2' | 'draw' | null;
  last_action: DotsAndBoxesLastAction | null;
}

export interface DotsAndBoxesStep {
  step: number;
  players: DotsAndBoxesPlayer[];
  boardState: DotsAndBoxesBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function humanizeActionString(raw: string | null | undefined): string {
  if (!raw) return '';
  // OpenSpiel format: "P1(h,1,6)" / "P2(v,5,2)"
  const match = raw.match(/^P[12]\(([hv]),(\d+),(\d+)\)$/);
  if (!match) return raw;
  const [, orientation, row, col] = match;
  const kind = orientation === 'h' ? 'horizontal line' : 'vertical line';
  return `${kind} at row ${row}, col ${col}`;
}

function parseBoardState(step: OpenSpielRawPlayer[]): DotsAndBoxesBoardState | null {
  // Both agents see the same observationString; pick whichever is populated.
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as DotsAndBoxesBoardState;
  } catch {
    return null;
  }
}

export const dotsAndBoxesTransformer = (environment: any): DotsAndBoxesStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: DotsAndBoxesStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: DotsAndBoxesPlayer[] = step.map((p, i): DotsAndBoxesPlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // A forfeit step's offender has submission === -1 but should still be
      // treated as "acting" so the step is retained and their thoughts /
      // last-attempt render in the side panel.
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Player 1' : 'Player 2'),
        thumbnail: '',
        isTurn,
        actionDisplayText: humanizeActionString(p.action?.actionString),
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    // Skip steps where neither player acted (the env's setup step where both
    // submit -1). The side-panel uses isTurn to pick the active player.
    if (!players.some((pl) => pl.isTurn)) return;

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    out.push({
      step: index,
      players,
      boardState: parseBoardState(step),
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    });
  });

  return out;
};
