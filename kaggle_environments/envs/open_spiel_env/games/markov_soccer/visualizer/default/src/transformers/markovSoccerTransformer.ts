// Markov Soccer replay transformer.
//
// Populates `players[]` so the side-panel can show each mover's thoughts
// and last action, and pre-parses the observation JSON into `boardState`.
// Markov Soccer is a simultaneous-move game — both seats submit a real
// action every round, so both players are `isTurn: true` on played steps.
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

// Extended raw player shape carrying the harness-emitted `info` payload used
// for the display string. Core's OpenSpielRawPlayer doesn't model `info`.
interface SoccerRawPlayer extends OpenSpielRawPlayer {
  info?: { actionSubmittedToString?: string | null };
}

interface SoccerPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  forfeited: boolean;
  forfeitLastAttempt: string | null;
}

type Pos = [number, number] | null;

export interface SoccerBoardState {
  board: string[][];
  current_player: string;
  is_terminal: boolean;
  winner: 'A' | 'B' | 'draw' | null;
  player_a_pos: Pos;
  player_b_pos: Pos;
  ball_pos: Pos;
  ball_owner: 'A' | 'B' | null;
}

export interface SoccerStep {
  step: number;
  players: SoccerPlayer[];
  boardState: SoccerBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  // Non-null on the terminal step when the game ended because a player
  // forfeited (illegal-move retries exhausted, timeout, or crash). The
  // renderer surfaces this instead of the normal "X wins!" line so the
  // reason for the early end is clear.
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): SoccerBoardState | null {
  const raw = step?.[0]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as SoccerBoardState;
  } catch {
    return null;
  }
}

const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const markovSoccerTransformer = (environment: any): SoccerStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: SoccerRawPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): SoccerStep => {
    const forfeit = detectForfeit(step);

    const players: SoccerPlayer[] = step.map((p, i): SoccerPlayer => {
      const isForfeiter = forfeit?.index === i;
      return {
        id: i,
        name: teamNames[i] || (i === 0 ? 'Player A' : 'Player B'),
        thumbnail: '',
        // Forfeiter submits -1 but should still be treated as acting so
        // their thoughts / last attempt render in the sidebar.
        isTurn: isRealMove(p.action?.submission) || isForfeiter,
        // info.actionSubmittedToString is what the renderer historically
        // displayed. Fall back to action.actionString when info is absent.
        actionDisplayText: p.info?.actionSubmittedToString ?? p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    const boardState = parseBoardState(step);
    const observationTerminal = !!step[0]?.observation?.isTerminal || !!boardState?.is_terminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal; treat it as terminal so downstream UI shows the end state.
    const isTerminal = observationTerminal || forfeit !== null;

    return {
      step: index,
      players,
      boardState,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    };
  });
};
