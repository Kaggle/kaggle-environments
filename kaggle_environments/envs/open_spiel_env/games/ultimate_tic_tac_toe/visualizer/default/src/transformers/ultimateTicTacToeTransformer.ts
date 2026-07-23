import {
  BaseGamePlayer,
  BaseGameStep,
  detectForfeit,
  buildForfeitReason,
  deriveWinnerFromRewards,
  parseThoughts,
  OpenSpielRawPlayer,
} from '@kaggle-environments/core';

export interface UltimateTicTacToeBoardState {
  board: string[][];
  subgrid_winners: string[];
  active_subgrid: number | null;
  phase: 'choose_subgrid' | 'choose_cell';
  current_player: 'x' | 'o';
  is_terminal: boolean;
  winner: 'x' | 'o' | 'draw' | null;
}

// Player shape extends the generic BaseGamePlayer with forfeit metadata
// carried per seat. Renderers can key off `forfeited` to badge the loser.
export interface UltimateTicTacToePlayer extends BaseGamePlayer {
  forfeited: boolean;
  forfeitLastAttempt: string | null;
}

export interface UltimateTicTacToeStep extends BaseGameStep {
  players: UltimateTicTacToePlayer[];
  boardState: UltimateTicTacToeBoardState | null;
  move?: {
    player: 'x' | 'o';
    subgridIdx: number;
    cellIdx: number | null;
  } | null;
  // isTerminal / winner reflect the terminal state including forfeits (which
  // OpenSpiel's own is_terminal flag doesn't). forfeitReason is non-null on
  // the final step iff the episode ended because a player forfeited.
  isTerminal: boolean;
  winner: string | null;
  forfeitReason: string | null;
}

function parseBoardState(step: OpenSpielRawPlayer[]): UltimateTicTacToeBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as UltimateTicTacToeBoardState;
  } catch {
    return null;
  }
}

function humanizeActionString(raw: string | null | undefined, playerLabel: string): string {
  if (!raw) return '';
  if (raw.startsWith('Choose local board')) {
    const parts = raw.split(' ');
    const subgridIdx = parseInt(parts[parts.length - 1]);
    return `${playerLabel} chose Sub-grid ${subgridIdx}`;
  }
  const match = raw.match(/Local board (\d+): [xo]\(([0-2]),([0-2])\)/);
  if (match) {
    const subgridIdx = parseInt(match[1]);
    const cellRow = parseInt(match[2]);
    const cellCol = parseInt(match[3]);
    const cellIdx = cellRow * 3 + cellCol;
    return `${playerLabel} placed at cell ${cellIdx} of Sub-grid ${subgridIdx}`;
  }
  return `${playerLabel}: ${raw}`;
}

export const ultimateTicTacToeTransformer = (environment: any): UltimateTicTacToeStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: OpenSpielRawPlayer[][] = environment?.steps ?? [];
  const out: UltimateTicTacToeStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);
    let activeMove: { player: 'x' | 'o'; subgridIdx: number; cellIdx: number | null } | null = null;

    const players: UltimateTicTacToePlayer[] = step.map((p, i): UltimateTicTacToePlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      const isTurn = (submission !== undefined && submission !== null && submission !== -1) || isForfeiter;
      const name = teamNames[i] ?? (i === 0 ? 'Player 1' : 'Player 2');
      const playerSymbol = i === 0 ? 'X' : 'O';
      const playerLabel = `${name} (${playerSymbol})`;

      if (isTurn && !isForfeiter) {
        const rawAction = p.action?.actionString || '';
        const prevRawAction = index > 0 ? rawSteps[index - 1][i]?.action?.actionString || '' : '';
        const isNewAction = rawAction !== '' && rawAction !== prevRawAction;

        if (isNewAction) {
          let subgridIdx = -1;
          let cellIdx: number | null = null;
          if (rawAction.startsWith('Choose local board')) {
            const parts = rawAction.split(' ');
            subgridIdx = parseInt(parts[parts.length - 1]);
          } else {
            const match = rawAction.match(/Local board (\d+): [xo]\(([0-2]),([0-2])\)/);
            if (match) {
              subgridIdx = parseInt(match[1]);
              const cellRow = parseInt(match[2]);
              const cellCol = parseInt(match[3]);
              cellIdx = cellRow * 3 + cellCol;
            }
          }
          if (subgridIdx !== -1) {
            activeMove = {
              player: i === 0 ? 'x' : 'o',
              subgridIdx,
              cellIdx,
            };
          }
        }
      }

      return {
        id: i,
        name,
        thumbnail: '',
        isTurn,
        actionDisplayText: humanizeActionString(p.action?.actionString, playerLabel),
        thoughts: parseThoughts(p.action),
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    // Skip steps where neither player acted (the env's setup steps).
    if (!players.some((pl) => pl.isTurn)) return;

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    const isTerminal = observationTerminal || forfeit !== null;

    out.push({
      step: index,
      players,
      boardState: parseBoardState(step),
      move: activeMove,
      isTerminal,
      winner: isTerminal ? deriveWinnerFromRewards(step, teamNames) : null,
      forfeitReason: forfeit ? buildForfeitReason(forfeit, teamNames) : null,
    });
  });

  return out;
};
