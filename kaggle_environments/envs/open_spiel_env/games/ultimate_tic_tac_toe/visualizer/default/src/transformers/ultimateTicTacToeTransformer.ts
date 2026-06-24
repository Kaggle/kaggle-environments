import { BaseGamePlayer, BaseGameStep } from '@kaggle-environments/core';

export interface UltimateTicTacToeBoardState {
  board: string[][];
  subgrid_winners: string[];
  active_subgrid: number | null;
  phase: 'choose_subgrid' | 'choose_cell';
  current_player: 'x' | 'o';
  is_terminal: boolean;
  winner: 'x' | 'o' | 'draw' | null;
}

export interface UltimateTicTacToeStep extends BaseGameStep {
  boardState: UltimateTicTacToeBoardState | null;
  move?: {
    player: 'x' | 'o';
    subgridIdx: number;
    cellIdx: number | null;
  } | null;
}

function parseThoughts(action?: any): string {
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      // fall through
    }
  }
  return action?.thoughts ?? '';
}

function parseBoardState(step: any[]): UltimateTicTacToeBoardState | null {
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
    const row = Math.floor(subgridIdx / 3);
    const col = subgridIdx % 3;
    return `${playerLabel} chose Sub-grid [Row ${row}, Col ${col}]`;
  }
  const match = raw.match(/Local board (\d+): [xo]\(([0-2]),([0-2])\)/);
  if (match) {
    const subgridIdx = parseInt(match[1]);
    const cellRow = parseInt(match[2]);
    const cellCol = parseInt(match[3]);
    const subRow = Math.floor(subgridIdx / 3);
    const subCol = subgridIdx % 3;
    return `${playerLabel} placed at [Row ${cellRow}, Col ${cellCol}] of Sub-grid [Row ${subRow}, Col ${subCol}]`;
  }
  return `${playerLabel}: ${raw}`;
}

export const ultimateTicTacToeTransformer = (environment: any): UltimateTicTacToeStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: any[][] = environment?.steps ?? [];
  const out: UltimateTicTacToeStep[] = [];

  rawSteps.forEach((step, index) => {
    let activeMove: { player: 'x' | 'o'; subgridIdx: number; cellIdx: number | null } | null = null;

    const players: BaseGamePlayer[] = step.map((p, i): BaseGamePlayer => {
      const submission = p.action?.submission;
      const isTurn = submission !== undefined && submission !== -1;
      const name = teamNames[i] ?? (i === 0 ? 'Player 1' : 'Player 2');
      const playerSymbol = i === 0 ? 'X' : 'O';
      const playerLabel = `${name} (${playerSymbol})`;

      if (isTurn) {
        const rawAction = p.action?.actionString || '';
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

      return {
        id: i,
        name,
        thumbnail: '',
        isTurn,
        actionDisplayText: humanizeActionString(p.action?.actionString, playerLabel),
        thoughts: parseThoughts(p.action),
      };
    });

    // Skip steps where neither player acted (the env's setup steps).
    if (!players.some((pl) => pl.isTurn)) return;

    out.push({
      step: index,
      players,
      boardState: parseBoardState(step),
      move: activeMove,
    });
  });

  return out;
};
