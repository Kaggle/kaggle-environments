import type { RendererOptions } from '@kaggle-environments/core';
import { buildShell, collectRefs, renderObservation, type BoardSize, type LayoutRefs } from './renderFarm';
import { buildView } from './utils';

const DEFAULT_BOARD: BoardSize = { rows: 10, cols: 10 };

interface CachedShell {
  key: string;
  refs: LayoutRefs;
}

const shellCache = new WeakMap<HTMLElement, CachedShell>();

function inferBoardSize(replay: any): BoardSize {
  const cfg = replay?.configuration ?? {};
  const size = Number(cfg.boardSize);
  if (Number.isFinite(size) && size > 0) {
    return { rows: size, cols: size };
  }
  return DEFAULT_BOARD;
}

export function renderer(options: RendererOptions): void {
  const { parent, replay, step, agents } = options;
  if (!parent || !replay) return;

  const board = inferBoardSize(replay);
  const playerNames = [0, 1].map((i) => agents?.[i]?.name || `Player ${i + 1}`);
  const expectedKey = `${board.rows}x${board.cols}|${playerNames.join('|')}`;
  let cached = shellCache.get(parent);
  if (!cached || cached.key !== expectedKey) {
    buildShell(parent, board, playerNames);
    cached = { key: expectedKey, refs: collectRefs(parent, board) };
    shellCache.set(parent, cached);
  }

  const cfg = replay?.configuration ?? {};
  const turnsPerDay = Math.max(1, Number(cfg.turnsPerDay) || 24);
  const view = buildView(replay, step, turnsPerDay);
  if (!view) return;
  renderObservation(cached.refs, view, cfg);
}
