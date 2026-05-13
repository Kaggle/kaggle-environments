import type { RendererOptions } from '@kaggle-environments/core';
import { getStepData } from '@kaggle-environments/core';
import { buildShell, collectRefs, renderObservation, type BoardSize, type LayoutRefs } from './renderFarm';
import type { ViewModel, PrivateState } from './types';

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

function buildView(replay: any, step: number): ViewModel | null {
  const stepData = getStepData(replay, step) as any;
  if (!stepData) return null;
  const entries = Array.isArray(stepData) ? stepData : [stepData];
  const obs0 = entries[0]?.observation;
  if (!obs0) return null;

  const privates: (PrivateState | undefined)[] = [];
  for (const entry of entries) {
    const obs = entry?.observation;
    const idx = typeof obs?.player === 'number' ? obs.player : privates.length;
    privates[idx] = obs?.private;
  }

  return {
    day: Number(obs0.day ?? 0),
    hour: Number(obs0.hour ?? 0),
    farms: obs0.farms ?? [],
    market: obs0.market ?? { prices: {}, inventory: {} },
    town: obs0.town ?? { unlocked_shops: [] },
    privates,
  };
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

  const view = buildView(replay, step);
  if (!view) return;
  renderObservation(cached.refs, view, replay?.configuration ?? {});
}
