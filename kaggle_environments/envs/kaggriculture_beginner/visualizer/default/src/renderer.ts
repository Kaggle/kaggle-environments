import type { RendererOptions } from '@kaggle-environments/core';
import { getStepData } from '@kaggle-environments/core';
import { buildSkeleton, collectRefs, renderObservation, type BoardSize, type LayoutRefs } from './renderFarm';

const DEFAULT_BOARD: BoardSize = { rows: 5, cols: 5 };

interface CachedSkeleton {
  key: string;
  refs: LayoutRefs;
}

const skeletonCache = new WeakMap<HTMLElement, CachedSkeleton>();

function inferBoardSize(replay: any): BoardSize {
  const cfg = replay?.configuration ?? {};
  const size = Number(cfg.boardSize) || DEFAULT_BOARD.rows;
  return { rows: size, cols: size };
}

function getObservation(replay: any, step: number): any | null {
  const stepData = getStepData(replay, step) as any;
  if (!stepData) return null;
  const agentStep = Array.isArray(stepData) ? stepData[0] : stepData;
  return agentStep?.observation ?? null;
}

export function renderer(options: RendererOptions): void {
  const { parent, replay, step } = options;
  if (!parent || !replay) return;

  const board = inferBoardSize(replay);
  const expectedKey = `${board.rows}x${board.cols}`;
  let cached = skeletonCache.get(parent);
  if (!cached || cached.key !== expectedKey) {
    buildSkeleton(parent, board);
    cached = { key: expectedKey, refs: collectRefs(parent, board) };
    skeletonCache.set(parent, cached);
  }

  const obs = getObservation(replay, step);
  if (!obs) return;
  renderObservation(cached.refs, obs);
}
