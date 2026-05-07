import type { RendererOptions } from '@kaggle-environments/core';
import { getStepData } from '@kaggle-environments/core';
import { buildShell, collectRefs, renderObservation, type BoardSize, type LayoutRefs } from './renderFarm';

const DEFAULT_BOARD: BoardSize = { rows: 10, cols: 10 };

interface CachedShell {
  key: string;
  refs: LayoutRefs;
}

const shellCache = new WeakMap<HTMLElement, CachedShell>();

function inferBoardSize(replay: any): BoardSize {
  const cfg = replay?.configuration ?? {};
  const rows = Number(cfg.height) || DEFAULT_BOARD.rows;
  const cols = Number(cfg.width) || DEFAULT_BOARD.cols;
  return { rows, cols };
}

function getObservation(replay: any, step: number): any | null {
  const stepData = getStepData(replay, step) as any;
  if (!stepData) return null;
  const agentStep = Array.isArray(stepData) ? stepData[0] : stepData;
  return agentStep?.observation ?? null;
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

  const obs = getObservation(replay, step);
  if (!obs) return;
  renderObservation(cached.refs, obs);
}
