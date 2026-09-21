import type { RendererOptions } from '@kaggle-environments/core';
import { buildShell, renderPanels, type PanelRefs } from './panels';
import { drawPipeline } from './pipeline';
import { buildView } from './utils';

// The shell is torn down and rebuilt only when the host swaps replays, so
// scrubbing the slider just updates text and repaints the canvases.
const shellCache = new WeakMap<HTMLElement, PanelRefs>();

export function renderer(options: RendererOptions): void {
  const { parent, replay, step } = options;
  if (!parent || !replay) return;

  let refs = shellCache.get(parent);
  // On a significant resize the host removes every <canvas> under the container
  // (replay-adapter.ts) without touching the rest of the DOM, so check the
  // canvases are still attached, not just the root.
  if (!refs || !parent.contains(refs.root) || !refs.root.contains(refs.canvas) || !refs.root.contains(refs.spark)) {
    refs = buildShell(parent);
    shellCache.set(parent, refs);
  }

  const view = buildView(replay, step);
  if (!view) return;

  renderPanels(refs, view);
  drawPipeline(refs.canvas, view.players);
}
