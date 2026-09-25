import type { RendererOptions } from '@kaggle-environments/core';
import { buildShell, renderPanels, type PanelRefs } from './panels';
import { chipText, drawPipeline, type ChipHit } from './pipeline';
import type { StepView } from './types';
import { buildView } from './utils';

// The shell is torn down and rebuilt only when the host swaps replays, so
// scrubbing the slider just updates text and repaints the canvases.
const shellCache = new WeakMap<HTMLElement, PanelRefs>();

/** The chips and view the board last drew, read by the hover handler. */
const drawn = new WeakMap<PanelRefs, { hits: ChipHit[]; view: StepView }>();

function attachHover(refs: PanelRefs) {
  const hide = () => refs.tip.classList.remove('open');
  refs.canvas.addEventListener('mouseleave', hide);
  refs.canvas.addEventListener('mousemove', (e) => {
    const last = drawn.get(refs);
    const hit = last?.hits.find((h) => Math.hypot(h.x - e.offsetX, h.y - e.offsetY) <= h.r + 2);
    if (!last || !hit) return hide();
    refs.tip.textContent = chipText(hit.asset, last.view.players[hit.playerIdx]?.name ?? '');
    // The canvas is centred in the board, so offset by where it sits. Chips on
    // the right half open leftward so the card never runs off the edge.
    const flip = hit.x > refs.canvas.clientWidth / 2;
    const gap = hit.r + 6;
    refs.tip.style.left = `${refs.canvas.offsetLeft + hit.x + (flip ? -gap : gap)}px`;
    refs.tip.style.transform = flip ? 'translateX(-100%)' : '';
    refs.tip.style.top = `${refs.canvas.offsetTop + hit.y - 8}px`;
    refs.tip.classList.add('open');
  });
}

export function renderer(options: RendererOptions): void {
  const { parent, replay, step } = options;
  if (!parent || !replay) return;

  let refs = shellCache.get(parent);
  // On a significant resize the host removes every <canvas> under the container
  // (replay-adapter.ts) without touching the rest of the DOM, so check the
  // canvases are still attached, not just the root.
  if (!refs || !parent.contains(refs.root) || !refs.root.contains(refs.canvas) || !refs.root.contains(refs.spark)) {
    refs = buildShell(parent);
    attachHover(refs);
    shellCache.set(parent, refs);
  }

  const view = buildView(replay, step);
  if (!view) return;

  renderPanels(refs, view);
  drawn.set(refs, { hits: drawPipeline(refs.canvas, view.players), view });
  refs.tip.classList.remove('open');
}
