import { BOARD_SIZE } from '../engine/constants';
import type { GameState, Planet } from '../engine/types';

export interface BoardCoord {
  x: number;
  y: number;
}

export function screenToBoard(canvas: HTMLCanvasElement, evt: { clientX: number; clientY: number }): BoardCoord {
  const r = canvas.getBoundingClientRect();
  const scale = r.width / BOARD_SIZE;
  return {
    x: (evt.clientX - r.left) / scale,
    y: (evt.clientY - r.top) / scale,
  };
}

/** Returns the planet under (bx, by), with 0.5-unit click tolerance. Comets can be selected too. */
export function planetAt(state: GameState, bx: number, by: number): Planet | null {
  for (const p of state.planets) {
    const dx = bx - p.x;
    const dy = by - p.y;
    const r = p.radius + 0.5;
    if (dx * dx + dy * dy <= r * r) return p;
  }
  return null;
}
