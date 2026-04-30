import type { Action } from '../engine/types';
import type { AgentFn } from './types';

/** Minimal: each owned planet sends half its ships toward nearest non-owned planet. */
export const starterAgent: AgentFn = (obs) => {
  const moves: Action[] = [];
  const mine = obs.planets.filter((p) => p.owner === obs.player);
  const targets = obs.planets.filter((p) => p.owner !== obs.player);
  if (targets.length === 0) return moves;

  for (const src of mine) {
    if (src.ships < 10) continue;
    let best = targets[0];
    let bestD = Infinity;
    for (const t of targets) {
      const d = Math.hypot(t.x - src.x, t.y - src.y);
      if (d < bestD) {
        bestD = d;
        best = t;
      }
    }
    const angle = Math.atan2(best.y - src.y, best.x - src.x);
    const ships = Math.floor(src.ships / 2);
    moves.push([src.id, angle, ships]);
  }
  return moves;
};
