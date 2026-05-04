import type { Action } from '../engine/types';
import type { AgentFn } from './types';

export const randomAgent: AgentFn = (obs) => {
  const moves: Action[] = [];
  const mine = obs.planets.filter((p) => p.owner === obs.player && p.ships >= 5);
  if (mine.length === 0) return moves;
  // 50% chance of doing nothing
  if (Math.random() < 0.5) return moves;
  const src = mine[Math.floor(Math.random() * mine.length)];
  const angle = Math.random() * Math.PI * 2;
  const ships = Math.max(1, Math.floor(src.ships * (0.25 + Math.random() * 0.5)));
  moves.push([src.id, angle, ships]);
  return moves;
};
