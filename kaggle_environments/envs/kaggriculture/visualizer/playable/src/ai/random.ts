/**
 * Random agent — direct port of `random_agent` from kaggriculture.py.
 * Picks farmer ops uniformly, occasionally buys a seed, occasionally plants.
 *
 * Each hand also gets a uniform op (no PLANT for hands, matching Python).
 */

import { CROPS } from '../engine/constants';
import type { CropId, MarketOrder, PlayerAction, UnitAction } from '../engine/types';
import type { AgentFn, Observation } from './types';

type FarmerOp = 'NORTH' | 'SOUTH' | 'EAST' | 'WEST' | 'WATER' | 'HARVEST' | 'PASS';
const FARMER_OPS: FarmerOp[] = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'WATER', 'HARVEST', 'PASS'];

function pick<T>(items: readonly T[]): T {
  return items[Math.floor(Math.random() * items.length)];
}

export const randomAgent: AgentFn = (obs: Observation): PlayerAction => {
  const farm = obs.farms[obs.player];
  if (!farm) return { farmer: ['PASS'], hands: [], market: [] };

  const seeds = obs.private.seeds;
  const market: MarketOrder[] = [];

  const affordable = (Object.keys(CROPS) as CropId[]).filter((c) => CROPS[c].seed <= farm.money);
  if (affordable.length > 0 && Math.random() < 0.1) {
    market.push(['BUY_SEED', pick(affordable), 1]);
  }

  const availableSeeds = (Object.keys(seeds) as CropId[]).filter((c) => (seeds[c] ?? 0) > 0);
  let farmer: UnitAction;
  if (availableSeeds.length > 0 && Math.random() < 0.3) {
    farmer = ['PLANT', pick(availableSeeds)];
  } else {
    farmer = [pick(FARMER_OPS)];
  }

  const hands: UnitAction[] = farm.hands.map(() => [pick(FARMER_OPS)] as UnitAction);

  return { farmer, hands, market };
};
