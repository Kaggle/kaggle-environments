/**
 * Starter agent — direct port of `starter_agent` from kaggriculture.py.
 *
 * Single-tile carrot loop: sell any carrots in the shed, keep a seed on hand,
 * then PLANT → WATER → HARVEST on the tile the farmer is standing on.
 */

import { CROPS } from '../engine/constants';
import type { MarketOrder, PlantTile, PlayerAction, UnitAction } from '../engine/types';
import type { AgentFn, Observation } from './types';

export const starterAgent: AgentFn = (obs: Observation): PlayerAction => {
  const farm = obs.farms[obs.player];
  if (!farm) return { farmer: ['PASS'], hands: [], market: [] };

  const [fx, fy] = farm.farmer;
  const tile = farm.tiles[fy][fx];
  const seeds = obs.private.seeds;
  const shed = obs.private.shed;

  const market: MarketOrder[] = [];
  const carrotShed = shed.CARROT ?? 0;
  if (carrotShed > 0) market.push(['SELL', 'CARROT', carrotShed]);
  if ((seeds.CARROT ?? 0) === 0 && farm.money >= CROPS.CARROT.seed) {
    market.push(['BUY_SEED', 'CARROT', 1]);
  }

  let farmer: UnitAction = ['PASS'];
  if (tile === null && (seeds.CARROT ?? 0) > 0) {
    farmer = ['PLANT', 'CARROT'];
  } else if (isCarrotPlant(tile)) {
    const age = obs.day - tile.planted_day;
    if (age >= CROPS.CARROT.max_yield_day) {
      farmer = ['HARVEST'];
    } else if (!tile.watered_today) {
      farmer = ['WATER'];
    }
  }

  return { farmer, hands: [], market };
};

function isCarrotPlant(tile: unknown): tile is PlantTile {
  return (
    typeof tile === 'object' &&
    tile !== null &&
    (tile as { kind?: string }).kind === 'PLANT' &&
    (tile as { crop?: string }).crop === 'CARROT'
  );
}
