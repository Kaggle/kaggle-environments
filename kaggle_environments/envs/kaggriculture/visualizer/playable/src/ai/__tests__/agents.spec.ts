import { describe, expect, it, vi } from 'vitest';
import { CROPS } from '../../engine/constants';
import { initGameState, newPlant, resolveConfig } from '../../engine/state';
import type { GameState, PlantTile } from '../../engine/types';
import { randomAgent } from '../random';
import { starterAgent } from '../starter';
import type { Observation } from '../types';

const cfg = resolveConfig();

function obsFor(state: GameState, player: number): Observation {
  return {
    player,
    step: state.step,
    day: state.day,
    hour: state.hour,
    numAgents: state.numAgents,
    farms: state.farms,
    private: state.privates[player],
    market: state.market,
    town: state.town,
  };
}

describe('randomAgent', () => {
  it('returns a well-shaped PlayerAction', () => {
    const s = initGameState(2, cfg, 1);
    const a = randomAgent(obsFor(s, 0));
    expect(Array.isArray(a.farmer)).toBe(true);
    expect(Array.isArray(a.hands)).toBe(true);
    expect(a.hands).toHaveLength(0); // no hands yet
    expect(Array.isArray(a.market)).toBe(true);
  });

  it('emits a PLANT when seeds are present (sampling controlled via Math.random)', () => {
    const s = initGameState(2, cfg, 1);
    s.privates[0].seeds.WHEAT = 1;
    // Force the "plant" branch: 0.0 < 0.1 (skip BUY_SEED) and 0.0 < 0.3 (PLANT).
    // pick() = items[floor(0.0 * len)] = items[0].
    const spy = vi.spyOn(Math, 'random').mockReturnValue(0.0);
    try {
      const a = randomAgent(obsFor(s, 0));
      expect(a.market[0]).toEqual(['BUY_SEED', 'WHEAT', 1]); // affordable, first item
      expect(a.farmer[0]).toBe('PLANT');
      expect(a.farmer[1]).toBe('WHEAT');
    } finally {
      spy.mockRestore();
    }
  });

  it('emits a farmer op (no PLANT) when no seeds are owned', () => {
    const s = initGameState(2, cfg, 1);
    // Force "no buy" + "no PLANT branch": 0.99 > 0.1 and 0.99 > 0.3.
    const spy = vi.spyOn(Math, 'random').mockReturnValue(0.99);
    try {
      const a = randomAgent(obsFor(s, 0));
      expect(a.market).toEqual([]);
      expect(['NORTH', 'SOUTH', 'EAST', 'WEST', 'WATER', 'HARVEST', 'PASS']).toContain(a.farmer[0]);
    } finally {
      spy.mockRestore();
    }
  });

  it('produces one op per existing hand', () => {
    const s = initGameState(2, cfg, 1);
    s.farms[0].hands = [
      [5, 4],
      [4, 5],
    ];
    const a = randomAgent(obsFor(s, 0));
    expect(a.hands).toHaveLength(2);
  });
});

describe('starterAgent', () => {
  it('buys a CARROT seed when none owned and money allows', () => {
    const s = initGameState(2, cfg, 1);
    const a = starterAgent(obsFor(s, 0));
    expect(a.market).toContainEqual(['BUY_SEED', 'CARROT', 1]);
    expect(a.farmer).toEqual(['PASS']); // empty tile but no seed yet
  });

  it('plants CARROT on an empty tile once a seed is in hand', () => {
    const s = initGameState(2, cfg, 1);
    s.privates[0].seeds.CARROT = 1;
    const a = starterAgent(obsFor(s, 0));
    expect(a.farmer).toEqual(['PLANT', 'CARROT']);
    expect(a.market).not.toContainEqual(['BUY_SEED', 'CARROT', 1]); // already have one
  });

  it('waters its CARROT plant when not yet watered today', () => {
    const s = initGameState(2, cfg, 1);
    const [fx, fy] = s.farms[0].farmer;
    s.farms[0].tiles[fy][fx] = newPlant('CARROT', 0, cfg.turnsPerDay);
    const a = starterAgent(obsFor(s, 0));
    expect(a.farmer).toEqual(['WATER']);
  });

  it('harvests its CARROT plant once age >= max_yield_day', () => {
    const s = initGameState(2, cfg, 1);
    const [fx, fy] = s.farms[0].farmer;
    const tile = newPlant('CARROT', 0, cfg.turnsPerDay);
    tile.watered_today = true;
    s.farms[0].tiles[fy][fx] = tile;
    s.day = CROPS.CARROT.max_yield_day;
    const a = starterAgent(obsFor(s, 0));
    expect(a.farmer).toEqual(['HARVEST']);
  });

  it('sells any CARROT sitting in the shed', () => {
    const s = initGameState(2, cfg, 1);
    s.privates[0].shed.CARROT = 4;
    const a = starterAgent(obsFor(s, 0));
    expect(a.market).toContainEqual(['SELL', 'CARROT', 4]);
  });

  it('ignores non-CARROT plant tiles (just PASS)', () => {
    const s = initGameState(2, cfg, 1);
    const [fx, fy] = s.farms[0].farmer;
    s.farms[0].tiles[fy][fx] = newPlant('WHEAT', 0, cfg.turnsPerDay) as PlantTile;
    s.privates[0].seeds.CARROT = 1;
    const a = starterAgent(obsFor(s, 0));
    // Tile is non-null and not a CARROT plant, so no PLANT and no WATER/HARVEST.
    expect(a.farmer).toEqual(['PASS']);
  });
});
