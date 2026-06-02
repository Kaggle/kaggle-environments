import { describe, expect, it } from 'vitest';
import { ANIMALS, CROPS, MARKET_PARAMS, PRODUCTS, DEFAULT_CONFIG } from '../constants';
import { marketPrice, refreshPrices, resolveMarketParams, shape } from '../market';
import {
  defaultSpawn,
  initGameState,
  isShedAdjacent,
  newPrivate,
  quadrantOf,
  resolveConfig,
  shedAccessTiles,
  spawnHand,
} from '../state';

describe('quadrants', () => {
  it('splits a 10x10 board into NW/NE/SW/SE around the half line', () => {
    expect(quadrantOf(0, 0, 10)).toBe('NW');
    expect(quadrantOf(4, 4, 10)).toBe('NW');
    expect(quadrantOf(5, 4, 10)).toBe('NE');
    expect(quadrantOf(4, 5, 10)).toBe('SW');
    expect(quadrantOf(5, 5, 10)).toBe('SE');
  });
});

describe('shed access', () => {
  it('returns the four inner corners around the shed in NWSE order', () => {
    expect(shedAccessTiles(10)).toEqual([
      [4, 4],
      [5, 4],
      [4, 5],
      [5, 5],
    ]);
  });
  it('default spawn picks the NW shed-access tile', () => {
    expect(defaultSpawn(10)).toEqual([4, 4]);
  });
  it('isShedAdjacent matches the four corners only', () => {
    expect(isShedAdjacent([4, 4], 10)).toBe(true);
    expect(isShedAdjacent([5, 5], 10)).toBe(true);
    expect(isShedAdjacent([3, 4], 10)).toBe(false);
    expect(isShedAdjacent([4, 3], 10)).toBe(false);
  });
});

describe('initGameState', () => {
  const cfg = resolveConfig();
  const state = initGameState(2, cfg, 12345);

  it('records the resolved seed and starts at step/day/hour 0', () => {
    expect(state.seed).toBe(12345);
    expect(state.step).toBe(0);
    expect(state.day).toBe(0);
    expect(state.hour).toBe(0);
    expect(state.done).toBe(false);
  });

  it('gives each player a farm with starting money and only NW unlocked', () => {
    expect(state.farms).toHaveLength(2);
    for (const farm of state.farms) {
      expect(farm.money).toBe(DEFAULT_CONFIG.startingMoney);
      expect(farm.unlocked_quadrants).toEqual(['NW']);
      expect(farm.farmer).toEqual([4, 4]);
      expect(farm.hands).toEqual([]);
      expect(farm.hires_today).toBe(0);
    }
  });

  it('marks all non-NW tiles as LOCKED and NW tiles as empty', () => {
    const tiles = state.farms[0].tiles;
    expect(tiles).toHaveLength(10);
    expect(tiles[0]).toHaveLength(10);
    for (let y = 0; y < 10; y++) {
      for (let x = 0; x < 10; x++) {
        const expected = quadrantOf(x, y, 10) === 'NW' ? null : 'LOCKED';
        expect(tiles[y][x]).toBe(expected);
      }
    }
  });

  it('initializes the market at I0 with base prices for every product', () => {
    for (const item of PRODUCTS) {
      expect(state.market.inventory[item]).toBe(MARKET_PARAMS[item].I0);
      expect(state.market.prices[item]).toBe(MARKET_PARAMS[item].base);
    }
    expect(state.market.params).toBeUndefined();
  });

  it('starts town with no unlocked shops', () => {
    expect(state.town.unlocked_shops).toEqual([]);
  });

  it('initializes each private with zero shed/seeds and one (empty) inventory slot', () => {
    for (const priv of state.privates) {
      expect(priv.inventories).toEqual([{}]);
      for (const crop of Object.keys(CROPS)) expect(priv.seeds[crop as keyof typeof CROPS]).toBe(0);
      for (const item of PRODUCTS) expect(priv.shed[item]).toBe(0);
      for (const animal of Object.keys(ANIMALS)) expect(priv.shed[animal as keyof typeof ANIMALS]).toBe(0);
    }
  });
});

describe('market pricing', () => {
  it('prices at I0 equal the base price', () => {
    for (const item of PRODUCTS) {
      expect(marketPrice(item, MARKET_PARAMS[item].I0)).toBe(MARKET_PARAMS[item].base);
    }
  });

  it('selling units below I0 drives price upward, buying above drives it down', () => {
    const item = 'WHEAT';
    const I0 = MARKET_PARAMS[item].I0;
    expect(marketPrice(item, I0 - 100)).toBeGreaterThan(MARKET_PARAMS[item].base);
    expect(marketPrice(item, I0 + 100)).toBeLessThan(MARKET_PARAMS[item].base);
  });

  it('floors at PRICE_FLOOR for arbitrarily large inventories', () => {
    // STRAWBERRY uses linear above-I0 with a steep slope so it'll bottom out fast.
    expect(marketPrice('STRAWBERRY', 1_000_000_000)).toBe(1);
  });

  it('shape() implements the expected function set with non-negative input', () => {
    expect(shape('linear', 4)).toBe(4);
    expect(shape('sq', 3)).toBe(9);
    expect(shape('sqrt', 16)).toBe(4);
    expect(shape('log', 0)).toBe(0); // ln(1+0) = 0
    expect(shape('log10', 9)).toBeCloseTo(1, 10);
    expect(shape('linear', -5)).toBe(0); // clamps negatives
  });

  it('refreshPrices recomputes from current inventory', () => {
    const cfg = resolveConfig();
    const state = initGameState(2, cfg, 1);
    state.market.inventory.WHEAT -= 50;
    refreshPrices(state.market);
    expect(state.market.prices.WHEAT).toBe(marketPrice('WHEAT', state.market.inventory.WHEAT));
  });

  it('resolveMarketParams clones defaults and applies sparse overrides', () => {
    const resolved = resolveMarketParams({ WHEAT: { base: 100 } });
    expect(resolved.WHEAT.base).toBe(100);
    expect(resolved.WHEAT.T).toBe(MARKET_PARAMS.WHEAT.T);
    expect(MARKET_PARAMS.WHEAT.base).toBe(25); // defaults untouched
  });
});

describe('spawnHand', () => {
  it('places new hands in NWSE order, preferring less-occupied tiles', () => {
    const priv = newPrivate(); // unused but verifies it builds
    const cfg = resolveConfig();
    const state = initGameState(2, cfg, 0);
    const farm = state.farms[0];
    expect(priv).toBeDefined();
    expect(spawnHand(farm, 10)).toEqual([5, 4]); // [4,4] is the farmer; NWSE pick the next free
    farm.hands.push([5, 4]);
    expect(spawnHand(farm, 10)).toEqual([4, 5]);
    farm.hands.push([4, 5]);
    expect(spawnHand(farm, 10)).toEqual([5, 5]);
    farm.hands.push([5, 5]);
    expect(spawnHand(farm, 10)).toEqual([4, 4]); // all occupied once; wraps to NWSE order again
  });
});
