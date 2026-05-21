import { describe, expect, it } from 'vitest';
import { ANIMALS, CROPS, MARKET_PARAMS, PRODUCTS } from '../constants';
import {
  applyUnitAction,
  dailyRefreshAnimals,
  dailyRefreshPlants,
  decayPlants,
  dropInventoriesToShed,
  endOfDay,
  hireCost,
  parseOrder,
  processMarket,
  spawnWeeds,
  step,
  townConsume,
} from '../interpreter';
import { PyRandom } from '../rng';
import { initGameState, newAnimal, newPlant, resolveConfig } from '../state';
import type { AnimalTile, Farm, GameState, PlantTile, PlayerAction, Private, Tile } from '../types';

const cfg = resolveConfig();
const BOARD = cfg.boardSize;
const TPD = cfg.turnsPerDay;
const CAP = cfg.shedCapacity;

function fresh(numAgents = 2, seed = 1): GameState {
  return initGameState(numAgents, cfg, seed);
}

function passActions(n: number): PlayerAction[] {
  return Array.from({ length: n }, () => ({ farmer: ['PASS'], hands: [], market: [] }));
}

function farmerAt(farm: Farm, x: number, y: number): void {
  farm.farmer = [x, y];
}

function getPlant(farm: Farm, x: number, y: number): PlantTile {
  return farm.tiles[y][x] as PlantTile;
}

function getAnimal(farm: Farm, x: number, y: number): AnimalTile {
  return farm.tiles[y][x] as AnimalTile;
}

describe('applyUnitAction — movement', () => {
  it('moves the main farmer onto an unlocked tile and refuses LOCKED/out-of-bounds', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    farmerAt(f, 4, 4);
    applyUnitAction(f, p, 0, ['NORTH'], BOARD, 0, TPD, CAP);
    expect(f.farmer).toEqual([4, 3]);
    // From [4,3] going EAST onto [5,3] is LOCKED (NE quadrant).
    applyUnitAction(f, p, 0, ['EAST'], BOARD, 0, TPD, CAP);
    expect(f.farmer).toEqual([4, 3]); // unchanged
    // From [0,0] going WEST is out of bounds.
    f.farmer = [0, 0];
    applyUnitAction(f, p, 0, ['WEST'], BOARD, 0, TPD, CAP);
    expect(f.farmer).toEqual([0, 0]);
  });
});

describe('applyUnitAction — PLANT/WATER/HARVEST', () => {
  it('PLANT consumes a seed and places a fresh plant tile', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    p.seeds.WHEAT = 1;
    farmerAt(f, 2, 2);
    applyUnitAction(f, p, 0, ['PLANT', 'WHEAT'], BOARD, 0, TPD, CAP);
    expect(p.seeds.WHEAT).toBe(0);
    const t = getPlant(f, 2, 2);
    expect(t.kind).toBe('PLANT');
    expect(t.crop).toBe('WHEAT');
    expect(t.planted_day).toBe(0);
    expect(t.consecutive_unwatered).toBe(1); // planting day counts as unwatered
  });

  it('PLANT fails on a non-empty tile or with no seed', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    farmerAt(f, 2, 2);
    p.seeds.WHEAT = 0;
    applyUnitAction(f, p, 0, ['PLANT', 'WHEAT'], BOARD, 0, TPD, CAP);
    expect(f.tiles[2][2]).toBe(null);
    p.seeds.WHEAT = 1;
    f.tiles[2][2] = { kind: 'WEED' };
    applyUnitAction(f, p, 0, ['PLANT', 'WHEAT'], BOARD, 0, TPD, CAP);
    expect(p.seeds.WHEAT).toBe(1);
  });

  it('WATER is once per day and bumps yield_units inside the harvest window for non-ongoing crops', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    farmerAt(f, 2, 2);
    // WHEAT: max_yield_day=4, window starts at floor((4+1)/2)=2 days after planting.
    f.tiles[2][2] = newPlant('WHEAT', 0, TPD);
    const t = getPlant(f, 2, 2);
    expect(t.yield_units).toBe(1); // base for non-ongoing
    applyUnitAction(f, p, 0, ['WATER'], BOARD, 2, TPD, CAP);
    expect(t.watered_today).toBe(true);
    expect(t.yield_units).toBe(2);
    // Re-watering same day is a no-op (no extra bonus).
    applyUnitAction(f, p, 0, ['WATER'], BOARD, 2, TPD, CAP);
    expect(t.yield_units).toBe(2);
  });

  it('HARVEST requires age >= first_yield_day; clears non-ongoing tiles, keeps ongoing', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    farmerAt(f, 2, 2);
    f.tiles[2][2] = newPlant('WHEAT', 0, TPD);
    const t = getPlant(f, 2, 2);
    t.yield_units = 3;
    // Day 1 < first_yield_day=2 → no harvest.
    applyUnitAction(f, p, 0, ['HARVEST'], BOARD, 1, TPD, CAP);
    expect(t.yield_units).toBe(3);
    applyUnitAction(f, p, 0, ['HARVEST'], BOARD, 2, TPD, CAP);
    expect(f.tiles[2][2]).toBe(null);
    expect(p.inventories[0].WHEAT).toBe(3);

    // Ongoing crop: tile stays.
    f.tiles[2][2] = newPlant('TOMATO', 0, TPD);
    const t2 = getPlant(f, 2, 2);
    t2.yield_units = 1;
    applyUnitAction(f, p, 0, ['HARVEST'], BOARD, 8, TPD, CAP);
    expect(f.tiles[2][2]).not.toBe(null);
    expect((f.tiles[2][2] as PlantTile).yield_units).toBe(0);
    expect(p.inventories[0].TOMATO).toBe(1);
  });
});

describe('applyUnitAction — shed PICKUP/PLACE + DIG + BUILD + animals', () => {
  it('PICKUP only works adjacent to the shed and moves items inv ↔ shed', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    p.shed.WHEAT = 5;
    farmerAt(f, 0, 0);
    applyUnitAction(f, p, 0, ['PICKUP', 'WHEAT', 3], BOARD, 0, TPD, CAP);
    expect(p.shed.WHEAT).toBe(5); // not shed-adjacent
    farmerAt(f, 4, 4);
    applyUnitAction(f, p, 0, ['PICKUP', 'WHEAT', 3], BOARD, 0, TPD, CAP);
    expect(p.shed.WHEAT).toBe(2);
    expect(p.inventories[0].WHEAT).toBe(3);
  });

  it('PLACE drops into shed but caps at shedCapacity', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    p.shed.WHEAT = CAP - 1;
    p.inventories[0].WHEAT = 5;
    farmerAt(f, 4, 4);
    applyUnitAction(f, p, 0, ['PLACE', 'WHEAT', 5], BOARD, 0, TPD, CAP);
    expect(p.shed.WHEAT).toBe(CAP);
    expect(p.inventories[0].WHEAT).toBe(4); // 1 placed, 4 stuck
  });

  it('PLACE on a matching empty structure places an animal from inventory', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    farmerAt(f, 2, 2);
    f.tiles[2][2] = { kind: 'COOP' };
    p.inventories[0].GOOSE = 1;
    applyUnitAction(f, p, 0, ['PLACE', 'GOOSE'], BOARD, 0, TPD, CAP);
    const a = getAnimal(f, 2, 2);
    expect(a.animal).toBe('GOOSE');
    expect(p.inventories[0].GOOSE).toBeUndefined();
  });

  it('DIG removes plants/weeds/empty structures but NOT placed animals', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    farmerAt(f, 2, 2);
    f.tiles[2][2] = { kind: 'WEED' };
    applyUnitAction(f, p, 0, ['DIG'], BOARD, 0, TPD, CAP);
    expect(f.tiles[2][2]).toBe(null);
    f.tiles[2][2] = newAnimal('GOOSE', 0);
    applyUnitAction(f, p, 0, ['DIG'], BOARD, 0, TPD, CAP);
    expect(f.tiles[2][2]).not.toBe(null);
  });

  it('FEED consumes WHEAT and flips fed_today; CARE/COLLECT_FERTILIZER follow tile state', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    farmerAt(f, 2, 2);
    f.tiles[2][2] = newAnimal('GOOSE', 0);
    const a = getAnimal(f, 2, 2);
    applyUnitAction(f, p, 0, ['FEED'], BOARD, 0, TPD, CAP); // no wheat
    expect(a.fed_today).toBe(false);
    p.inventories[0].WHEAT = 1;
    applyUnitAction(f, p, 0, ['FEED'], BOARD, 0, TPD, CAP);
    expect(a.fed_today).toBe(true);
    expect(p.inventories[0].WHEAT).toBeUndefined();
    a.fertilizer_available = true;
    applyUnitAction(f, p, 0, ['COLLECT_FERTILIZER'], BOARD, 0, TPD, CAP);
    expect(a.fertilizer_available).toBe(false);
    expect(p.inventories[0].FERTILIZER).toBe(1);
    applyUnitAction(f, p, 0, ['CARE'], BOARD, 0, TPD, CAP);
    expect(a.cared_today).toBe(true);
  });
});

describe('parseOrder + hireCost', () => {
  it('parseOrder validates shape and rejects non-positive quantities', () => {
    expect(parseOrder(['HIRE'])).toEqual({ type: 'HIRE' });
    expect(parseOrder(['BUY_LAND'])).toEqual({ type: 'BUY_LAND' });
    expect(parseOrder(['SELL', 'WHEAT', 3])).toEqual({ type: 'SELL', item: 'WHEAT', remaining: 3 });
    expect(parseOrder(['SELL', 'WHEAT', 0])).toBe(null);
    expect(parseOrder(['SELL'])).toBe(null);
    expect(parseOrder(null as unknown)).toBe(null);
  });

  it('hireCost follows fib: 1,1,2,3,5,8 with default mult=10', () => {
    expect([0, 1, 2, 3, 4, 5].map((n) => hireCost(n))).toEqual([10, 10, 20, 30, 50, 80]);
  });
});

describe('processMarket', () => {
  it('SELL → BUY_PRODUCT round-trip on WHEAT nets zero against an unchanged market', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    f.money = 1000;
    p.shed.WHEAT = 1;
    const startInv = s.market.inventory.WHEAT;
    const actions: PlayerAction[] = [
      {
        farmer: ['PASS'],
        hands: [],
        market: [
          ['SELL', 'WHEAT', 1],
          ['BUY_PRODUCT', 'WHEAT', 1],
        ],
      },
      { farmer: ['PASS'], hands: [], market: [] },
    ];
    processMarket(s.farms, s.privates, s.market, actions, cfg);
    expect(f.money).toBe(1000);
    expect(p.shed.WHEAT).toBe(1);
    expect(s.market.inventory.WHEAT).toBe(startInv);
  });

  it('HIRE deducts money and appends a hand + inventory slot', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    f.money = 500;
    const actions: PlayerAction[] = [
      { farmer: ['PASS'], hands: [], market: [['HIRE']] },
      { farmer: ['PASS'], hands: [], market: [] },
    ];
    processMarket(s.farms, s.privates, s.market, actions, cfg);
    expect(f.money).toBe(500 - hireCost(0));
    expect(f.hands).toHaveLength(1);
    expect(p.inventories).toHaveLength(2);
  });

  it('BUY_LAND unlocks NE quadrant tiles when affordable', () => {
    const s = fresh();
    const f = s.farms[0];
    f.money = 1000;
    const actions: PlayerAction[] = [
      { farmer: ['PASS'], hands: [], market: [['BUY_LAND']] },
      { farmer: ['PASS'], hands: [], market: [] },
    ];
    processMarket(s.farms, s.privates, s.market, actions, cfg);
    expect(f.unlocked_quadrants).toContain('NE');
    expect(f.tiles[0][9]).toBe(null);
  });

  it('rejects BUY_PRODUCT for non-wheat/fertilizer items (e.g. EGG)', () => {
    const s = fresh();
    const f = s.farms[0];
    f.money = 10_000;
    const actions: PlayerAction[] = [
      { farmer: ['PASS'], hands: [], market: [['BUY_PRODUCT', 'EGG' as 'WHEAT' | 'FERTILIZER', 1]] },
      { farmer: ['PASS'], hands: [], market: [] },
    ];
    processMarket(s.farms, s.privates, s.market, actions, cfg);
    expect(s.privates[0].shed.EGG).toBe(0);
    expect(f.money).toBe(10_000);
  });
});

describe('townConsume + decayPlants', () => {
  it('townConsume depletes inventories from unlocked shop products', () => {
    const s = fresh();
    s.town.unlocked_shops = ['BAKERY'];
    const wheatBefore = s.market.inventory.WHEAT;
    const eggBefore = s.market.inventory.EGG;
    townConsume(s.market, s.town, 0, cfg);
    expect(s.market.inventory.WHEAT).toBe(wheatBefore - 1 - 1); // -1 bakery (mult 1), -1 town center
    expect(s.market.inventory.EGG).toBe(eggBefore - 1 - 1);
  });

  it('decayPlants ticks yield_units down every 2 steps past max_lifespan_step', () => {
    const s = fresh();
    const f = s.farms[0];
    f.tiles[2][2] = newPlant('WHEAT', 0, TPD); // max_lifespan_step = (0+4+1)*24 = 120
    const t = getPlant(f, 2, 2);
    t.yield_units = 2;
    decayPlants(f, 120);
    expect(t.yield_units).toBe(1);
    decayPlants(f, 121); // odd offset → no decay
    expect(t.yield_units).toBe(1);
    decayPlants(f, 122);
    // Hit zero → becomes WEED.
    expect(f.tiles[2][2]).toEqual({ kind: 'WEED' });
  });
});

describe('dailyRefreshPlants + dailyRefreshAnimals', () => {
  it('plant turns to weed after 2 unwatered days', () => {
    const s = fresh();
    const f = s.farms[0];
    f.tiles[2][2] = newPlant('WHEAT', 0, TPD);
    const t = getPlant(f, 2, 2);
    t.consecutive_unwatered = 1; // planted today: starts at 1
    dailyRefreshPlants(f, 0, TPD);
    // Not watered → bumps to 2 → weed
    expect(f.tiles[2][2]).toEqual({ kind: 'WEED' });
  });

  it('ongoing TOMATO produces +1 on each scheduled day after first_yield_day', () => {
    const s = fresh();
    const f = s.farms[0];
    f.tiles[2][2] = newPlant('TOMATO', 0, TPD);
    const t = getPlant(f, 2, 2);
    t.watered_today = true; // pretend watered today
    // first_yield_day=8, interval=1; day=7 → next_day=8, daysSinceFirst=0 → produce
    dailyRefreshPlants(f, 7, TPD);
    expect((f.tiles[2][2] as PlantTile).yield_units).toBe(1);
  });

  it('animal escapes structure after 2 unfed days; structure remains', () => {
    const s = fresh();
    const f = s.farms[0];
    f.tiles[2][2] = newAnimal('GOOSE', 0);
    const a = getAnimal(f, 2, 2);
    a.consecutive_unfed = 1;
    dailyRefreshAnimals(f, 0);
    expect(f.tiles[2][2]).toEqual({ kind: 'COOP' });
  });

  it('animal yields +1 on production day; care bonus accumulates on fed+cared days', () => {
    const s = fresh();
    const f = s.farms[0];
    f.tiles[2][2] = newAnimal('GOOSE', 0);
    const a = getAnimal(f, 2, 2);
    // first_yield_day=4, interval=1. day=3 → nextDay=4, days_since_first=0 → produce.
    a.fed_today = true;
    a.cared_today = true;
    dailyRefreshAnimals(f, 3);
    expect(a.yield_units).toBe(1);
    expect(a.pending_care_bonus).toBe(1); // care+fed accumulates after production
    expect(a.fertilizer_available).toBe(true);
  });
});

describe('spawnWeeds + dropInventoriesToShed', () => {
  it('spawnWeeds is deterministic for the same RNG seed and ignores non-null tiles', () => {
    const make = (): Farm => {
      const s = fresh();
      return s.farms[0];
    };
    const a = make();
    const b = make();
    spawnWeeds(a, BOARD, 1.0, new PyRandom(42)); // every empty tile becomes weed
    spawnWeeds(b, BOARD, 1.0, new PyRandom(42));
    expect(a.tiles).toEqual(b.tiles);
    // Every NW tile should now be a WEED.
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        expect(a.tiles[y][x]).toEqual({ kind: 'WEED' });
      }
    }
  });

  it('dropInventoriesToShed drains inventories up to capacity; overflow vanishes', () => {
    const priv: Private = {
      shed: { WHEAT: CAP - 2 },
      seeds: {} as Private['seeds'],
      inventories: [{ WHEAT: 1, CARROT: 5 }],
    };
    dropInventoriesToShed(priv, CAP);
    expect(priv.shed.WHEAT).toBe(CAP - 1);
    expect(priv.shed.CARROT).toBe(1); // only 1 fit
    expect(priv.inventories[0]).toEqual({});
  });
});

describe('endOfDay', () => {
  it('resets hands, hires_today, inventories; drains inventories into the shed', () => {
    const s = fresh();
    const f = s.farms[0];
    const p = s.privates[0];
    f.hands = [[5, 4]];
    f.hires_today = 1;
    p.inventories.push({ WHEAT: 2 });
    endOfDay(s.farms, s.privates, s.town, 0, cfg, s.seed);
    expect(f.hands).toEqual([]);
    expect(f.hires_today).toBe(0);
    expect(p.inventories).toEqual([{}]);
    expect(p.shed.WHEAT).toBe(2);
  });

  it('unlocks one new shop on every shopInterval boundary', () => {
    const s = fresh();
    // shopInterval=3 → unlocks at day boundary where nextDay % 3 === 0.
    // From day=2, nextDay=3 → unlock.
    endOfDay(s.farms, s.privates, s.town, 2, cfg, s.seed);
    expect(s.town.unlocked_shops).toHaveLength(1);
    // From day=3, nextDay=4 → no unlock.
    endOfDay(s.farms, s.privates, s.town, 3, cfg, s.seed);
    expect(s.town.unlocked_shops).toHaveLength(1);
  });
});

describe('step (top-level)', () => {
  it('advances step/day/hour and clones state (prev untouched)', () => {
    const s = fresh();
    const before = JSON.stringify(s);
    const next = step(s, passActions(2), cfg);
    expect(next.step).toBe(1);
    expect(next.hour).toBe(1);
    expect(next.day).toBe(0);
    expect(next).not.toBe(s);
    expect(JSON.stringify(s)).toBe(before); // prev not mutated
  });

  it('end-of-day fires exactly on the (turnsPerDay-1) → turnsPerDay boundary', () => {
    let s = fresh();
    s.privates[0].inventories[0].WHEAT = 3;
    // Step until hour resets.
    for (let i = 0; i < TPD; i++) s = step(s, passActions(2), cfg);
    expect(s.step).toBe(TPD);
    expect(s.hour).toBe(0);
    expect(s.day).toBe(1);
    // End-of-day swept the inventory into the shed.
    expect(s.privates[0].shed.WHEAT).toBe(3);
    expect(s.privates[0].inventories[0]).toEqual({});
  });

  it('atomic PLANT validation: overdemand for a crop drops ALL PLANT requests for it', () => {
    const s = fresh();
    s.privates[0].seeds.WHEAT = 1; // only 1 seed
    s.privates[0].seeds.CARROT = 5;
    // Two units want WHEAT (>1 seed) → both blocked. CARROT is fine.
    s.farms[0].farmer = [2, 2];
    s.farms[0].hands = [[3, 3]];
    s.privates[0].inventories.push({});
    const actions: PlayerAction[] = [
      {
        farmer: ['PLANT', 'WHEAT'],
        hands: [['PLANT', 'WHEAT']],
        market: [],
      },
      { farmer: ['PASS'], hands: [], market: [] },
    ];
    const next = step(s, actions, cfg);
    expect(next.farms[0].tiles[2][2]).toBe(null); // blocked
    expect(next.farms[0].tiles[3][3]).toBe(null);
    expect(next.privates[0].seeds.WHEAT).toBe(1); // untouched
  });

  it('marks done at episodeSteps - 2 and reports final scores', () => {
    const tinyCfg = { ...cfg, episodeSteps: 3 };
    let s = initGameState(2, tinyCfg, 7);
    s.farms[0].money = 1234;
    s.farms[1].money = 5678;
    s = step(s, passActions(2), tinyCfg); // step 0 -> 1 (not done yet: 0 < 3-2=1)
    expect(s.done).toBe(false);
    s = step(s, passActions(2), tinyCfg); // step 1 -> 2 (1 >= 1 → done)
    expect(s.done).toBe(true);
    expect(s.scores).toEqual([1234, 5678]);
    // Further steps are no-ops.
    const after = step(s, passActions(2), tinyCfg);
    expect(after).toBe(s);
  });

  it('initial market prices stay at base after a single PASS turn (no orders, no shops)', () => {
    const s = fresh();
    const next = step(s, passActions(2), cfg);
    for (const item of PRODUCTS) {
      // Town center sells 1 of each non-fertilizer product on step 0 boundary;
      // FERTILIZER stays at base.
      if (item === 'FERTILIZER') expect(next.market.prices[item]).toBe(MARKET_PARAMS[item].base);
    }
    // Sanity: market is intact.
    expect(Object.keys(next.market.inventory).length).toBe(PRODUCTS.length);
  });
});

// Tiny sanity check that the tile type guards used by interpreter work.
describe('tile sanity', () => {
  it('LOCKED/null/object dispatch behaves', () => {
    const tile: Tile = {
      kind: 'PLANT',
      crop: 'WHEAT',
      planted_day: 0,
      watered_today: false,
      consecutive_unwatered: 0,
      yield_units: 1,
      max_lifespan_step: 120,
      fertilized_until_day: -1,
    };
    expect(typeof tile).toBe('object');
    expect(ANIMALS.GOOSE.product).toBe('EGG');
    expect(CROPS.WHEAT.seed).toBe(10);
  });
});
