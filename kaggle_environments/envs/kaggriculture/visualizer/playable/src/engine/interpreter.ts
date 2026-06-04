/**
 * Per-step game logic. Ports `interpreter()` from kaggriculture.py and every
 * helper it calls — `_apply_unit_action`, `_process_market`, `_town_consume`,
 * `_decay_plants`, `_daily_refresh_plants`, `_daily_refresh_animals`,
 * `_spawn_weeds`, `_drop_inventories_to_shed`, `_end_of_day`.
 *
 * The top-level `step(prev, actions, config)` clones `prev` and returns the
 * next state — callers are free to keep `prev` around for replay/undo.
 */

import {
  ANIMALS,
  CROPS,
  FARMER_MOVES,
  FARM_HAND_COST_MULT,
  LAND_ORDER,
  LAND_PRICES,
  SHOPS,
  SHOP_NAMES,
  TOWN_CENTER_DEMAND_SCHEDULE,
  TOWN_CENTER_PRODUCTS,
} from './constants';
import { marketPrice, refreshPrices } from './market';
import { endOfDaySeed, PyRandom } from './rng';
import { newAnimal, newPlant, defaultSpawn, isShedAdjacent, shedAccessTiles, quadrantOf } from './state';
import type {
  AnimalId,
  AnimalTile,
  Config,
  CropId,
  Farm,
  GameState,
  Market,
  MarketOrder,
  PlantTile,
  PlayerAction,
  Position,
  Private,
  ProductId,
  ShedItemId,
  Tile,
  Town,
  UnitAction,
} from './types';
import { LOCKED } from './types';

// ---------- clone helpers ----------

function cloneTile(tile: Tile): Tile {
  if (tile === null || tile === LOCKED) return tile;
  return { ...tile };
}

function cloneFarm(f: Farm): Farm {
  return {
    money: f.money,
    tiles: f.tiles.map((row) => row.map(cloneTile)),
    farmer: [f.farmer[0], f.farmer[1]],
    hands: f.hands.map((h) => [h[0], h[1]] as Position),
    unlocked_quadrants: [...f.unlocked_quadrants],
    hires_today: f.hires_today,
  };
}

function clonePrivate(p: Private): Private {
  return {
    shed: { ...p.shed },
    seeds: { ...p.seeds },
    inventories: p.inventories.map((inv) => ({ ...inv })),
  };
}

function cloneMarket(m: Market): Market {
  const out: Market = {
    inventory: { ...m.inventory } as Record<ProductId, number>,
    prices: { ...m.prices } as Record<ProductId, number>,
  };
  if (m.params) out.params = m.params; // params are frozen — share ref
  return out;
}

function cloneTown(t: Town): Town {
  return { unlocked_shops: [...t.unlocked_shops] };
}

// ---------- inventory helpers ----------

function invAdd(inv: Partial<Record<ShedItemId, number>>, item: ShedItemId, n = 1): void {
  inv[item] = (inv[item] ?? 0) + n;
}

function invTake(inv: Partial<Record<ShedItemId, number>>, item: ShedItemId, n = 1): boolean {
  if ((inv[item] ?? 0) < n) return false;
  inv[item] = (inv[item] as number) - n;
  if (inv[item] === 0) delete inv[item];
  return true;
}

function unitPosition(farm: Farm, idx: number): Position | null {
  if (idx === 0) return farm.farmer;
  const hi = idx - 1;
  return hi < farm.hands.length ? farm.hands[hi] : null;
}

function setUnitPosition(farm: Farm, idx: number, pos: Position): void {
  if (idx === 0) farm.farmer = [pos[0], pos[1]];
  else farm.hands[idx - 1] = [pos[0], pos[1]];
}

function unitInventory(priv: Private, idx: number): Partial<Record<ShedItemId, number>> {
  while (priv.inventories.length <= idx) priv.inventories.push({});
  return priv.inventories[idx] as Partial<Record<ShedItemId, number>>;
}

function shedTotal(shed: Partial<Record<ShedItemId, number>>): number {
  let sum = 0;
  for (const v of Object.values(shed)) sum += v ?? 0;
  return sum;
}

// ---------- per-unit action ----------

export function applyUnitAction(
  farm: Farm,
  priv: Private,
  idx: number,
  action: UnitAction,
  boardSize: number,
  day: number,
  turnsPerDay: number,
  shedCapacity: number
): void {
  if (!Array.isArray(action)) return;
  const op = action[0];
  if (op === undefined) return;
  const pos = unitPosition(farm, idx);
  if (pos === null) return;
  const [fx, fy] = pos;
  const inv = unitInventory(priv, idx);

  if (op === 'NORTH' || op === 'SOUTH' || op === 'EAST' || op === 'WEST') {
    const [dx, dy] = FARMER_MOVES[op];
    const nx = fx + dx;
    const ny = fy + dy;
    if (nx < 0 || nx >= boardSize || ny < 0 || ny >= boardSize) return;
    if (farm.tiles[ny][nx] === LOCKED) return;
    setUnitPosition(farm, idx, [nx, ny]);
    return;
  }

  if (op === 'PASS') return;

  const tile = farm.tiles[fy][fx];
  if (tile === LOCKED) return;

  if (op === 'PICKUP') {
    if (!isShedAdjacent([fx, fy], boardSize)) return;
    if (action.length < 2) return;
    const item = action[1] as ShedItemId;
    const requested = action.length >= 3 ? (action[2] as number) : 1;
    if (requested <= 0) return;
    const available = priv.shed[item] ?? 0;
    const n = Math.min(requested, available);
    if (n <= 0) return;
    priv.shed[item] = available - n;
    invAdd(inv, item, n);
    return;
  }

  if (op === 'PLANT') {
    if (action.length < 2) return;
    const crop = action[1] as CropId;
    if (!(crop in CROPS)) return;
    if (tile !== null) return;
    if ((priv.seeds[crop] ?? 0) <= 0) return;
    priv.seeds[crop] -= 1;
    farm.tiles[fy][fx] = newPlant(crop, day, turnsPerDay);
    return;
  }

  if (op === 'WATER') {
    if (!isPlant(tile)) return;
    if (tile.watered_today) return;
    tile.watered_today = true;
    const cd = CROPS[tile.crop];
    if (!cd.ongoing) {
      const ageDays = day - tile.planted_day;
      const windowStart = Math.floor((cd.max_yield_day + 1) / 2);
      if (ageDays >= windowStart && ageDays <= cd.max_yield_day) {
        const bonus = tile.fertilized_until_day >= day ? 2 : 1;
        tile.yield_units = Math.min(cd.max_yield, tile.yield_units + bonus);
      }
    }
    return;
  }

  if (op === 'HARVEST') {
    if (!isObjectTile(tile)) return;
    const units = (tile as PlantTile | AnimalTile).yield_units ?? 0;
    if (units <= 0) return;
    if (isPlant(tile)) {
      const cd = CROPS[tile.crop];
      if (day - tile.planted_day < cd.first_yield_day) return;
      tile.yield_units = 0;
      invAdd(inv, tile.crop, units);
      if (!cd.ongoing) farm.tiles[fy][fx] = null;
    } else if (isAnimal(tile)) {
      tile.yield_units = 0;
      invAdd(inv, ANIMALS[tile.animal].product, units);
    }
    return;
  }

  if (op === 'FERTILIZE') {
    if (!isPlant(tile)) return;
    if (!invTake(inv, 'FERTILIZER', 1)) return;
    tile.fertilized_until_day = Math.max(tile.fertilized_until_day, day + 2);
    return;
  }

  if (op === 'DIG') {
    if (tile === null) return;
    if (isAnimal(tile)) return; // can't dig a placed animal
    farm.tiles[fy][fx] = null;
    return;
  }

  if (op === 'BUILD_COOP') {
    if (tile !== null) return;
    farm.tiles[fy][fx] = { kind: 'COOP' };
    return;
  }

  if (op === 'BUILD_PASTURE') {
    if (tile !== null) return;
    farm.tiles[fy][fx] = { kind: 'PASTURE' };
    return;
  }

  if (op === 'PLACE') {
    if (action.length < 2) return;
    const item = action[1] as ShedItemId;
    // Animal placement: standing on a matching unoccupied structure.
    if (item in ANIMALS && isObjectTile(tile)) {
      const t = tile as { kind: string; animal?: AnimalId };
      if (t.kind === ANIMALS[item as AnimalId].structure && !('animal' in t)) {
        if (invTake(inv, item, 1)) {
          farm.tiles[fy][fx] = newAnimal(item as AnimalId, day);
        }
        return;
      }
    }
    // Shed drop: adjacent to the shed; obeys shedCapacity.
    if (isShedAdjacent([fx, fy], boardSize)) {
      const requested = action.length >= 3 ? (action[2] as number) : 1;
      if (requested <= 0) return;
      let n = Math.min(requested, inv[item] ?? 0);
      if (n <= 0) return;
      const current = shedTotal(priv.shed);
      const room = Math.max(0, shedCapacity - current);
      n = Math.min(n, room);
      if (n <= 0) return;
      inv[item] = (inv[item] as number) - n;
      if (inv[item] === 0) delete inv[item];
      priv.shed[item] = (priv.shed[item] ?? 0) + n;
    }
    return;
  }

  if (op === 'FEED') {
    if (!isAnimal(tile)) return;
    if (tile.fed_today) return;
    if (!invTake(inv, 'WHEAT', 1)) return;
    tile.fed_today = true;
    return;
  }

  if (op === 'COLLECT_FERTILIZER') {
    if (!isAnimal(tile)) return;
    if (!tile.fertilizer_available) return;
    tile.fertilizer_available = false;
    invAdd(inv, 'FERTILIZER', 1);
    return;
  }

  if (op === 'CARE') {
    if (!isAnimal(tile)) return;
    if (tile.cared_today) return;
    tile.cared_today = true;
    return;
  }
}

function isObjectTile(tile: Tile): tile is Exclude<Tile, null | typeof LOCKED> {
  return tile !== null && tile !== LOCKED;
}

function isPlant(tile: Tile): tile is PlantTile {
  return isObjectTile(tile) && (tile as { kind: string }).kind === 'PLANT';
}

function isAnimal(tile: Tile): tile is AnimalTile {
  return isObjectTile(tile) && 'animal' in (tile as object);
}

// ---------- market order processing ----------

type OrderType = 'HIRE' | 'BUY_LAND' | 'BUY_SEED' | 'BUY_PRODUCT' | 'BUY_ANIMAL' | 'SELL';

interface AtomicOrderState {
  type: 'HIRE' | 'BUY_LAND';
}
interface QuantityOrderState {
  type: 'BUY_SEED' | 'BUY_PRODUCT' | 'BUY_ANIMAL' | 'SELL';
  item: string;
  remaining: number;
}
type OrderState = AtomicOrderState | QuantityOrderState;

function isQuantityOrder(o: OrderState): o is QuantityOrderState {
  return o.type !== 'HIRE' && o.type !== 'BUY_LAND';
}

export function parseOrder(order: MarketOrder | unknown): OrderState | null {
  if (!Array.isArray(order) || order.length === 0) return null;
  const op = order[0] as OrderType;
  if (op === 'HIRE') return { type: 'HIRE' };
  if (op === 'BUY_LAND') return { type: 'BUY_LAND' };
  if (op === 'BUY_SEED' || op === 'BUY_PRODUCT' || op === 'BUY_ANIMAL' || op === 'SELL') {
    if (order.length < 3) return null;
    const n = Math.floor(Number(order[2]));
    if (!Number.isFinite(n) || n <= 0) return null;
    return { type: op, item: order[1] as string, remaining: n };
  }
  return null;
}

function fib(n: number): number {
  let a = 1;
  let b = 1;
  for (let i = 0; i < n; i++) {
    const next = a + b;
    a = b;
    b = next;
  }
  return a;
}

export function hireCost(nAlreadyToday: number, mult = FARM_HAND_COST_MULT): number {
  return mult * fib(nAlreadyToday);
}

function spawnHand(farm: Farm, boardSize: number): Position {
  const tiles = shedAccessTiles(boardSize);
  const occupants: number[] = tiles.map(() => 0);
  const allPos: Position[] = [farm.farmer, ...farm.hands];
  for (const p of allPos) {
    const idx = tiles.findIndex(([tx, ty]) => tx === p[0] && ty === p[1]);
    if (idx >= 0) occupants[idx]++;
  }
  let best = 0;
  for (let i = 1; i < tiles.length; i++) {
    if (occupants[i] < occupants[best]) best = i;
  }
  return [tiles[best][0], tiles[best][1]];
}

function doHire(farm: Farm, priv: Private, boardSize: number, mult: number): void {
  const cost = hireCost(farm.hires_today, mult);
  if (farm.money < cost) return;
  farm.money -= cost;
  farm.hires_today += 1;
  farm.hands.push(spawnHand(farm, boardSize));
  priv.inventories.push({});
}

function doBuyLand(farm: Farm, boardSize: number): void {
  const nExtra = farm.unlocked_quadrants.length - 1;
  if (nExtra >= LAND_ORDER.length) return;
  const cost = LAND_PRICES[nExtra];
  if (farm.money < cost) return;
  farm.money -= cost;
  const quadrant = LAND_ORDER[nExtra];
  farm.unlocked_quadrants.push(quadrant);
  for (let y = 0; y < boardSize; y++) {
    for (let x = 0; x < boardSize; x++) {
      if (quadrantOf(x, y, boardSize) === quadrant && farm.tiles[y][x] === LOCKED) {
        farm.tiles[y][x] = null;
      }
    }
  }
}

function commitUnit(
  op: QuantityOrderState['type'],
  item: string,
  price: number,
  farm: Farm,
  priv: Private,
  market: Market
): boolean {
  if (op === 'SELL') {
    const have = priv.shed[item as ShedItemId] ?? 0;
    if (have <= 0) return false;
    priv.shed[item as ShedItemId] = have - 1;
    farm.money += price;
    if (price > 1) market.inventory[item as ProductId] += 1;
    return true;
  }
  if (op === 'BUY_PRODUCT') {
    if (farm.money < price) return false;
    farm.money -= price;
    priv.shed[item as ShedItemId] = (priv.shed[item as ShedItemId] ?? 0) + 1;
    market.inventory[item as ProductId] -= 1;
    return true;
  }
  if (op === 'BUY_SEED') {
    if (farm.money < price) return false;
    farm.money -= price;
    priv.seeds[item as CropId] = (priv.seeds[item as CropId] ?? 0) + 1;
    return true;
  }
  if (op === 'BUY_ANIMAL') {
    if (farm.money < price) return false;
    farm.money -= price;
    priv.shed[item as ShedItemId] = (priv.shed[item as ShedItemId] ?? 0) + 1;
    return true;
  }
  return false;
}

export function processMarket(
  farms: Farm[],
  privates: Private[],
  market: Market,
  actions: PlayerAction[],
  config: Config
): void {
  const maxOrders = Math.max(1, config.maxMarketOrdersPerTurn);
  const hireMult = config.farmHandCostMult;
  const boardSize = config.boardSize;

  const queues: MarketOrder[][] = farms.map((_, pid) => {
    const m = actions[pid]?.market ?? [];
    return (Array.isArray(m) ? m : []).slice(0, maxOrders);
  });

  const maxLen = queues.reduce((a, q) => Math.max(a, q.length), 0);
  for (let i = 0; i < maxLen; i++) {
    const orderStates: (OrderState | null)[] = queues.map((q) => (i < q.length ? parseOrder(q[i]) : null));

    // Atomic orders: HIRE, BUY_LAND — once each, in player order.
    for (let pid = 0; pid < orderStates.length; pid++) {
      const os = orderStates[pid];
      if (!os) continue;
      if (os.type === 'HIRE') {
        doHire(farms[pid], privates[pid], boardSize, hireMult);
        orderStates[pid] = null;
      } else if (os.type === 'BUY_LAND') {
        doBuyLand(farms[pid], boardSize);
        orderStates[pid] = null;
      }
    }

    // Per-unit lockstep loop for SELL / BUY_*.
    let esc = 0;
    while (true) {
      esc += 1;
      if (esc >= 100_000) break;
      const quoted: ([QuantityOrderState['type'], string, number, QuantityOrderState] | null)[] = [null, null];
      // Resize quoted to match player count.
      while (quoted.length < orderStates.length) quoted.push(null);
      quoted.length = orderStates.length;
      for (let pid = 0; pid < orderStates.length; pid++) quoted[pid] = null;

      for (let pid = 0; pid < orderStates.length; pid++) {
        const raw = orderStates[pid];
        if (!raw || !isQuantityOrder(raw)) continue;
        const os = raw;
        if (os.remaining <= 0) continue;
        const { type: op, item } = os;
        if (op === 'SELL' && item in market.inventory) {
          quoted[pid] = [
            op,
            item,
            marketPrice(item as ProductId, market.inventory[item as ProductId], market.params),
            os,
          ];
        } else if (op === 'BUY_PRODUCT' && (item === 'WHEAT' || item === 'FERTILIZER')) {
          quoted[pid] = [
            op,
            item,
            marketPrice(item as ProductId, market.inventory[item as ProductId] - 1, market.params),
            os,
          ];
        } else if (op === 'BUY_SEED' && item in CROPS) {
          quoted[pid] = [op, item, CROPS[item as CropId].seed, os];
        } else if (op === 'BUY_ANIMAL' && item in ANIMALS) {
          quoted[pid] = [op, item, ANIMALS[item as AnimalId].cost, os];
        } else {
          orderStates[pid] = null;
        }
      }

      if (quoted.every((q) => q === null)) break;

      let committedAny = false;
      for (let pid = 0; pid < quoted.length; pid++) {
        const q = quoted[pid];
        if (q === null) continue;
        const [op, item, price, os] = q;
        const ok = commitUnit(op, item, price, farms[pid], privates[pid], market);
        if (ok) {
          os.remaining -= 1;
          committedAny = true;
        } else {
          orderStates[pid] = null;
        }
      }

      if (!committedAny) break;
    }

    refreshPrices(market);
  }
}

// ---------- town consumption ----------

export function townConsume(market: Market, town: Town, step: number, config: Config): void {
  const shopInterval = Math.max(1, config.townShopSellInterval);
  const centerInterval = Math.max(1, config.townCenterSellInterval);
  const turnsPerDay = Math.max(1, config.turnsPerDay);
  const day = Math.floor(step / turnsPerDay);

  if (step % shopInterval === 0) {
    for (const shop of town.unlocked_shops) {
      const products = SHOPS[shop];
      const mult = products.length === 1 ? 2 : 1;
      for (const item of products) {
        market.inventory[item as ProductId] -= mult;
      }
    }
  }

  if (step % centerInterval === 0) {
    let centerMult = 1;
    for (const [threshold, m] of TOWN_CENTER_DEMAND_SCHEDULE) {
      if (day >= threshold) {
        centerMult = m;
        break;
      }
    }
    for (const item of TOWN_CENTER_PRODUCTS) {
      market.inventory[item] -= centerMult;
    }
  }

  refreshPrices(market);
}

// ---------- plant decay ----------

export function decayPlants(farm: Farm, step: number): void {
  const size = farm.tiles.length;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const tile = farm.tiles[y][x];
      if (!isPlant(tile)) continue;
      const mls = tile.max_lifespan_step;
      if (mls < 0 || step < mls) continue;
      if ((step - mls) % 2 !== 0) continue;
      tile.yield_units -= 1;
      if (tile.yield_units <= 0) farm.tiles[y][x] = { kind: 'WEED' };
    }
  }
}

// ---------- daily refresh ----------

export function dailyRefreshPlants(farm: Farm, currentDay: number, turnsPerDay: number): void {
  const size = farm.tiles.length;
  const nextDay = currentDay + 1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const tile = farm.tiles[y][x];
      if (!isPlant(tile)) continue;
      const wasWatered = tile.watered_today;
      if (wasWatered) tile.consecutive_unwatered = 0;
      else tile.consecutive_unwatered += 1;
      tile.watered_today = false;
      if (tile.consecutive_unwatered >= 2) {
        farm.tiles[y][x] = { kind: 'WEED' };
        continue;
      }
      const cd = CROPS[tile.crop];
      if (!cd.ongoing) continue;
      const daysSinceFirst = nextDay - tile.planted_day - cd.first_yield_day;
      if (daysSinceFirst < 0) continue;
      const interval = cd.interval;
      if (daysSinceFirst % interval !== 0) continue;
      const productionCount = Math.floor(daysSinceFirst / interval) + 1;
      if (productionCount > cd.max_yield) continue;
      const fertilized = wasWatered && tile.fertilized_until_day >= currentDay;
      tile.yield_units = Math.min(cd.max_yield, tile.yield_units + (fertilized ? 2 : 1));
      if (productionCount === cd.max_yield) {
        tile.max_lifespan_step = (nextDay + 1) * turnsPerDay;
      }
    }
  }
}

export function dailyRefreshAnimals(farm: Farm, day: number): void {
  const size = farm.tiles.length;
  const nextDay = day + 1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const tile = farm.tiles[y][x];
      if (!isAnimal(tile)) continue;
      if (tile.fed_today) tile.consecutive_unfed = 0;
      else tile.consecutive_unfed += 1;
      if (tile.consecutive_unfed >= 2) {
        farm.tiles[y][x] = { kind: ANIMALS[tile.animal].structure };
        continue;
      }
      const a = ANIMALS[tile.animal];
      const daysSinceFirst = nextDay - tile.placed_day - a.first_yield_day;
      if (daysSinceFirst >= 0 && daysSinceFirst % a.interval === 0) {
        const base = 1;
        const bonus = tile.fed_today ? tile.pending_care_bonus : 0;
        tile.yield_units = Math.min(a.max_held, tile.yield_units + base + bonus);
        tile.pending_care_bonus = 0;
      }
      if (tile.cared_today && tile.fed_today) {
        tile.pending_care_bonus = tile.pending_care_bonus + 1;
      }
      tile.fertilizer_available = true;
      tile.fed_today = false;
      tile.cared_today = false;
    }
  }
}

// ---------- weeds, drop, end-of-day ----------

export function spawnWeeds(farm: Farm, boardSize: number, weedChance: number, rng: PyRandom): void {
  for (let y = 0; y < boardSize; y++) {
    for (let x = 0; x < boardSize; x++) {
      if (farm.tiles[y][x] === null && rng.random() < weedChance) {
        farm.tiles[y][x] = { kind: 'WEED' };
      }
    }
  }
}

export function dropInventoriesToShed(priv: Private, capacity: number): void {
  const shed = priv.shed;
  for (const inv of priv.inventories) {
    for (const [item, n] of Object.entries(inv) as Array<[ShedItemId, number]>) {
      if (n <= 0) {
        delete inv[item];
        continue;
      }
      const current = shedTotal(shed);
      const room = Math.max(0, capacity - current);
      const take = Math.min(n, room);
      if (take > 0) shed[item] = (shed[item] ?? 0) + take;
      delete inv[item];
    }
  }
}

export function endOfDay(
  farms: Farm[],
  privates: Private[],
  town: Town,
  day: number,
  config: Config,
  seed: number
): void {
  const boardSize = config.boardSize;
  const turnsPerDay = Math.max(1, config.turnsPerDay);
  const weedChance = config.weedSpawnChance;
  const shedCap = config.shedCapacity;
  const shopInterval = Math.max(1, config.townShopUnlockInterval);

  const rng = new PyRandom(endOfDaySeed(seed, day));

  for (let pid = 0; pid < farms.length; pid++) {
    const farm = farms[pid];
    const priv = privates[pid];
    dailyRefreshPlants(farm, day, turnsPerDay);
    dailyRefreshAnimals(farm, day);
    spawnWeeds(farm, boardSize, weedChance, rng);
    dropInventoriesToShed(priv, shedCap);
    farm.farmer = defaultSpawn(boardSize);
    farm.hands = [];
    farm.hires_today = 0;
    priv.inventories = [{}];
  }

  const nextDay = day + 1;
  if (nextDay > 0 && nextDay % shopInterval === 0) {
    const remaining = SHOP_NAMES.filter((s) => !town.unlocked_shops.includes(s));
    if (remaining.length > 0) {
      const choice = rng.choice([...remaining].sort());
      town.unlocked_shops.push(choice);
    }
  }
}

// ---------- top-level step ----------

export function step(prev: GameState, actions: PlayerAction[], config: Config): GameState {
  if (prev.done) return prev;

  const farms = prev.farms.map(cloneFarm);
  const privates = prev.privates.map(clonePrivate);
  const market = cloneMarket(prev.market);
  const town = cloneTown(prev.town);

  const currentStep = prev.step;
  const turnsPerDay = Math.max(1, config.turnsPerDay);
  const boardSize = config.boardSize;
  const shedCapacity = config.shedCapacity;
  const day = Math.floor(currentStep / turnsPerDay);

  for (let i = 0; i < prev.numAgents; i++) {
    const action = actions[i] ?? { farmer: ['PASS'], hands: [], market: [] };
    const farmerAction: UnitAction = action.farmer ?? ['PASS'];
    const handsActions: UnitAction[] = Array.isArray(action.hands) ? action.hands : [];

    // Atomic PLANT validation: if total PLANT requests for a crop this turn
    // exceed available seeds, drop ALL PLANT requests for that crop.
    const unitActions: UnitAction[] = [farmerAction, ...handsActions];
    const plantDemand: Partial<Record<CropId, number>> = {};
    for (const a of unitActions) {
      if (Array.isArray(a) && a.length >= 2 && a[0] === 'PLANT') {
        const crop = a[1] as CropId;
        plantDemand[crop] = (plantDemand[crop] ?? 0) + 1;
      }
    }
    const seeds = privates[i].seeds;
    const blocked = new Set<CropId>();
    for (const [crop, n] of Object.entries(plantDemand) as Array<[CropId, number]>) {
      if (n > (seeds[crop] ?? 0)) blocked.add(crop);
    }
    const allowed = (a: UnitAction): UnitAction => {
      if (Array.isArray(a) && a.length >= 2 && a[0] === 'PLANT' && blocked.has(a[1] as CropId)) {
        return ['PASS'];
      }
      return a;
    };

    applyUnitAction(farms[i], privates[i], 0, allowed(farmerAction), boardSize, day, turnsPerDay, shedCapacity);
    for (let h = 0; h < handsActions.length; h++) {
      applyUnitAction(
        farms[i],
        privates[i],
        h + 1,
        allowed(handsActions[h]),
        boardSize,
        day,
        turnsPerDay,
        shedCapacity
      );
    }
  }

  processMarket(farms, privates, market, actions, config);
  townConsume(market, town, currentStep, config);
  for (const farm of farms) decayPlants(farm, currentStep);

  if ((currentStep + 1) % turnsPerDay === 0) {
    endOfDay(farms, privates, town, day, config, prev.seed);
  }

  const newStep = currentStep + 1;
  const done = currentStep >= config.episodeSteps - 2;
  const scores = done ? farms.map((f) => f.money) : prev.scores;

  return {
    step: newStep,
    day: Math.floor(newStep / turnsPerDay),
    hour: newStep % turnsPerDay,
    numAgents: prev.numAgents,
    seed: prev.seed,
    farms,
    privates,
    market,
    town,
    done,
    scores,
  };
}
