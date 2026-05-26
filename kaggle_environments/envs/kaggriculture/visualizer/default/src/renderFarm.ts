import {
  CROP_FIRST_YIELD_DAY,
  INVENTORY_SLOTS,
  MARKET_ITEMS,
  QUADRANT_BY_SEGMENT,
  SEGMENT,
  SURROUNDING_BUILDINGS,
  TOWN_CENTER_INDEX,
  TOWN_GRID_COLS,
  TOWN_GRID_ROWS,
  TOWN_SIGN_INDEX,
  type BoardSize,
  type CellRefs,
  type FarmPublic,
  type LayoutRefs,
  type MarketPublic,
  type PlayerRefs,
  type PrivateState,
  type RawTile,
  type TownPublic,
  type ViewModel,
} from './types';
import { BG_URLS, clearChildren, plantSprite, spriteSrc, marketSpriteSrc, titleCase } from './utils';

export type { BoardSize, LayoutRefs } from './types';

// Quote characters in inlined data: URIs must be percent-encoded so the
// unquoted CSS url() value stays valid when embedded in an HTML style attribute.
const encUrl = (u: string) => u.replace(/'/g, '%27').replace(/"/g, '%22');
const BG_GRASS = `background-image:url(${encUrl(BG_URLS.grass)})`;
const BG_WOOD = `background-image:url(${encUrl(BG_URLS.wood)})`;
const BG_COBBLE_SLOT = `background-image:url(${encUrl(BG_URLS.cobble)});background-repeat:repeat;background-size:100% auto;image-rendering:pixelated;`;

function marketList(): string {
  return MARKET_ITEMS.map(
    ({ sprite, key }) => `
    <div class="market-item" data-item="${key}">
      <img class="market-item-icon" src="${marketSpriteSrc(sprite)}" alt="${sprite}" />
      <div class="market-price-row">
        <span class="market-coin-stack">
          <img class="market-coin" src="${spriteSrc('coin')}" alt="coins" />
          <svg class="market-sparkline" viewBox="0 0 100 20" preserveAspectRatio="none" aria-hidden="true">
            <path class="market-sparkline-path" d="" fill="none" />
          </svg>
        </span>
        <span class="market-price">--</span>
      </div>
    </div>
  `
  ).join('');
}

function farmCell(row: number, col: number, rows: number, cols: number): string {
  const segR = Math.floor(row / SEGMENT);
  const segC = Math.floor(col / SEGMENT);
  const segId = segR * 2 + segC;
  const fences: string[] = [];
  if (row === 0) {
    fences.push(`<img class="cell-fence cell-fence-top" src="${spriteSrc('fence_horizontal')}" alt="" />`);
  }
  if (row === rows - 1) {
    fences.push(`<img class="cell-fence cell-fence-bottom" src="${spriteSrc('fence_horizontal')}" alt="" />`);
  }
  if (col === 0) {
    fences.push(`<img class="cell-fence cell-fence-left" src="${spriteSrc('fence_vertical')}" alt="" />`);
  }
  if (col === cols - 1) {
    fences.push(`<img class="cell-fence cell-fence-right" src="${spriteSrc('fence_vertical')}" alt="" />`);
  }
  return `
    <div class="cell" data-row="${row}" data-col="${col}" data-segment="${segId}">
      <div class="cell-base">
        <img class="cell-sprite" src="${spriteSrc('locked_cell')}" alt="locked_cell" />
      </div>
      <div class="cell-overlay"></div>
      <div class="cell-object"></div>
      <div class="cell-agent"></div>
      ${fences.join('')}
    </div>
  `;
}

function farmGrid(rows: number, cols: number): string {
  const cells: string[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      cells.push(farmCell(r, c, rows, cols));
    }
  }
  return `<div class="farm-grid" style="grid-template-columns: repeat(${cols}, 1fr);">
    ${cells.join('')}
    <div class="shed-overlay">
      <img class="shed-sprite" src="${spriteSrc('shed')}" alt="shed" />
    </div>
  </div>`;
}

function inventoryGrid(): string {
  const slots: string[] = [];
  for (let i = 0; i < INVENTORY_SLOTS; i++) {
    slots.push(`
      <div class="inventory-slot" data-slot="${i}">
        <div class="item-icon"></div>
        <div class="item-count"></div>
      </div>
    `);
  }
  return `<div class="inventory-grid">${slots.join('')}</div>`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function farmPanel(player: 1 | 2, rows: number, cols: number, name: string): string {
  return `
    <section class="farm-panel" data-player="${player}">
      <div class="farm-area">
        ${farmGrid(rows, cols)}
      </div>
      <div class="player-box sketched-border" style="${BG_WOOD}">
        <header class="farm-header">
          <div class="farm-header-left">
            <span class="player-name">
              <img class="player-name-icon" src="${spriteSrc(`farmer_p${player}`)}" alt="farmer p${player}" />
              <span>${escapeHtml(name)}</span>
            </span>
            <button type="button" class="header-toggle shed-toggle" data-dialog="shed-${player}">Shed</button>
          </div>
          <span class="player-balance">
            <img class="balance-icon" src="${spriteSrc('coin')}" alt="coins" />
            <span class="balance-amount">0</span>
          </span>
        </header>
        <div class="shed-area">
          ${inventoryGrid()}
        </div>
      </div>
    </section>
  `;
}

function townPanel(): string {
  return `
    <section class="town-panel">
      <div class="town-grid" style="grid-template-columns: repeat(${TOWN_GRID_COLS}, 1fr);">
        <div class="town-flower town-flower-top" style="background-image:url(${encUrl(spriteSrc('flowers_horizontal'))})"></div>
        <div class="town-flower town-flower-bottom" style="background-image:url(${encUrl(spriteSrc('flowers_horizontal'))})"></div>
        <div class="town-flower town-flower-left" style="--flower-bg:url(${encUrl(spriteSrc('flowers_horizontal'))})"></div>
        <div class="town-flower town-flower-right" style="--flower-bg:url(${encUrl(spriteSrc('flowers_horizontal'))})"></div>
        ${Array.from({ length: TOWN_GRID_COLS * TOWN_GRID_ROWS }, (_, i) => {
          if (i === TOWN_CENTER_INDEX) {
            return `<div class="town-slot town-slot--center" data-slot="${i}" style="${BG_COBBLE_SLOT}">
                      <img class="town-sprite" src="${spriteSrc('town_center')}" alt="Town Center" title="Town Center" />
                    </div>`;
          }
          if (i === TOWN_SIGN_INDEX) {
            return `<div class="town-slot town-slot--sign" data-slot="${i}" style="${BG_COBBLE_SLOT}">
                      <img class="town-sprite" src="${spriteSrc('town_sign')}" alt="Town Sign" title="Town Sign" />
                    </div>`;
          }
          const building = SURROUNDING_BUILDINGS[i];
          if (building) {
            // renderTown injects the shop sprite once the shop unlocks.
            return `<div class="town-slot town-slot--shop" data-slot="${i}" data-building="${building.shop}" style="${BG_COBBLE_SLOT}"></div>`;
          }
          return `<div class="town-slot" data-slot="${i}"></div>`;
        }).join('')}
      </div>
      <div class="market-panel sketched-border" style="${BG_WOOD}">
        <div class="market-header">
          <span class="market-header-clock">
            Day <span class="day-value">1</span> / <span class="day-total">30</span>
            <span class="market-header-sep">·</span>
            Turn <span class="turn-value">1</span> / <span class="turn-total">24</span>
          </span>
        </div>
        <div class="market-list">${marketList()}</div>
      </div>
    </section>
  `;
}

function mobileTitleBar(): string {
  return `
    <header class="mobile-title-bar sketched-border" style="${BG_WOOD}">
      <div class="mobile-title-bar-info">
        Day <span class="day-value">1</span> / <span class="day-total">30</span>
        <span class="market-header-sep">·</span>
        Turn <span class="turn-value">1</span> / <span class="turn-total">24</span>
      </div>
      <div class="mobile-title-bar-toggles">
        <button type="button" class="header-toggle market-toggle" data-dialog="market">Market</button>
        <button type="button" class="header-toggle town-toggle" data-dialog="town">Town</button>
      </div>
    </header>
  `;
}

export function buildShell(root: HTMLElement, board: BoardSize, playerNames: string[]): void {
  root.innerHTML = `
    <div class="kaggriculture-container" style="${BG_GRASS}">
      <main class="kaggriculture-main">
        ${mobileTitleBar()}
        ${farmPanel(1, board.rows, board.cols, playerNames[0] ?? 'Player 1')}
        ${townPanel()}
        ${farmPanel(2, board.rows, board.cols, playerNames[1] ?? 'Player 2')}
      </main>
      <div class="simple-dialog" hidden>
        <div class="simple-dialog-titlebar">
          <span class="simple-dialog-title"></span>
          <button type="button" class="simple-dialog-close" aria-label="Close">×</button>
        </div>
        <div class="simple-dialog-body"></div>
      </div>
    </div>
  `;
}

// --- Cached refs --------------------------------------------------------------

function collectPlayerRefs(panel: HTMLElement, board: BoardSize): PlayerRefs {
  const cells: CellRefs[][] = Array.from({ length: board.rows }, () => new Array<CellRefs>(board.cols));
  panel.querySelectorAll<HTMLElement>('.cell').forEach((el) => {
    const row = Number(el.dataset.row);
    const col = Number(el.dataset.col);
    cells[row][col] = {
      el,
      segment: Number(el.dataset.segment),
      baseImg: el.querySelector<HTMLImageElement>('.cell-base .cell-sprite')!,
      objectSlot: el.querySelector<HTMLElement>('.cell-object')!,
      agentSlot: el.querySelector<HTMLElement>('.cell-agent')!,
    };
  });
  const inventory = Array.from(panel.querySelectorAll<HTMLElement>('.inventory-slot'), (slot) => ({
    icon: slot.querySelector<HTMLElement>('.item-icon')!,
    count: slot.querySelector<HTMLElement>('.item-count')!,
  }));
  return {
    panel,
    balance: panel.querySelector<HTMLElement>('.balance-amount')!,
    cells,
    inventory,
  };
}

export function collectRefs(root: HTMLElement, board: BoardSize): LayoutRefs {
  const marketItems: LayoutRefs['marketItems'] = {};
  for (const { key } of MARKET_ITEMS) {
    const item = root.querySelector<HTMLElement>(`.market-item[data-item="${key}"]`)!;
    marketItems[key] = {
      item,
      price: item.querySelector<HTMLElement>('.market-price')!,
      sparkPath: item.querySelector<SVGPathElement>('.market-sparkline-path')!,
    };
  }
  const overlay = root.querySelector<HTMLElement>('.simple-dialog')!;
  const dialog = {
    overlay,
    title: overlay.querySelector<HTMLElement>('.simple-dialog-title')!,
    body: overlay.querySelector<HTMLElement>('.simple-dialog-body')!,
    closeBtn: overlay.querySelector<HTMLElement>('.simple-dialog-close')!,
  };
  const refs: LayoutRefs = {
    dayValues: Array.from(root.querySelectorAll<HTMLElement>('.day-value')),
    turnValues: Array.from(root.querySelectorAll<HTMLElement>('.turn-value')),
    marketItems,
    shopSlots: Array.from(root.querySelectorAll<HTMLElement>('.town-slot--shop')),
    players: [1, 2].map((p) =>
      collectPlayerRefs(root.querySelector<HTMLElement>(`.farm-panel[data-player="${p}"]`)!, board)
    ),
    dialog,
  };
  wireDialogs(root, refs);
  return refs;
}

interface DialogTarget {
  title: string;
  panel: HTMLElement;
}

function wireDialogs(root: HTMLElement, refs: LayoutRefs): void {
  const targets = new Map<string, DialogTarget>();
  const p1Shed = root.querySelector<HTMLElement>('.farm-panel[data-player="1"] .shed-area');
  const p2Shed = root.querySelector<HTMLElement>('.farm-panel[data-player="2"] .shed-area');
  const market = root.querySelector<HTMLElement>('.market-panel');
  const town = root.querySelector<HTMLElement>('.town-grid');
  if (p1Shed) targets.set('shed-1', { title: 'Player 1 Shed', panel: p1Shed });
  if (p2Shed) targets.set('shed-2', { title: 'Player 2 Shed', panel: p2Shed });
  if (market) targets.set('market', { title: 'Market', panel: market });
  if (town) targets.set('town', { title: 'Town', panel: town });

  const homes = new Map<HTMLElement, { parent: HTMLElement; next: ChildNode | null }>();
  for (const { panel } of targets.values()) {
    homes.set(panel, { parent: panel.parentElement!, next: panel.nextSibling });
  }

  let currentKey: string | null = null;
  const close = () => {
    if (!currentKey) return;
    const target = targets.get(currentKey);
    if (target) {
      const home = homes.get(target.panel);
      if (home) home.parent.insertBefore(target.panel, home.next);
    }
    refs.dialog.overlay.hidden = true;
    currentKey = null;
  };
  const open = (key: string) => {
    if (currentKey === key) {
      close();
      return;
    }
    if (currentKey) close();
    const target = targets.get(key);
    if (!target) return;
    refs.dialog.title.textContent = target.title;
    refs.dialog.body.appendChild(target.panel);
    refs.dialog.overlay.hidden = false;
    currentKey = key;
  };

  root.querySelectorAll<HTMLElement>('.header-toggle[data-dialog]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const key = btn.dataset.dialog;
      if (key) open(key);
    });
  });
  refs.dialog.closeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    close();
  });
}

// --- Per-step render ---------------------------------------------------------

function plantStage(crop: string, plantedDay: number, day: number, yieldUnits: number): string {
  // Non-ongoing crops are seeded with yield_units=1 immediately, so yield_units
  // alone doesn't mean "harvestable" -- the interpreter also gates harvest on
  // age_days >= first_yield_day. Mirror that here so we don't show the ready
  // sprite until the plant has actually matured.
  const firstYield = CROP_FIRST_YIELD_DAY[crop] ?? 4;
  const age = day - plantedDay;
  if (age >= firstYield && yieldUnits >= 1) return 'ready';
  return age * 2 >= firstYield ? 'midgrowth' : 'sprout';
}

// Compute keys describing the desired DOM for each cell layer. Two cells with
// equal keys must produce identical DOM, so the per-step render can skip the
// write when nothing has changed -- otherwise the browser tears down and
// re-decodes each <img> every step, which shows up as visible flashing.

function objectPlan(tile: RawTile, day: number): { key: string; html: string } {
  if (!tile || typeof tile !== 'object') return { key: '', html: '' };
  if (tile.kind === 'PLANT') {
    const cropLower = String(tile.crop).toLowerCase();
    const stage = plantStage(tile.crop, tile.planted_day, day, tile.yield_units);
    const sprite = plantSprite(cropLower, stage);
    const fertilized = (tile.fertilized_until_day ?? -1) >= day;
    const label = `${titleCase(tile.crop)} (${stage}${fertilized ? ', fertilized' : ''})`;
    const yieldShown = stage === 'ready' ? tile.yield_units : 0;
    const yieldHtml = yieldShown > 0 ? `<span class="cell-yield">${yieldShown}</span>` : '';
    return {
      key: `plant:${cropLower}:${stage}:${fertilized ? 1 : 0}:y${yieldShown}`,
      html: `<img class="cell-sprite" src="${spriteSrc(sprite)}" alt="${sprite}" title="${label}" />${yieldHtml}`,
    };
  }
  if (tile.kind === 'WEED') {
    return {
      key: 'weed',
      html: `<img class="cell-sprite" src="${spriteSrc('weed')}" alt="weed" title="Weed" />`,
    };
  }
  if (tile.kind === 'COOP' || tile.kind === 'PASTURE') {
    const structure = tile.kind === 'COOP' ? 'coop' : 'pasture';
    const structureLabel = tile.kind === 'COOP' ? 'Coop' : 'Pasture';
    const animal = tile.animal ? tile.animal.toLowerCase() : '';
    const parts = [
      `<img class="cell-sprite" src="${spriteSrc(structure)}" alt="${structure}" title="${structureLabel}" />`,
    ];
    if (tile.animal) {
      parts.push(
        `<img class="cell-sprite cell-animal" src="${spriteSrc(animal)}" alt="${animal}" title="${titleCase(tile.animal)}" />`
      );
    }
    const yieldShown = tile.animal ? (tile.yield_units ?? 0) : 0;
    if (yieldShown > 0) parts.push(`<span class="cell-yield">${yieldShown}</span>`);
    return { key: `${structure}:${animal}:y${yieldShown}`, html: parts.join('') };
  }
  return { key: '', html: '' };
}

function applyAgentSlot(cellRef: CellRefs, key: string, html: string): void {
  if (cellRef.lastAgentKey === key) return;
  cellRef.lastAgentKey = key;
  if (html) cellRef.agentSlot.innerHTML = html;
  else clearChildren(cellRef.agentSlot);
}

function renderFarm(refs: PlayerRefs, playerNum: number, farm: FarmPublic, day: number): void {
  const unlocked = new Set(farm.unlocked_quadrants ?? ['NW']);
  const tiles = farm.tiles ?? [];

  // Pre-compute the desired agent for each cell so we can diff against the
  // cached state instead of wiping every agent slot first.
  const desiredAgents = new Map<CellRefs, { key: string; html: string }>();
  if (farm.farmer) {
    const [fx, fy] = farm.farmer;
    const cellRef = refs.cells[fy]?.[fx];
    if (cellRef) {
      const sprite = `farmer_p${playerNum}`;
      desiredAgents.set(cellRef, {
        key: `farmer:${playerNum}`,
        html: `<img class="cell-sprite cell-agent-sprite" src="${spriteSrc(sprite)}" alt="${sprite}" title="Farmer ${playerNum}" />`,
      });
    }
  }
  (farm.hands ?? []).forEach((pos, i) => {
    const [hx, hy] = pos;
    const cellRef = refs.cells[hy]?.[hx];
    if (!cellRef) return;
    const variant = (i % 3) + 1;
    const sprite = `farmhand_${variant}`;
    desiredAgents.set(cellRef, {
      key: `hand:${i}:${variant}`,
      html: `<img class="cell-sprite cell-agent-sprite" src="${spriteSrc(sprite)}" alt="${sprite}" title="Farmhand ${i + 1}" />`,
    });
  });

  for (let row = 0; row < refs.cells.length; row++) {
    const rowRefs = refs.cells[row];
    for (let col = 0; col < rowRefs.length; col++) {
      const cellRef = rowRefs[col];
      if (!cellRef) continue;
      const quadrant = QUADRANT_BY_SEGMENT[cellRef.segment];
      const isUnlocked = unlocked.has(quadrant);
      const tile: RawTile = isUnlocked ? (tiles[row]?.[col] ?? null) : 'LOCKED';

      let baseKey: string;
      let baseSprite: string;
      let obj: { key: string; html: string };
      if (!isUnlocked || tile === 'LOCKED') {
        baseKey = 'locked';
        baseSprite = 'locked_cell';
        obj = { key: '', html: '' };
      } else {
        const isPlant = tile && typeof tile === 'object' && tile.kind === 'PLANT';
        const watered = isPlant && (tile as any).watered_today;
        baseSprite = watered ? 'soil_watered' : 'soil_dry';
        baseKey = baseSprite;
        obj = objectPlan(tile, day);
      }

      if (cellRef.lastBaseKey !== baseKey) {
        cellRef.lastBaseKey = baseKey;
        cellRef.baseImg.src = spriteSrc(baseSprite);
        cellRef.baseImg.alt = baseSprite;
      }
      if (cellRef.lastObjectKey !== obj.key) {
        cellRef.lastObjectKey = obj.key;
        if (obj.html) cellRef.objectSlot.innerHTML = obj.html;
        else clearChildren(cellRef.objectSlot);
      }

      const desired = desiredAgents.get(cellRef);
      applyAgentSlot(cellRef, desired?.key ?? '', desired?.html ?? '');
    }
  }
}

function renderShed(refs: PlayerRefs, priv: PrivateState | undefined): void {
  const shed = priv?.shed ?? {};
  const seeds = priv?.seeds ?? {};
  const entries: [string, number, boolean][] = [];
  for (const [k, v] of Object.entries(shed)) {
    if (v > 0) entries.push([k, v, false]);
  }
  for (const [k, v] of Object.entries(seeds)) {
    if (v > 0) entries.push([k, v, true]);
  }

  refs.inventory.forEach((slot, i) => {
    const entry = entries[i];
    if (!entry) {
      if (slot.lastIconKey !== '') {
        slot.lastIconKey = '';
        clearChildren(slot.icon);
      }
      if (slot.lastCount !== '') {
        slot.lastCount = '';
        slot.count.textContent = '';
      }
      return;
    }
    const [item, qty, isSeed] = entry;
    const sprite = isSeed ? 'seed_packet' : item.toLowerCase();
    const alt = isSeed ? `${item} seed` : item.toLowerCase();
    const label = isSeed ? `${titleCase(item)} seed` : titleCase(item);
    const iconKey = isSeed ? `seed:${item}` : `item:${item}`;
    if (slot.lastIconKey !== iconKey) {
      slot.lastIconKey = iconKey;
      const cropOverlay = isSeed
        ? `<img class="item-seed-crop" src="${marketSpriteSrc(item.toLowerCase())}" alt="" aria-hidden="true" />`
        : '';
      slot.icon.innerHTML = `<img class="item-icon-img" src="${marketSpriteSrc(sprite)}" alt="${alt}" title="${label}" />${cropOverlay}`;
    }
    const qtyStr = String(qty);
    if (slot.lastCount !== qtyStr) {
      slot.lastCount = qtyStr;
      slot.count.textContent = qtyStr;
    }
  });
}

function renderPlayer(
  refs: PlayerRefs,
  idx: number,
  farm: FarmPublic,
  priv: PrivateState | undefined,
  day: number
): void {
  refs.balance.textContent = String(Math.floor(farm.money ?? 0));
  renderFarm(refs, idx + 1, farm, day);
  renderShed(refs, priv);
}

// SVG viewBox; CSS scales the element to its rendered size while preserving
// none aspect ratio, so we draw in these abstract units.
const SPARK_W = 100;
const SPARK_H = 20;
const SPARK_PAD = 2;

function sparklinePath(series: number[]): string {
  if (series.length === 0) return '';
  let min = series[0];
  let max = series[0];
  for (const v of series) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min;
  const yMid = SPARK_H / 2;
  const yTop = SPARK_PAD;
  const yBot = SPARK_H - SPARK_PAD;
  const step = series.length > 1 ? SPARK_W / (series.length - 1) : 0;
  let d = '';
  for (let i = 0; i < series.length; i++) {
    const x = i * step;
    const y = range === 0 ? yMid : yBot - ((series[i] - min) / range) * (yBot - yTop);
    d += (i === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2);
  }
  return d;
}

function renderMarket(refs: LayoutRefs, market: MarketPublic, priceHistory: Record<string, number[]>): void {
  const prices = market?.prices ?? {};
  for (const { key } of MARKET_ITEMS) {
    const slot = refs.marketItems[key];
    if (!slot) continue;
    const price = prices[key];
    slot.price.textContent = price == null ? '--' : String(Math.round(price));
    const series = priceHistory[key] ?? [];
    // Key off length + each value rounded to 2 decimals so we skip the SVG
    // write when the visible curve hasn't changed.
    const sparkKey = series.length + ':' + series.map((v) => v.toFixed(2)).join(',');
    if (slot.lastSparkKey !== sparkKey) {
      slot.lastSparkKey = sparkKey;
      slot.sparkPath.setAttribute('d', sparklinePath(series));
    }
  }
}

function renderTown(refs: LayoutRefs, town: TownPublic): void {
  const active = new Set(town?.unlocked_shops ?? []);
  // Look up by interpreter shop key.
  const buildingByShop = new Map<string, { sprite: string; label: string }>();
  for (const b of Object.values(SURROUNDING_BUILDINGS)) buildingByShop.set(b.shop, b);

  for (const slot of refs.shopSlots) {
    const shop = slot.dataset.building ?? '';
    const isActive = active.has(shop);
    const meta = buildingByShop.get(shop);
    const existing = slot.querySelector<HTMLImageElement>('.town-sprite');
    if (isActive && meta) {
      if (!existing) {
        const img = document.createElement('img');
        img.className = 'town-sprite';
        img.src = spriteSrc(meta.sprite);
        img.alt = meta.label;
        img.title = meta.label;
        // Insert before flower overlays so flowers stay on top.
        slot.insertBefore(img, slot.firstChild);
      }
    } else if (existing) {
      existing.remove();
    }
  }
}

function renderHeader(refs: LayoutRefs, view: ViewModel, cfg: any): void {
  const turnsPerDay = Number(cfg?.turnsPerDay) || 24;
  const totalDays = Math.max(1, Math.floor((Number(cfg?.episodeSteps) || 30 * turnsPerDay) / turnsPerDay));
  const dayText = String((view.day ?? 0) + 1);
  const turnText = String((view.hour ?? 0) + 1);
  const dayTotalText = String(totalDays);
  const turnTotalText = String(turnsPerDay);
  for (const el of refs.dayValues) {
    el.textContent = dayText;
    const total = el.parentElement?.querySelector<HTMLElement>('.day-total');
    if (total) total.textContent = dayTotalText;
  }
  for (const el of refs.turnValues) {
    el.textContent = turnText;
    const total = el.parentElement?.querySelector<HTMLElement>('.turn-total');
    if (total) total.textContent = turnTotalText;
  }
}

export function renderObservation(refs: LayoutRefs, view: ViewModel, cfg: any): void {
  renderHeader(refs, view, cfg);
  renderMarket(refs, view.market, view.priceHistory);
  renderTown(refs, view.town);
  const farms = view.farms ?? [];
  farms.forEach((farm, i) => {
    const playerRefs = refs.players[i];
    if (playerRefs) renderPlayer(playerRefs, i, farm, view.privates[i], view.day);
  });
}
