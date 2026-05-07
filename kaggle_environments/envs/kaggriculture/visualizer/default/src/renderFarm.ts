import {
  INVENTORY_SLOTS,
  MARKET_ITEMS,
  SEGMENT,
  SURROUNDING_BUILDINGS,
  TOWN_CENTER_INDEX,
  TOWN_GRID_COLS,
  TOWN_GRID_ROWS,
  type BoardSize,
  type Cell,
  type CellRefs,
  type LayoutRefs,
  type MarketEntry,
  type Observation,
  type PlayerRefs,
  type PlayerState,
} from './types';
import { BG_URLS, clearChildren, plantSprite, setCellSprite, spriteSrc } from './utils';

export type { BoardSize, LayoutRefs } from './types';

const BG_GRASS = `background-image:url(${BG_URLS.grass})`;
const BG_WOOD = `background-image:url(${BG_URLS.wood})`;
const BG_COBBLE = `background-image:url(${BG_URLS.cobble});background-size:100% 100%;image-rendering:pixelated;`;

function marketList(): string {
  return MARKET_ITEMS.map(
    ({ sprite, key }) => `
    <div class="market-item" data-item="${key}">
      <img class="market-item-icon" src="${spriteSrc(sprite)}" alt="${sprite}" />
      <div class="market-price-row">
        <img class="market-coin" src="${spriteSrc('coin')}" alt="coins" />
        <span class="market-price">--</span>
      </div>
    </div>
  `
  ).join('');
}

function farmCell(row: number, col: number): string {
  const segR = Math.floor(row / SEGMENT);
  const segC = Math.floor(col / SEGMENT);
  const segId = segR * 2 + segC;
  return `
    <div class="cell" data-row="${row}" data-col="${col}" data-segment="${segId}">
      <div class="cell-base">
        <img class="cell-sprite" src="${spriteSrc('locked_cell')}" alt="locked_cell" />
      </div>
      <div class="cell-overlay"></div>
      <div class="cell-object"></div>
      <div class="cell-agent"></div>
    </div>
  `;
}

function farmGrid(rows: number, cols: number): string {
  const cells: string[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      cells.push(farmCell(r, c));
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
      <header class="farm-header sketched-border" style="${BG_WOOD}">
        <span class="player-name">
          <img class="player-name-icon" src="${spriteSrc(`farmer_p${player}`)}" alt="farmer p${player}" />
          <span class="player-name-text">${escapeHtml(name)}</span>
        </span>
        <span class="player-balance">
          <img class="balance-icon" src="${spriteSrc('coin')}" alt="coins" />
          <span class="balance-amount">0</span>
        </span>
      </header>
      <div class="farm-area">
        ${farmGrid(rows, cols)}
      </div>
      <div class="shed-area sketched-border" style="${BG_WOOD}">
        <div class="shed-header">Shed</div>
        ${inventoryGrid()}
      </div>
    </section>
  `;
}

function townPanel(): string {
  return `
    <section class="town-panel">
      <header class="town-header sketched-border" style="${BG_WOOD}">
        <div class="town-title">Welcome to Kernel Cove</div>
        <div class="town-subheader">
          <span class="day-counter">Day <span class="day-value">1</span> / 30</span>
          <span class="turn-counter">Turn <span class="turn-value">1</span> / 24</span>
        </div>
      </header>
      <div class="market-panel sketched-border" style="${BG_WOOD}">
        <div class="market-header">Market Prices</div>
        <div class="market-list">${marketList()}</div>
      </div>
      <div class="town-grid" style="grid-template-columns: repeat(${TOWN_GRID_COLS}, 1fr);">
        ${Array.from({ length: TOWN_GRID_COLS * TOWN_GRID_ROWS }, (_, i) => {
          if (i === TOWN_CENTER_INDEX) {
            return `<div class="town-slot town-slot--center" data-slot="${i}" style="${BG_COBBLE}">
                      <img class="town-sprite" src="${spriteSrc('town_center')}" alt="Town Center" />
                    </div>`;
          }
          const building = SURROUNDING_BUILDINGS[i];
          if (building) {
            return `<div class="town-slot town-slot--shop" data-slot="${i}" data-building="${building}" style="${BG_COBBLE}">
                      <img class="town-sprite" src="${spriteSrc(building)}" alt="${building} shop" />
                    </div>`;
          }
          return `<div class="town-slot" data-slot="${i}"></div>`;
        }).join('')}
      </div>
    </section>
  `;
}

export function buildShell(root: HTMLElement, board: BoardSize, playerNames: string[]): void {
  root.innerHTML = `
    <div class="demo-container" style="${BG_GRASS}">
      <main class="demo-main">
        ${farmPanel(1, board.rows, board.cols, playerNames[0] ?? 'Player 1')}
        ${townPanel()}
        ${farmPanel(2, board.rows, board.cols, playerNames[1] ?? 'Player 2')}
      </main>
    </div>
  `;
}

// --- Cached refs --------------------------------------------------------------
// The shell DOM (background, panels, shed/inventory boxes, market layout, town
// grid scaffolding) is built once per board size and never re-rendered. We
// walk it once to grab handles to every element that *does* update per step,
// so the per-step render is a series of direct property writes — no
// querySelector calls in the hot path.

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
  const marketItems: Record<string, { item: HTMLElement; price: HTMLElement }> = {};
  for (const { key } of MARKET_ITEMS) {
    const item = root.querySelector<HTMLElement>(`.market-item[data-item="${key}"]`)!;
    marketItems[key] = {
      item,
      price: item.querySelector<HTMLElement>('.market-price')!,
    };
  }
  return {
    dayValue: root.querySelector<HTMLElement>('.day-value')!,
    turnValue: root.querySelector<HTMLElement>('.turn-value')!,
    marketItems,
    shopSlots: Array.from(root.querySelectorAll<HTMLElement>('.town-slot--shop')),
    players: [1, 2].map((p) =>
      collectPlayerRefs(root.querySelector<HTMLElement>(`.farm-panel[data-player="${p}"]`)!, board)
    ),
  };
}

function renderFarm(refs: PlayerRefs, playerNum: number, player: PlayerState): void {
  const unlocked = new Set(player.farm.unlockedSegments ?? [0]);
  const cellMap = new Map<string, Cell>();
  for (const c of player.farm.cells ?? []) {
    cellMap.set(`${c.row},${c.col}`, c);
  }

  for (let row = 0; row < refs.cells.length; row++) {
    const rowRefs = refs.cells[row];
    for (let col = 0; col < rowRefs.length; col++) {
      const cellRef = rowRefs[col];
      if (!cellRef) continue;
      const isUnlocked = unlocked.has(cellRef.segment);

      clearChildren(cellRef.objectSlot);
      clearChildren(cellRef.agentSlot);

      const cell = isUnlocked ? cellMap.get(`${row},${col}`) : undefined;

      if (!isUnlocked) {
        cellRef.baseImg.src = spriteSrc('locked_cell');
        cellRef.baseImg.alt = 'locked_cell';
      } else {
        const watered = cell && cell.type === 'plant' && cell.wateredToday;
        const sprite = watered ? 'soil_watered' : 'soil_dry';
        cellRef.baseImg.src = spriteSrc(sprite);
        cellRef.baseImg.alt = sprite;
      }

      if (!cell) continue;

      if (cell.type === 'plant') {
        const sprite = plantSprite(cell.plant, cell.stage);
        const fertilized = (cell.fertilizedDaysLeft ?? 0) > 0;
        setCellSprite(cellRef.objectSlot, spriteSrc(sprite), sprite, fertilized ? 'cell-fertilized' : '');
      } else if (cell.type === 'weed') {
        setCellSprite(cellRef.objectSlot, spriteSrc('weed'), 'weed');
      } else if (cell.type === 'coop' || cell.type === 'pasture') {
        const structure = cell.type === 'coop' ? 'coop' : 'pasture';
        const parts: string[] = [
          `<img class="cell-sprite cell-structure" src="${spriteSrc(structure)}" alt="${structure}" />`,
        ];
        if (cell.animal) {
          parts.push(
            `<img class="cell-sprite cell-animal" src="${spriteSrc(cell.animal.kind)}" alt="${cell.animal.kind}" />`
          );
        }
        cellRef.objectSlot.innerHTML = parts.join('');
      }
    }
  }

  for (const agent of player.agents ?? []) {
    const cellRef = refs.cells[agent.row]?.[agent.col];
    if (!cellRef) continue;
    const sprite = agent.role === 'farmer' ? `farmer_p${playerNum}` : `farmhand_${agent.variant ?? 1}`;
    cellRef.agentSlot.innerHTML = `<img class="cell-sprite cell-agent-sprite" src="${spriteSrc(sprite)}" alt="${sprite}" />`;
  }
}

function renderShed(refs: PlayerRefs, shed: Record<string, number> | undefined): void {
  const entries = Object.entries(shed ?? {});
  refs.inventory.forEach((slot, i) => {
    const entry = entries[i];
    if (!entry) {
      clearChildren(slot.icon);
      slot.count.textContent = '';
      return;
    }
    const [item, qty] = entry;
    slot.icon.innerHTML = `<img class="item-icon-img" src="${spriteSrc(item)}" alt="${item}" />`;
    slot.count.textContent = String(qty);
  });
}

function renderPlayer(refs: PlayerRefs, idx: number, player: PlayerState): void {
  refs.balance.textContent = String(player.coins ?? 0);
  renderFarm(refs, idx + 1, player);
  renderShed(refs, player.shed);
}

function renderMarket(refs: LayoutRefs, market: Record<string, MarketEntry> | undefined): void {
  if (!market) return;
  for (const { key } of MARKET_ITEMS) {
    const entry = market[key];
    const slot = refs.marketItems[key];
    if (!entry || !slot) continue;
    slot.price.textContent = String(entry.current);
    slot.item.classList.toggle('market-item--down', entry.current < entry.base);
  }
}

function renderTown(refs: LayoutRefs, town: { active: string[] } | undefined): void {
  const active = new Set(town?.active ?? []);
  for (const slot of refs.shopSlots) {
    const building = slot.dataset.building ?? '';
    slot.classList.toggle('town-slot--inactive', active.size > 0 && !active.has(building));
  }
}

function renderHeader(refs: LayoutRefs, obs: Observation): void {
  refs.dayValue.textContent = String(obs.day ?? 1);
  refs.turnValue.textContent = String(obs.turnOfDay ?? 1);
}

export function renderObservation(refs: LayoutRefs, obs: Observation): void {
  renderHeader(refs, obs);
  renderMarket(refs, obs.market);
  renderTown(refs, obs.townBuildings);
  const players = obs.players ?? [];
  players.forEach((p, i) => {
    const playerRefs = refs.players[i];
    if (playerRefs) renderPlayer(playerRefs, i, p);
  });
}
