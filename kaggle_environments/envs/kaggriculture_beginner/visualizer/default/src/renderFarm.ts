import {
  CROP_SPRITE,
  CROPS,
  type BoardSize,
  type CellRefs,
  type Crop,
  type Farm,
  type LayoutRefs,
  type Observation,
  type PlayerRefs,
  type SeedSlotRefs,
} from './types';
import { makeHiddenImg, plantSprite, spriteSrc } from './utils';

export type { BoardSize, LayoutRefs } from './types';

function farmCell(row: number, col: number): string {
  return `
    <div class="cell" data-row="${row}" data-col="${col}">
      <div class="cell-base">
        <img class="cell-sprite" src="${spriteSrc('soil_dry')}" alt="soil_dry" />
      </div>
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
  </div>`;
}

function seedRow(): string {
  return `<div class="seed-row">
    ${CROPS.map(
      (crop) => `
      <div class="seed-slot" data-crop="${crop}">
        <img class="seed-icon" src="${spriteSrc(CROP_SPRITE[crop])}" alt="${crop}" />
        <span class="seed-count">0</span>
      </div>
    `
    ).join('')}
  </div>`;
}

function farmPanel(player: 1 | 2, rows: number, cols: number): string {
  return `
    <section class="farm-panel" data-player="${player}">
      <header class="farm-header sketched-border">
        <span class="player-name">
          <img class="player-name-icon" src="${spriteSrc(`farmer_p${player}`)}" alt="farmer p${player}" />
          Player ${player}
        </span>
        <span class="player-balance">
          <img class="balance-icon" src="${spriteSrc('coin')}" alt="coins" />
          <span class="balance-amount">0</span>
        </span>
      </header>
      <div class="farm-area">
        ${farmGrid(rows, cols)}
      </div>
      <div class="seed-area sketched-border">
        <div class="seed-header">Seeds</div>
        ${seedRow()}
      </div>
    </section>
  `;
}

function statusPanel(): string {
  return `
    <section class="status-panel sketched-border">
      <div class="status-title">Kaggriculture (Beginner)</div>
      <div class="status-counters">
        <span class="day-counter">Day <span class="day-value">0</span></span>
        <span class="turn-counter">Turn <span class="turn-value">0</span></span>
      </div>
    </section>
  `;
}

export function buildSkeleton(root: HTMLElement, board: BoardSize): void {
  root.innerHTML = `
    <div class="demo-container">
      <main class="demo-main">
        ${farmPanel(1, board.rows, board.cols)}
        ${statusPanel()}
        ${farmPanel(2, board.rows, board.cols)}
      </main>
    </div>
  `;
}

function collectPlayerRefs(panel: HTMLElement, board: BoardSize): PlayerRefs {
  const cells: CellRefs[][] = Array.from({ length: board.rows }, () => new Array<CellRefs>(board.cols));
  panel.querySelectorAll<HTMLElement>('.cell').forEach((el) => {
    const row = Number(el.dataset.row);
    const col = Number(el.dataset.col);
    const objectSlot = el.querySelector<HTMLElement>('.cell-object')!;
    const objectImg = makeHiddenImg();
    objectSlot.appendChild(objectImg);
    const agentSlot = el.querySelector<HTMLElement>('.cell-agent')!;
    const agentImg = makeHiddenImg('cell-agent-sprite');
    agentSlot.appendChild(agentImg);
    cells[row][col] = {
      el,
      baseImg: el.querySelector<HTMLImageElement>('.cell-base .cell-sprite')!,
      baseSprite: 'soil_dry',
      objectSlot,
      objectImg,
      objectSprite: '',
      agentSlot,
      agentImg,
      agentSprite: '',
    };
  });
  const seeds = {} as Record<Crop, SeedSlotRefs>;
  for (const crop of CROPS) {
    const slot = panel.querySelector<HTMLElement>(`.seed-slot[data-crop="${crop}"]`)!;
    seeds[crop] = {
      count: slot.querySelector<HTMLElement>('.seed-count')!,
      lastCount: -1,
    };
  }
  return {
    panel,
    balance: panel.querySelector<HTMLElement>('.balance-amount')!,
    lastBalance: Number.NaN,
    cells,
    seeds,
    farmerCell: null,
  };
}

export function collectRefs(root: HTMLElement, board: BoardSize): LayoutRefs {
  return {
    dayValue: root.querySelector<HTMLElement>('.day-value')!,
    lastDay: Number.NaN,
    turnValue: root.querySelector<HTMLElement>('.turn-value')!,
    lastTurn: Number.NaN,
    players: [1, 2].map((p) =>
      collectPlayerRefs(root.querySelector<HTMLElement>(`.farm-panel[data-player="${p}"]`)!, board)
    ),
  };
}

function setBaseSprite(ref: CellRefs, sprite: string): void {
  if (ref.baseSprite === sprite) return;
  ref.baseSprite = sprite;
  ref.baseImg.src = spriteSrc(sprite);
  ref.baseImg.alt = sprite;
}

function setObjectSprite(ref: CellRefs, sprite: string): void {
  if (ref.objectSprite === sprite) return;
  ref.objectSprite = sprite;
  if (sprite === '') {
    ref.objectImg.style.display = 'none';
    ref.objectImg.removeAttribute('src');
    return;
  }
  ref.objectImg.src = spriteSrc(sprite);
  ref.objectImg.alt = sprite;
  ref.objectImg.style.display = '';
}

function setAgentSprite(ref: CellRefs, sprite: string): void {
  if (ref.agentSprite === sprite) return;
  ref.agentSprite = sprite;
  if (sprite === '') {
    ref.agentImg.style.display = 'none';
    ref.agentImg.removeAttribute('src');
    return;
  }
  ref.agentImg.src = spriteSrc(sprite);
  ref.agentImg.alt = sprite;
  ref.agentImg.style.display = '';
}

function renderFarm(refs: PlayerRefs, playerNum: number, farm: Farm, day: number): void {
  for (let row = 0; row < refs.cells.length; row++) {
    const rowRefs = refs.cells[row];
    for (let col = 0; col < rowRefs.length; col++) {
      const cellRef = rowRefs[col];
      if (!cellRef) continue;
      const tile = farm.tiles?.[row]?.[col] ?? null;
      const watered = tile?.watered_today === true;
      setBaseSprite(cellRef, watered ? 'soil_watered' : 'soil_dry');
      if (tile) {
        const ageDays = day - tile.planted_day;
        setObjectSprite(cellRef, plantSprite(tile.crop, ageDays));
      } else {
        setObjectSprite(cellRef, '');
      }
    }
  }

  const [fx, fy] = farm.farmer ?? [0, 0];
  const newFarmerCell = refs.cells[fy]?.[fx] ?? null;
  if (refs.farmerCell !== newFarmerCell) {
    if (refs.farmerCell) setAgentSprite(refs.farmerCell, '');
    refs.farmerCell = newFarmerCell;
  }
  if (newFarmerCell) {
    setAgentSprite(newFarmerCell, `farmer_p${playerNum}`);
  }
}

function renderSeeds(refs: PlayerRefs, seeds: Record<Crop, number> | undefined): void {
  for (const crop of CROPS) {
    const slot = refs.seeds[crop];
    const count = seeds?.[crop] ?? 0;
    if (slot.lastCount === count) continue;
    slot.lastCount = count;
    slot.count.textContent = String(count);
  }
}

function renderPlayer(refs: PlayerRefs, idx: number, farm: Farm, day: number): void {
  const balance = Math.round(farm.money ?? 0);
  if (refs.lastBalance !== balance) {
    refs.lastBalance = balance;
    refs.balance.textContent = String(balance);
  }
  renderFarm(refs, idx + 1, farm, day);
  renderSeeds(refs, farm.seeds);
}

function renderHeader(refs: LayoutRefs, obs: Observation): void {
  const day = obs.day ?? 0;
  if (refs.lastDay !== day) {
    refs.lastDay = day;
    refs.dayValue.textContent = String(day);
  }
  const turn = obs.hour ?? 0;
  if (refs.lastTurn !== turn) {
    refs.lastTurn = turn;
    refs.turnValue.textContent = String(turn);
  }
}

export function renderObservation(refs: LayoutRefs, obs: Observation): void {
  renderHeader(refs, obs);
  const farms = obs.farms ?? [];
  const day = obs.day ?? 0;
  farms.forEach((f, i) => {
    const playerRefs = refs.players[i];
    if (playerRefs) renderPlayer(playerRefs, i, f, day);
  });
}
