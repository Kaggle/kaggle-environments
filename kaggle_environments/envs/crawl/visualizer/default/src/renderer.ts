import type { RendererOptions } from '@kaggle-environments/core';

// Wall bitfield constants
const WALL_N = 1;
const WALL_E = 2;
const WALL_S = 4;
const WALL_W = 8;

// Walls workers cannot build or remove (mirrors crawl.py is_fixed_wall):
// E/W perimeter and the central mirror axis. Rendered as double lines.
function isFixedWall(col: number, dir: 'E' | 'W', width: number): boolean {
  const half = Math.floor(width / 2);
  if (dir === 'W' && col === 0) return true;
  if (dir === 'E' && col === width - 1) return true;
  if (dir === 'E' && col === half - 1) return true;
  if (dir === 'W' && col === half) return true;
  return false;
}

function drawDoubleLine(c: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number): void {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len === 0) return;
  const nx = -dy / len;
  const ny = dx / len;
  const off = 1.5;
  c.beginPath();
  c.moveTo(x1 + nx * off, y1 + ny * off);
  c.lineTo(x2 + nx * off, y2 + ny * off);
  c.moveTo(x1 - nx * off, y1 - ny * off);
  c.lineTo(x2 - nx * off, y2 - ny * off);
  c.stroke();
}

// Robot type constants
const FACTORY = 0;
const SCOUT = 1;
const WORKER = 2;
const MINER = 3;

// Player colors
const PLAYER_COLORS = ['#2196F3', '#F44336']; // blue, red
const PLAYER_COLORS_LIGHT = ['rgba(33,150,243,0.2)', 'rgba(244,67,54,0.2)'];
// Robot shape drawing helpers
const TYPE_LABELS: Record<number, string> = {
  [FACTORY]: 'F',
  [SCOUT]: 'S',
  [WORKER]: 'W',
  [MINER]: 'M',
};

interface RobotData {
  type: number;
  col: number;
  row: number;
  energy: number;
  owner: number;
}

interface MineData {
  energy: number;
  maxEnergy: number;
  owner: number;
}

type CrawlStep = {
  observation: Record<string, any>;
  action: Record<string, any> | null;
  reward: number | null;
  info: Record<string, any>;
  status: string;
}[];

type CrawlOptions = RendererOptions<CrawlStep[]>;

function parseRobots(globalRobots: Record<string, number[]>): RobotData[] {
  if (!globalRobots) return [];
  return Object.values(globalRobots).map((d) => ({
    type: d[0],
    col: d[1],
    row: d[2],
    energy: d[3],
    owner: d[4],
  }));
}

function parseMines(globalMines: Record<string, number[]>): Map<string, MineData> {
  const mines = new Map<string, MineData>();
  if (!globalMines) return mines;
  for (const [key, d] of Object.entries(globalMines)) {
    mines.set(key, { energy: d[0], maxEnergy: d[1], owner: d[2] });
  }
  return mines;
}

function drawRobot(c: CanvasRenderingContext2D, x: number, y: number, size: number, robot: RobotData) {
  const color = PLAYER_COLORS[robot.owner];
  const lightColor = PLAYER_COLORS_LIGHT[robot.owner];
  const cx = x + size / 2;
  const cy = y + size / 2;
  const r = size * 0.35;

  if (robot.type === FACTORY) {
    // Factory: filled square with border
    const s = size * 0.6;
    c.fillStyle = lightColor;
    c.strokeStyle = color;
    c.lineWidth = 2;
    c.fillRect(cx - s / 2, cy - s / 2, s, s);
    c.strokeRect(cx - s / 2, cy - s / 2, s, s);
    // Gear-like notches
    const notch = s * 0.15;
    c.fillStyle = color;
    c.fillRect(cx - notch, cy - s / 2 - notch, notch * 2, notch);
    c.fillRect(cx - notch, cy + s / 2, notch * 2, notch);
    c.fillRect(cx - s / 2 - notch, cy - notch, notch, notch * 2);
    c.fillRect(cx + s / 2, cy - notch, notch, notch * 2);
  } else if (robot.type === SCOUT) {
    // Scout: small diamond
    c.fillStyle = lightColor;
    c.strokeStyle = color;
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(cx, cy - r);
    c.lineTo(cx + r, cy);
    c.lineTo(cx, cy + r);
    c.lineTo(cx - r, cy);
    c.closePath();
    c.fill();
    c.stroke();
  } else if (robot.type === WORKER) {
    // Worker: hexagon
    c.fillStyle = lightColor;
    c.strokeStyle = color;
    c.lineWidth = 1.5;
    c.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 6;
      const px = cx + r * Math.cos(angle);
      const py = cy + r * Math.sin(angle);
      if (i === 0) c.moveTo(px, py);
      else c.lineTo(px, py);
    }
    c.closePath();
    c.fill();
    c.stroke();
  } else if (robot.type === MINER) {
    // Miner: circle
    c.fillStyle = lightColor;
    c.strokeStyle = color;
    c.lineWidth = 1.5;
    c.beginPath();
    c.arc(cx, cy, r, 0, Math.PI * 2);
    c.closePath();
    c.fill();
    c.stroke();
  }

  // Type letter
  c.fillStyle = color;
  c.font = `bold ${Math.max(10, size * 0.35)}px Inter, sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.fillText(TYPE_LABELS[robot.type], cx, cy);

  // Energy bar below robot
  const barW = size * 0.7;
  const barH = Math.max(2, size * 0.08);
  const barX = cx - barW / 2;
  const barY = y + size - barH - 1;
  const maxE = robot.type === FACTORY ? 1000 : robot.type === SCOUT ? 50 : robot.type === WORKER ? 200 : 300;
  const pct = Math.min(1, robot.energy / maxE);
  c.fillStyle = '#ddd';
  c.fillRect(barX, barY, barW, barH);
  c.fillStyle = pct > 0.3 ? '#4CAF50' : pct > 0.1 ? '#FF9800' : '#F44336';
  c.fillRect(barX, barY, barW * pct, barH);
}

function drawMiningNode(c: CanvasRenderingContext2D, x: number, y: number, size: number) {
  const cx = x + size / 2;
  const cy = y + size / 2;
  const r = size * 0.22;

  // Dashed triangle outline, no fill, neutral color — signals an unclaimed
  // mining node that a worker could transform into a mine.
  c.strokeStyle = '#5a5a5a';
  c.lineWidth = 1.25;
  c.setLineDash([3, 2]);
  c.beginPath();
  c.moveTo(cx, cy - r);
  c.lineTo(cx + r, cy + r * 0.7);
  c.lineTo(cx - r, cy + r * 0.7);
  c.closePath();
  c.stroke();
  c.setLineDash([]);
}

function drawMine(c: CanvasRenderingContext2D, x: number, y: number, size: number, mine: MineData) {
  const cx = x + size / 2;
  const cy = y + size / 2;
  const r = size * 0.25;
  const color = PLAYER_COLORS[mine.owner];

  // Triangle pointing up
  c.fillStyle = color;
  c.globalAlpha = 0.4;
  c.beginPath();
  c.moveTo(cx, cy - r);
  c.lineTo(cx + r, cy + r * 0.7);
  c.lineTo(cx - r, cy + r * 0.7);
  c.closePath();
  c.fill();
  c.globalAlpha = 1;
  c.strokeStyle = color;
  c.lineWidth = 1.5;
  c.stroke();

  // Energy fill indicator
  const pct = mine.maxEnergy > 0 ? mine.energy / mine.maxEnergy : 0;
  const barW = size * 0.5;
  const barH = Math.max(2, size * 0.06);
  const barX = cx - barW / 2;
  const barY = y + size - barH - 1;
  c.fillStyle = '#ddd';
  c.fillRect(barX, barY, barW, barH);
  c.fillStyle = '#FFD700';
  c.fillRect(barX, barY, barW * pct, barH);
}

function drawCrystal(c: CanvasRenderingContext2D, x: number, y: number, size: number, energy: number) {
  const cx = x + size / 2;
  const cy = y + size / 2;
  const r = size * 0.34;

  // Sun rays radiating outward
  c.strokeStyle = 'rgba(218, 165, 32, 0.7)';
  c.lineWidth = 1.25;
  const rayInner = r * 1.1;
  const rayOuter = r * 1.55;
  for (let i = 0; i < 8; i++) {
    const ang = (Math.PI / 4) * i;
    const ix = cx + Math.cos(ang) * rayInner;
    const iy = cy + Math.sin(ang) * rayInner;
    const ox = cx + Math.cos(ang) * rayOuter;
    const oy = cy + Math.sin(ang) * rayOuter;
    c.beginPath();
    c.moveTo(ix, iy);
    c.lineTo(ox, oy);
    c.stroke();
  }

  // Crystal shape
  c.fillStyle = 'rgba(255, 215, 0, 0.6)';
  c.strokeStyle = '#DAA520';
  c.lineWidth = 1.25;
  c.beginPath();
  c.moveTo(cx, cy - r * 1.2);
  c.lineTo(cx + r, cy);
  c.lineTo(cx, cy + r * 0.8);
  c.lineTo(cx - r, cy);
  c.closePath();
  c.fill();
  c.stroke();

  // Energy value, centered, dark for legibility
  if (size > 14) {
    c.fillStyle = '#050001';
    c.font = `bold ${Math.max(8, size * 0.28)}px Inter, sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(`${energy}`, cx, cy);
  }
}

// Persistent UI state across step changes. The renderer is invoked anew on
// every step, so we keep the selected fog-of-war view at module scope.
type ViewMode = 'all' | 'p0' | 'p1';
let viewMode: ViewMode = 'all';
let lastOptions: CrawlOptions | null = null;

(window as any).__crawlSetView = (mode: ViewMode) => {
  viewMode = mode;
  if (lastOptions) renderer(lastOptions);
};

export function renderer(options: CrawlOptions) {
  lastOptions = options;
  const { step, replay, parent, agents } = options;
  const steps = replay.steps;

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <canvas></canvas>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;
  if (!canvas || !replay) return;

  const currentStep = steps[step] as CrawlStep;
  if (!currentStep || !currentStep[0]) return;

  const obs = currentStep[0].observation;
  const config = replay.configuration;
  const width: number = config.width ?? 20;
  const southBound: number = obs.southBound ?? 0;
  const northBound: number = obs.northBound ?? 29;
  const windowHeight = northBound - southBound + 1;

  // Pick per-view observations. In 'all' mode we use the omniscient global
  // state; in 'p0'/'p1' mode we use that player's restricted observation
  // (visible robots/crystals/mines/nodes + a per-cell wall list where -1
  // means "never explored").
  const playerObs = viewMode === 'p1' ? currentStep[1]?.observation : obs;
  const fogged = viewMode !== 'all' && playerObs;

  // Walls: omniscient version always available; player wall list is per-cell.
  const globalWalls: Record<string, number[]> = obs.globalWalls ?? {};
  const playerWallsFlat: number[] = fogged ? (playerObs.walls ?? []) : [];

  // Per-player currently-visible wall lists, used for the "All" view overlay
  // that tints each cell by which player(s) can currently see it.
  const p0WallsFlat: number[] = currentStep[0]?.observation?.walls ?? [];
  const p1WallsFlat: number[] = currentStep[1]?.observation?.walls ?? [];

  const globalCrystals: Record<string, number> = obs.globalCrystals ?? {};
  const globalMiningNodes: Record<string, number> = obs.globalMiningNodes ?? {};
  const robots = parseRobots(fogged ? playerObs.robots : obs.globalRobots);
  const mines = parseMines(fogged ? playerObs.mines : obs.globalMines);

  // Per-view collections used below for drawing.
  const drawCrystals: Record<string, number> = fogged ? (playerObs.crystals ?? {}) : globalCrystals;
  const drawMiningNodes: Record<string, number> = fogged ? (playerObs.miningNodes ?? {}) : globalMiningNodes;

  // Discovered cells (set of "col,row" strings) — only meaningful in fogged mode.
  const discoveredSet = new Set<string>();
  if (fogged) {
    const playerIdx = viewMode === 'p1' ? 1 : 0;
    // discoveredCells lives on player 0's observation as [p0_cells, p1_cells]
    const allDiscovered = obs.discoveredCells;
    const myCells = Array.isArray(allDiscovered) ? allDiscovered[playerIdx] : null;
    if (Array.isArray(myCells)) {
      for (const cell of myCells) {
        if (Array.isArray(cell) && cell.length >= 2) {
          discoveredSet.add(`${cell[0]},${cell[1]}`);
        }
      }
    }
  }

  // Helper: is a cell in the player's currently-visible (lit) area?
  // We treat "wall list != -1 for that cell" as currently visible.
  const isLit = (col: number, row: number): boolean => {
    if (!fogged) return true;
    if (row < southBound || row > northBound) return false;
    const idx = (row - southBound) * width + col;
    return playerWallsFlat[idx] !== undefined && playerWallsFlat[idx] !== -1;
  };
  const isDiscovered = (col: number, row: number): boolean => {
    if (!fogged) return true;
    return isLit(col, row) || discoveredSet.has(`${col},${row}`);
  };

  // Compute energy totals for each player
  const totalEnergy = [0, 0];
  const robotCounts = [
    { factory: 0, scout: 0, worker: 0, miner: 0 },
    { factory: 0, scout: 0, worker: 0, miner: 0 },
  ];
  for (const r of robots) {
    totalEnergy[r.owner] += r.energy;
    if (r.type === FACTORY) robotCounts[r.owner].factory++;
    else if (r.type === SCOUT) robotCounts[r.owner].scout++;
    else if (r.type === WORKER) robotCounts[r.owner].worker++;
    else if (r.type === MINER) robotCounts[r.owner].miner++;
  }

  // --- Header ---
  const getName = (idx: number) => {
    const agent = agents?.find((a: any) => a.index === idx);
    return agent?.name || `Player ${idx + 1}`;
  };
  const isGameOver = currentStep.every((s) => s.status === 'DONE');

  const viewBtn = (mode: ViewMode, label: string) => `
    <button onclick="window.__crawlSetView('${mode}')" class="view-btn ${viewMode === mode ? 'active' : ''}">${label}</button>
  `;

  header.innerHTML = `
    <div class="player-card ${!isGameOver ? 'active' : ''}" style="border-left: 3px solid ${PLAYER_COLORS[0]}">
      <span>${getName(0)}</span>
      <span class="energy">E: ${totalEnergy[0]} | F:${robotCounts[0].factory} S:${robotCounts[0].scout} W:${robotCounts[0].worker} M:${robotCounts[0].miner}</span>
    </div>
    <span style="color: #444343; font-size: 0.9rem">vs</span>
    <div class="player-card ${!isGameOver ? 'active' : ''}" style="border-left: 3px solid ${PLAYER_COLORS[1]}">
      <span>${getName(1)}</span>
      <span class="energy">E: ${totalEnergy[1]} | F:${robotCounts[1].factory} S:${robotCounts[1].scout} W:${robotCounts[1].worker} M:${robotCounts[1].miner}</span>
    </div>
    <div class="view-toggle">
      <span style="font-size: 0.75rem; color: #444343;">View:</span>
      ${viewBtn('all', 'All')}
      ${viewBtn('p0', 'P1')}
      ${viewBtn('p1', 'P2')}
    </div>
  `;

  // --- Canvas rendering ---
  canvas.width = 0;
  canvas.height = 0;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;

  const c = canvas.getContext('2d');
  if (!c) return;

  // Calculate cell size to fit the window
  const maxCanvasW = rect.width;
  const maxCanvasH = rect.height;
  const cellW = maxCanvasW / width;
  const cellH = maxCanvasH / windowHeight;
  const cellSize = Math.min(cellW, cellH);

  const gridW = cellSize * width;
  const gridH = cellSize * windowHeight;
  const offsetX = (maxCanvasW - gridW) / 2;
  const offsetY = (maxCanvasH - gridH) / 2;

  // Helper: cell position to canvas coordinates
  const cellToCanvas = (col: number, row: number) => {
    // Row increases northward; canvas Y increases downward, so flip
    const gridRow = northBound - row; // 0 = top of grid
    return {
      x: offsetX + col * cellSize,
      y: offsetY + gridRow * cellSize,
    };
  };

  // Clear canvas (transparent so parchment shows through)
  c.clearRect(0, 0, canvas.width, canvas.height);

  // Draw cells
  for (let row = southBound; row <= northBound; row++) {
    const rowKey = String(row);
    const rowWalls = globalWalls[rowKey];

    for (let col = 0; col < width; col++) {
      const { x, y } = cellToCanvas(col, row);
      let w = rowWalls ? rowWalls[col] : 0;

      const lit = isLit(col, row);
      const discovered = isDiscovered(col, row);

      // In fogged mode, hide walls in cells the player has never seen, and
      // tint the cell to convey unexplored / remembered / lit state.
      if (fogged) {
        if (!discovered) {
          // Unexplored: don't draw cell background or any walls — fully dark.
          c.fillStyle = 'rgba(40, 36, 28, 0.55)';
          c.fillRect(x, y, cellSize, cellSize);
          continue;
        }
        // For wall data in fogged mode, prefer the player's per-cell wall list
        // if the cell is currently lit. Cells that are remembered (discovered
        // but not lit) reuse the global wall data we still need to know which
        // walls exist there — that's a small fairness compromise for clarity.
        if (lit) {
          const idx = (row - southBound) * width + col;
          const pw = playerWallsFlat[idx];
          if (typeof pw === 'number' && pw >= 0) w = pw;
        }
        c.fillStyle = lit ? 'rgba(255, 255, 255, 0.15)' : 'rgba(40, 36, 28, 0.18)';
      } else {
        // All view: tint each cell by which player(s) currently see it, so
        // you can see fog-of-war coverage at a glance without switching views.
        const idx = (row - southBound) * width + col;
        const p0Sees = p0WallsFlat[idx] !== undefined && p0WallsFlat[idx] !== -1;
        const p1Sees = p1WallsFlat[idx] !== undefined && p1WallsFlat[idx] !== -1;
        if (p0Sees && p1Sees) {
          c.fillStyle = 'rgba(160, 90, 200, 0.18)'; // both — purple blend
        } else if (p0Sees) {
          c.fillStyle = 'rgba(33, 150, 243, 0.18)'; // p0 — blue
        } else if (p1Sees) {
          c.fillStyle = 'rgba(244, 67, 54, 0.18)'; // p1 — red
        } else {
          c.fillStyle = 'rgba(255, 255, 255, 0.15)'; // unseen by either — neutral
        }
      }
      c.fillRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);

      // Draw walls. Fixed walls (perimeter + middle axis) draw as double
      // lines to signal that workers cannot remove them.
      c.strokeStyle = '#3c3b37';
      c.lineWidth = 2;

      if (w & WALL_N) {
        c.beginPath();
        c.moveTo(x, y);
        c.lineTo(x + cellSize, y);
        c.stroke();
      }
      if (w & WALL_S) {
        c.beginPath();
        c.moveTo(x, y + cellSize);
        c.lineTo(x + cellSize, y + cellSize);
        c.stroke();
      }
      if (w & WALL_E) {
        if (isFixedWall(col, 'E', width)) {
          drawDoubleLine(c, x + cellSize, y, x + cellSize, y + cellSize);
        } else {
          c.beginPath();
          c.moveTo(x + cellSize, y);
          c.lineTo(x + cellSize, y + cellSize);
          c.stroke();
        }
      }
      if (w & WALL_W) {
        if (isFixedWall(col, 'W', width)) {
          drawDoubleLine(c, x, y, x, y + cellSize);
        } else {
          c.beginPath();
          c.moveTo(x, y);
          c.lineTo(x, y + cellSize);
          c.stroke();
        }
      }
    }
  }

  // Eastern + western perimeters: always draw as two clearly-spaced parallel
  // lines so it's obvious workers cannot remove them. We push both strokes
  // inward (the maze can sit flush against the canvas edge), so the outer
  // line at the literal edge isn't half-clipped.
  c.strokeStyle = '#3c3b37';
  c.lineWidth = 2;
  const perimInner = 1;
  const perimGap = 5;
  // Left perimeter
  c.beginPath();
  c.moveTo(offsetX + perimInner, offsetY);
  c.lineTo(offsetX + perimInner, offsetY + gridH);
  c.moveTo(offsetX + perimInner + perimGap, offsetY);
  c.lineTo(offsetX + perimInner + perimGap, offsetY + gridH);
  c.stroke();
  // Right perimeter
  c.beginPath();
  c.moveTo(offsetX + gridW - perimInner, offsetY);
  c.lineTo(offsetX + gridW - perimInner, offsetY + gridH);
  c.moveTo(offsetX + gridW - perimInner - perimGap, offsetY);
  c.lineTo(offsetX + gridW - perimInner - perimGap, offsetY + gridH);
  c.stroke();

  // Draw center divider line (dashed, subtle)
  const half = width / 2;
  c.strokeStyle = 'rgba(60, 59, 55, 0.3)';
  c.lineWidth = 1;
  c.setLineDash([4, 4]);
  const divX = offsetX + half * cellSize;
  c.beginPath();
  c.moveTo(divX, offsetY);
  c.lineTo(divX, offsetY + gridH);
  c.stroke();
  c.setLineDash([]);

  // Draw crystals
  for (const [key, energy] of Object.entries(drawCrystals)) {
    const [colStr, rowStr] = key.split(',');
    const col = parseInt(colStr);
    const row = parseInt(rowStr);
    if (row < southBound || row > northBound) continue;
    const { x, y } = cellToCanvas(col, row);
    drawCrystal(c, x, y, cellSize, energy);
  }

  // Draw mining nodes (unbuilt spots where a worker could create a mine)
  for (const key of Object.keys(drawMiningNodes)) {
    const [colStr, rowStr] = key.split(',');
    const col = parseInt(colStr);
    const row = parseInt(rowStr);
    if (row < southBound || row > northBound) continue;
    const { x, y } = cellToCanvas(col, row);
    drawMiningNode(c, x, y, cellSize);
  }

  // Draw mines
  for (const [key, mine] of mines.entries()) {
    const [colStr, rowStr] = key.split(',');
    const col = parseInt(colStr);
    const row = parseInt(rowStr);
    if (row < southBound || row > northBound) continue;
    const { x, y } = cellToCanvas(col, row);
    drawMine(c, x, y, cellSize, mine);
  }

  // Draw robots (factories first, then others on top)
  const sortedRobots = [...robots].sort((a, b) => {
    if (a.type === FACTORY && b.type !== FACTORY) return -1;
    if (a.type !== FACTORY && b.type === FACTORY) return 1;
    return 0;
  });

  for (const robot of sortedRobots) {
    if (robot.row < southBound || robot.row > northBound) continue;
    const { x, y } = cellToCanvas(robot.col, robot.row);
    drawRobot(c, x, y, cellSize, robot);
  }

  // Draw scroll boundary indicator at bottom
  c.fillStyle = 'rgba(244, 67, 54, 0.15)';
  const boundaryY = offsetY + gridH - cellSize;
  c.fillRect(offsetX, boundaryY, gridW, cellSize);
  c.strokeStyle = 'rgba(244, 67, 54, 0.5)';
  c.lineWidth = 2;
  c.setLineDash([6, 3]);
  c.beginPath();
  c.moveTo(offsetX, boundaryY);
  c.lineTo(offsetX + gridW, boundaryY);
  c.stroke();
  c.setLineDash([]);

  // Row labels on the left
  c.fillStyle = '#444343';
  c.font = `${Math.max(8, cellSize * 0.25)}px Inter, sans-serif`;
  c.textAlign = 'right';
  c.textBaseline = 'middle';
  for (let row = southBound; row <= northBound; row += Math.max(1, Math.floor(windowHeight / 10))) {
    const { y } = cellToCanvas(0, row);
    c.fillText(`${row}`, offsetX - 4, y + cellSize / 2);
  }

  // --- Status bar ---
  const currentStepNum = obs.step ?? step;
  const scrollInfo = `Scroll: ${southBound}-${northBound}`;
  const mineCount = mines.size;

  // Inline crystal SVG used in the scoreboard in place of the word "Crystals"
  const crystalSvg = `
    <svg viewBox="-12 -12 24 24" width="14" height="14" style="vertical-align: -2px;" aria-label="crystals">
      <g stroke="rgba(218,165,32,0.9)" stroke-width="1.25" stroke-linecap="round">
        <line x1="0" y1="-9" x2="0" y2="-12"/>
        <line x1="0" y1="9" x2="0" y2="12"/>
        <line x1="-9" y1="0" x2="-12" y2="0"/>
        <line x1="9" y1="0" x2="12" y2="0"/>
        <line x1="-6.4" y1="-6.4" x2="-8.5" y2="-8.5"/>
        <line x1="6.4" y1="-6.4" x2="8.5" y2="-8.5"/>
        <line x1="-6.4" y1="6.4" x2="-8.5" y2="8.5"/>
        <line x1="6.4" y1="6.4" x2="8.5" y2="8.5"/>
      </g>
      <polygon points="0,-8 7,0 0,5 -7,0" fill="rgba(255,215,0,0.7)" stroke="#DAA520" stroke-width="1.25"/>
    </svg>`;

  if (isGameOver) {
    const r0 = currentStep[0].reward;
    const r1 = currentStep[1].reward;
    let result = 'Draw';
    if (r0 !== null && r1 !== null) {
      if (r0 > r1) result = `${getName(0)} wins!`;
      else if (r1 > r0) result = `${getName(1)} wins!`;
    }
    statusContainer.innerHTML = `
      <span style="font-weight: 700; font-size: 1rem;">${result}</span>
      <span class="status-item">Step <strong>${currentStepNum}</strong></span>
      <span class="status-item">${scrollInfo}</span>
    `;
  } else {
    const visibleNodes = Object.keys(globalMiningNodes).filter((k) => {
      const r = parseInt(k.split(',')[1]);
      return r >= southBound && r <= northBound;
    }).length;
    statusContainer.innerHTML = `
      <span class="status-item">Step <strong>${currentStepNum}</strong></span>
      <span class="status-item">${scrollInfo}</span>
      <span class="status-item">Mines: <strong>${mineCount}</strong></span>
      <span class="status-item">Nodes: <strong>${visibleNodes}</strong></span>
      <span class="status-item">${crystalSvg} <strong>${Object.keys(globalCrystals).length}</strong></span>
    `;
  }
}
