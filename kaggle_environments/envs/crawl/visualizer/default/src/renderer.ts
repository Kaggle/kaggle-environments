import type { RendererOptions } from '@kaggle-environments/core';

// Wall bitfield constants
const WALL_N = 1;
const WALL_E = 2;
const WALL_S = 4;
const WALL_W = 8;
const PETRIFIED = 16;

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
  const r = size * 0.18;

  // Glowing crystal shape
  c.fillStyle = 'rgba(255, 215, 0, 0.6)';
  c.strokeStyle = '#DAA520';
  c.lineWidth = 1;
  c.beginPath();
  c.moveTo(cx, cy - r * 1.2);
  c.lineTo(cx + r, cy);
  c.lineTo(cx, cy + r * 0.8);
  c.lineTo(cx - r, cy);
  c.closePath();
  c.fill();
  c.stroke();

  // Energy value
  if (size > 16) {
    c.fillStyle = '#8B6914';
    c.font = `${Math.max(7, size * 0.22)}px Inter, sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(`${energy}`, cx, cy + r * 1.5);
  }
}

export function renderer(options: CrawlOptions) {
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

  // Parse global state from replay
  const globalWalls: Record<string, number[]> = obs.globalWalls ?? {};
  const globalCrystals: Record<string, number> = obs.globalCrystals ?? {};
  const robots = parseRobots(obs.globalRobots);
  const mines = parseMines(obs.globalMines);

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
      const w = rowWalls ? rowWalls[col] : 0;

      if (w === PETRIFIED) {
        // Petrified cell: filled dark
        c.fillStyle = '#5D4037';
        c.fillRect(x, y, cellSize, cellSize);
        // Draw cross-hatch
        c.strokeStyle = '#3E2723';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(x, y);
        c.lineTo(x + cellSize, y + cellSize);
        c.moveTo(x + cellSize, y);
        c.lineTo(x, y + cellSize);
        c.stroke();
        continue;
      }

      // Cell background - subtle grid
      c.fillStyle = 'rgba(255, 255, 255, 0.15)';
      c.fillRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);

      // Draw walls
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
        c.beginPath();
        c.moveTo(x + cellSize, y);
        c.lineTo(x + cellSize, y + cellSize);
        c.stroke();
      }
      if (w & WALL_W) {
        c.beginPath();
        c.moveTo(x, y);
        c.lineTo(x, y + cellSize);
        c.stroke();
      }
    }
  }

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
  for (const [key, energy] of Object.entries(globalCrystals)) {
    const [colStr, rowStr] = key.split(',');
    const col = parseInt(colStr);
    const row = parseInt(rowStr);
    if (row < southBound || row > northBound) continue;
    const { x, y } = cellToCanvas(col, row);
    drawCrystal(c, x, y, cellSize, energy);
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
    statusContainer.innerHTML = `
      <span class="status-item">Step <strong>${currentStepNum}</strong></span>
      <span class="status-item">${scrollInfo}</span>
      <span class="status-item">Mines: <strong>${mineCount}</strong></span>
      <span class="status-item">Crystals: <strong>${Object.keys(globalCrystals).length}</strong></span>
    `;
  }
}
