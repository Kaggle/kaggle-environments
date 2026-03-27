import type { RendererOptions } from '@kaggle-environments/core';

// --- Observation from proxy (JSON) ---

interface BattleshipObs {
  ships: string[][]; // rows × cols: ' '=empty, a-z=ship intact, A-Z=ship hit, '*'=opponent miss
  shots: string[][]; // rows × cols: ' '=not shot, '@'=miss, '#'=hit
  width: number;
  height: number;
  phase: 'placement' | 'war';
  current_player: number;
  is_terminal: boolean;
  winner: number | 'draw' | null;
}

function getObservation(step: any, playerIdx: number): BattleshipObs | null {
  const raw = step?.[playerIdx]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function getRewards(step: any): [number, number] {
  if (!step || !Array.isArray(step)) return [0, 0];
  return [step[0]?.reward ?? 0, step[1]?.reward ?? 0];
}

// --- Stats ---

interface PlayerStats {
  shipsIntact: number;
  shipsHit: number;
  shotsHit: number;
  shotsMissed: number;
}

function getStats(obs: BattleshipObs): PlayerStats {
  let shipsIntact = 0;
  let shipsHit = 0;
  for (const row of obs.ships) {
    for (const cell of row) {
      if (cell >= 'a' && cell <= 'z') shipsIntact++;
      if (cell >= 'A' && cell <= 'Z') shipsHit++;
    }
  }
  let shotsHit = 0;
  let shotsMissed = 0;
  for (const row of obs.shots) {
    for (const cell of row) {
      if (cell === '#') shotsHit++;
      if (cell === '@') shotsMissed++;
    }
  }
  return { shipsIntact, shipsHit, shotsHit, shotsMissed };
}

// --- Colors ---

const COLORS = {
  water: '#dbeafe', // light blue
  ship: '#78716c', // warm gray
  shipHit: '#ef4444', // red
  miss: '#94a3b8', // slate
  shotHit: '#ef4444', // red
  shotMiss: '#94a3b8', // slate
  grid: '#3c3b37',
  highlight: '#facc15', // gold for latest move
  labelText: '#050001',
};

// --- Renderer ---

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

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

  canvas.width = 0;
  canvas.height = 0;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;

  const c = canvas.getContext('2d');
  if (!c) return;

  const currentStep = steps[step];
  const p0 = getObservation(currentStep, 0);
  const p1 = getObservation(currentStep, 1);

  if (!p0 || !p1) {
    statusContainer.textContent = 'Waiting for game data...';
    return;
  }

  const terminal = p0.is_terminal;
  const cp = p0.current_player;
  const placement = p0.phase === 'placement' && p1.phase === 'placement';
  const p0Stats = getStats(p0);
  const p1Stats = getStats(p1);

  // Diff with previous step for move highlights
  let prevP0: BattleshipObs | null = null;
  let prevP1: BattleshipObs | null = null;
  if (step > 0) {
    prevP0 = getObservation(steps[step - 1], 0);
    prevP1 = getObservation(steps[step - 1], 1);
  }

  // --- Header ---
  const p0Active = !terminal && cp === 0;
  const p1Active = !terminal && cp === 1;
  const p0HitInfo = !placement ? ` (${p0Stats.shotsHit}H ${p0Stats.shotsMissed}M)` : '';
  const p1HitInfo = !placement ? ` (${p1Stats.shotsHit}H ${p1Stats.shotsMissed}M)` : '';
  header.innerHTML = `
    <span class="sketched-border" style="padding: 4px 14px; background-color: ${p0Active ? '#bdeeff' : 'white'}; font-weight: 700; transition: background-color 300ms;">
      Player 1<span style="color: #444343; font-size: 0.8rem; margin-left: 4px;">${p0HitInfo}</span>
    </span>
    <span style="color: #444343;">${placement ? 'DEPLOY' : 'WAR'}</span>
    <span class="sketched-border" style="padding: 4px 14px; background-color: ${p1Active ? '#bdeeff' : 'white'}; font-weight: 700; transition: background-color 300ms;">
      Player 2<span style="color: #444343; font-size: 0.8rem; margin-left: 4px;">${p1HitInfo}</span>
    </span>
  `;

  // --- Canvas layout ---
  // Layout: two columns (P1 | P2), each with ships board on top and shots board below
  const w = canvas.width;
  const h = canvas.height;
  c.clearRect(0, 0, w, h);

  const boardW = p0.width;
  const boardH = p0.height;
  const gap = 16;
  const labelMargin = 16;
  const titleHeight = 18;

  // Calculate cell size to fit 2 columns × 2 rows of boards
  const availW = (w - gap - labelMargin * 2) / 2;
  const availH = (h - gap - titleHeight * 2 - 8) / 2;
  const cellSize = Math.min(Math.floor(availW / boardW), Math.floor(availH / boardH), 40);

  const gridW = cellSize * boardW;
  const gridH = cellSize * boardH;

  // Column x-offsets (centered)
  const totalW = gridW * 2 + gap + labelMargin * 2;
  const startX = Math.max(0, (w - totalW) / 2) + labelMargin;
  const col1X = startX;
  const col2X = startX + gridW + gap + labelMargin;

  // Row y-offsets
  const startY = 4;
  const row1Y = startY + titleHeight;
  const row2Y = row1Y + gridH + titleHeight + 8;

  const fontSize = Math.max(9, Math.min(12, cellSize * 0.35));

  // --- Draw a single grid ---
  const drawGrid = (
    ox: number,
    oy: number,
    board: string[][],
    mode: 'ships' | 'shots',
    prevBoard: string[][] | null,
    title: string
  ) => {
    // Title
    c.fillStyle = COLORS.labelText;
    c.font = `600 ${fontSize}px Inter, sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'bottom';
    c.fillText(title, ox + gridW / 2, oy - 3);

    for (let r = 0; r < boardH; r++) {
      for (let col = 0; col < boardW; col++) {
        const x = ox + col * cellSize;
        const y = oy + r * cellSize;
        const cell = board[r]?.[col] ?? ' ';
        const prevCell = prevBoard?.[r]?.[col] ?? ' ';
        const changed = prevBoard !== null && cell !== prevCell;

        // Background
        c.fillStyle = COLORS.water;
        c.fillRect(x, y, cellSize, cellSize);

        if (mode === 'ships') {
          if (cell >= 'a' && cell <= 'z') {
            // Ship intact
            c.fillStyle = COLORS.ship;
            c.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
          } else if (cell >= 'A' && cell <= 'Z') {
            // Ship hit
            c.fillStyle = COLORS.shipHit;
            c.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
            // X mark
            c.strokeStyle = '#fff';
            c.lineWidth = 2;
            c.beginPath();
            c.moveTo(x + 4, y + 4);
            c.lineTo(x + cellSize - 4, y + cellSize - 4);
            c.moveTo(x + cellSize - 4, y + 4);
            c.lineTo(x + 4, y + cellSize - 4);
            c.stroke();
          } else if (cell === '*') {
            // Opponent miss
            c.fillStyle = COLORS.miss;
            c.beginPath();
            c.arc(x + cellSize / 2, y + cellSize / 2, cellSize * 0.15, 0, Math.PI * 2);
            c.fill();
          }
        } else {
          // Shots board
          if (cell === '#') {
            // Hit
            c.fillStyle = COLORS.shotHit;
            c.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
            c.strokeStyle = '#fff';
            c.lineWidth = 2;
            c.beginPath();
            c.moveTo(x + 4, y + 4);
            c.lineTo(x + cellSize - 4, y + cellSize - 4);
            c.moveTo(x + cellSize - 4, y + 4);
            c.lineTo(x + 4, y + cellSize - 4);
            c.stroke();
          } else if (cell === '@') {
            // Miss
            c.fillStyle = COLORS.miss;
            c.beginPath();
            c.arc(x + cellSize / 2, y + cellSize / 2, cellSize * 0.15, 0, Math.PI * 2);
            c.fill();
          }
        }

        // Highlight changed cells
        if (changed) {
          c.strokeStyle = COLORS.highlight;
          c.lineWidth = 2.5;
          c.strokeRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
        }

        // Grid line
        c.strokeStyle = COLORS.grid;
        c.lineWidth = 0.5;
        c.strokeRect(x, y, cellSize, cellSize);
      }
    }

    // Column labels
    c.fillStyle = '#444343';
    c.font = `${Math.max(8, fontSize - 1)}px Inter, sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'top';
    for (let col = 0; col < boardW; col++) {
      c.fillText(String.fromCharCode(65 + col), ox + col * cellSize + cellSize / 2, oy + gridH + 2);
    }

    // Row labels
    c.textAlign = 'right';
    c.textBaseline = 'middle';
    for (let r = 0; r < boardH; r++) {
      c.fillText(String(r + 1), ox - 3, oy + r * cellSize + cellSize / 2);
    }
  };

  // Draw 4 boards: P1 ships, P2 ships (top row), P1 shots, P2 shots (bottom row)
  drawGrid(col1X, row1Y, p0.ships, 'ships', prevP0?.ships ?? null, 'P1 Ships');
  drawGrid(col2X, row1Y, p1.ships, 'ships', prevP1?.ships ?? null, 'P2 Ships');

  if (!placement) {
    drawGrid(col1X, row2Y, p0.shots, 'shots', prevP0?.shots ?? null, 'P1 Shots');
    drawGrid(col2X, row2Y, p1.shots, 'shots', prevP1?.shots ?? null, 'P2 Shots');
  } else {
    // During placement, show "Deploying..." labels where shot boards would be
    c.fillStyle = '#444343';
    c.font = `italic ${fontSize}px Mynerve, cursive`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText('Deploying ships...', col1X + gridW / 2, row2Y + gridH / 2);
    c.fillText('Deploying ships...', col2X + gridW / 2, row2Y + gridH / 2);
  }

  // --- Status ---
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over — Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over — Player 1 wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over — Player 2 wins!';
    statusContainer.textContent = msg;
    statusContainer.style.fontWeight = '700';
  } else if (placement) {
    statusContainer.textContent = `Player ${cp + 1} is placing ships`;
  } else {
    statusContainer.textContent = `Player ${cp + 1}'s turn to fire`;
  }
}
