import type { RendererOptions } from '@kaggle-environments/core';

// Battleship observation string format (per player):
//
// State of player's ships:
// +-----+
// | b   |
// |ab   |    lowercase = unhit ship, UPPERCASE = hit ship, * = opponent miss, space = water
// |ab   |
// +-----+
//
// Player's shot outcomes:
// +-----+
// |@    |    # = hit on opponent, @ = miss, space = unknown
// |    #|
// +-----+

interface BoardGrid {
  rows: string[][]; // grid of single characters
  width: number;
  height: number;
}

interface PlayerObservation {
  ownBoard: BoardGrid; // player's ships + incoming shots
  shotsBoard: BoardGrid; // player's attacks on opponent
}

// Cell type on the player's own board
type OwnCellType = 'water' | 'ship' | 'ship_hit' | 'miss';
// Cell type on the shots board
type ShotCellType = 'unknown' | 'hit' | 'miss';

const COLORS = {
  water: '#b8cfe0',
  waterStroke: '#8dafc0',
  ship: '#6b8e9f',
  shipStroke: '#4a6b7a',
  shipHit: '#c0392b',
  shipHitStroke: '#962d22',
  missMarker: '#8a8aa0',
  hitMarker: '#c0392b',
  shotMiss: '#a0aab0',
  shotHit: '#c0392b',
  unknown: '#d4cfc4',
  unknownStroke: '#b8b3a6',
  text: '#050001',
  textSecondary: '#444343',
  lastShot: '#c0392b',
  boardBorder: '#3c3b37',
  boardLabel: '#444343',
};

function parseBoard(section: string): BoardGrid | null {
  const lines = section.split('\n').filter((l) => l.includes('|') || l.includes('+'));
  const dataLines = lines.filter((l) => l.includes('|') && !l.includes('+'));
  if (dataLines.length === 0) return null;

  const rows: string[][] = [];
  for (const line of dataLines) {
    const match = line.match(/\|(.+)\|/);
    if (match) {
      rows.push(match[1].split(''));
    }
  }

  if (rows.length === 0) return null;
  return { rows, width: rows[0].length, height: rows.length };
}

function parseObservation(obsString: string): PlayerObservation | null {
  if (!obsString) return null;

  const ownIdx = obsString.indexOf("State of player's ships:");
  const shotsIdx = obsString.indexOf("Player's shot outcomes:");
  if (ownIdx === -1 || shotsIdx === -1) return null;

  const ownSection = obsString.substring(ownIdx, shotsIdx);
  const shotsSection = obsString.substring(shotsIdx);

  const ownBoard = parseBoard(ownSection);
  const shotsBoard = parseBoard(shotsSection);
  if (!ownBoard || !shotsBoard) return null;

  return { ownBoard, shotsBoard };
}

function getOwnCellType(ch: string): OwnCellType {
  if (ch === ' ') return 'water';
  if (ch === '*') return 'miss';
  if (ch >= 'a' && ch <= 'z') return 'ship';
  if (ch >= 'A' && ch <= 'Z') return 'ship_hit';
  return 'water';
}

function getShotCellType(ch: string): ShotCellType {
  if (ch === '#') return 'hit';
  if (ch === '@') return 'miss';
  return 'unknown';
}

function getObservationString(step: any, playerIdx: number): string {
  if (!step || !Array.isArray(step)) return '';
  return step[playerIdx]?.observation?.observationString ?? '';
}

function isTerminal(step: any): boolean {
  if (!step || !Array.isArray(step)) return false;
  return step.some((p: any) => p?.status === 'DONE' || p?.observation?.isTerminal);
}

function getCurrentPlayer(step: any): number {
  if (!step || !Array.isArray(step)) return 0;
  for (const player of step) {
    const cp = player?.observation?.currentPlayer;
    if (cp !== undefined && cp >= 0) return cp;
  }
  return 0;
}

function getRewards(step: any): [number, number] {
  if (!step || !Array.isArray(step)) return [0, 0];
  return [step[0]?.reward ?? 0, step[1]?.reward ?? 0];
}

function detectPhase(obs: PlayerObservation): 'placement' | 'war' {
  for (const row of obs.shotsBoard.rows) {
    for (const ch of row) {
      if (ch === '@' || ch === '#') return 'war';
    }
  }
  return 'placement';
}

function countBoardStats(board: BoardGrid): {
  shipCells: number;
  hitCells: number;
  missCells: number;
  shipLetters: Set<string>;
} {
  let shipCells = 0,
    hitCells = 0,
    missCells = 0;
  const shipLetters = new Set<string>();
  for (const row of board.rows) {
    for (const ch of row) {
      const type = getOwnCellType(ch);
      if (type === 'ship') {
        shipCells++;
        shipLetters.add(ch);
      }
      if (type === 'ship_hit') {
        hitCells++;
        shipLetters.add(ch.toLowerCase());
      }
      if (type === 'miss') missCells++;
    }
  }
  return { shipCells, hitCells, missCells, shipLetters };
}

function countShotStats(board: BoardGrid): { hits: number; misses: number; total: number } {
  let hits = 0,
    misses = 0;
  for (const row of board.rows) {
    for (const ch of row) {
      if (ch === '#') hits++;
      if (ch === '@') misses++;
    }
  }
  return { hits, misses, total: hits + misses };
}

function findNewShot(prev: BoardGrid | null, curr: BoardGrid): { row: number; col: number } | null {
  if (!prev) return null;
  for (let r = 0; r < curr.height; r++) {
    for (let c = 0; c < curr.width; c++) {
      const prevCh = prev.rows[r]?.[c] ?? ' ';
      const currCh = curr.rows[r]?.[c] ?? ' ';
      if (prevCh !== currCh) {
        return { row: r, col: c };
      }
    }
  }
  return null;
}

// ---- Drawing helpers ----

function drawBoard(
  c: CanvasRenderingContext2D,
  board: BoardGrid,
  x: number,
  y: number,
  cellSize: number,
  mode: 'own' | 'shots',
  lastShot: { row: number; col: number } | null
) {
  const { width: bw, height: bh } = board;
  const boardPxW = cellSize * bw;
  const boardPxH = cellSize * bh;

  // Board background
  c.fillStyle = mode === 'own' ? COLORS.water : COLORS.unknown;
  c.beginPath();
  c.roundRect(x, y, boardPxW, boardPxH, 4);
  c.fill();

  // Grid lines
  c.strokeStyle = mode === 'own' ? COLORS.waterStroke : COLORS.unknownStroke;
  c.lineWidth = 0.5;
  for (let r = 0; r <= bh; r++) {
    c.beginPath();
    c.moveTo(x, y + r * cellSize);
    c.lineTo(x + boardPxW, y + r * cellSize);
    c.stroke();
  }
  for (let col = 0; col <= bw; col++) {
    c.beginPath();
    c.moveTo(x + col * cellSize, y);
    c.lineTo(x + col * cellSize, y + boardPxH);
    c.stroke();
  }

  // Draw cells
  for (let r = 0; r < bh; r++) {
    for (let col = 0; col < bw; col++) {
      const ch = board.rows[r][col];
      const cx = x + col * cellSize + cellSize / 2;
      const cy = y + r * cellSize + cellSize / 2;
      const isLast = lastShot && lastShot.row === r && lastShot.col === col;

      if (mode === 'own') {
        drawOwnCell(c, ch, x + col * cellSize, y + r * cellSize, cellSize, cx, cy, isLast);
      } else {
        drawShotCell(c, ch, x + col * cellSize, y + r * cellSize, cellSize, cx, cy, isLast);
      }
    }
  }

  // Board border (dashed)
  c.strokeStyle = COLORS.boardBorder;
  c.lineWidth = 1.5;
  c.setLineDash([4, 3]);
  c.beginPath();
  c.roundRect(x, y, boardPxW, boardPxH, 4);
  c.stroke();
  c.setLineDash([]);

  // Column labels (A, B, C, ...)
  c.fillStyle = COLORS.boardLabel;
  c.font = `${Math.max(9, cellSize * 0.28)}px 'Inter', sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'top';
  for (let col = 0; col < bw; col++) {
    c.fillText(String.fromCharCode(65 + col), x + col * cellSize + cellSize / 2, y + boardPxH + 3);
  }

  // Row labels (1, 2, 3, ...)
  c.textAlign = 'right';
  c.textBaseline = 'middle';
  for (let r = 0; r < bh; r++) {
    c.fillText(String(r + 1), x - 4, y + r * cellSize + cellSize / 2);
  }
}

function drawOwnCell(
  c: CanvasRenderingContext2D,
  ch: string,
  cellX: number,
  cellY: number,
  cellSize: number,
  cx: number,
  cy: number,
  isLast: boolean | null
) {
  const type = getOwnCellType(ch);
  const r = cellSize * 0.35;

  if (type === 'ship') {
    const pad = cellSize * 0.08;
    c.fillStyle = COLORS.ship;
    c.fillRect(cellX + pad, cellY + pad, cellSize - pad * 2, cellSize - pad * 2);
    c.strokeStyle = COLORS.shipStroke;
    c.lineWidth = 1;
    c.strokeRect(cellX + pad, cellY + pad, cellSize - pad * 2, cellSize - pad * 2);

    c.fillStyle = '#2d4a5a';
    c.font = `bold ${Math.max(9, cellSize * 0.3)}px 'Inter', sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(ch, cx, cy);
  } else if (type === 'ship_hit') {
    const pad = cellSize * 0.08;
    c.fillStyle = COLORS.shipHit;
    c.fillRect(cellX + pad, cellY + pad, cellSize - pad * 2, cellSize - pad * 2);
    c.strokeStyle = COLORS.shipHitStroke;
    c.lineWidth = 1;
    c.strokeRect(cellX + pad, cellY + pad, cellSize - pad * 2, cellSize - pad * 2);

    c.fillStyle = '#fff';
    c.font = `bold ${Math.max(9, cellSize * 0.3)}px 'Inter', sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(ch, cx, cy);

    if (isLast) {
      drawLastShotHighlight(c, cellX, cellY, cellSize);
    }
  } else if (type === 'miss') {
    c.fillStyle = COLORS.missMarker;
    c.globalAlpha = 0.5;
    c.beginPath();
    c.arc(cx, cy, r * 0.4, 0, Math.PI * 2);
    c.fill();
    c.globalAlpha = 1;

    if (isLast) {
      drawLastShotHighlight(c, cellX, cellY, cellSize);
    }
  }
}

function drawShotCell(
  c: CanvasRenderingContext2D,
  ch: string,
  cellX: number,
  cellY: number,
  cellSize: number,
  cx: number,
  cy: number,
  isLast: boolean | null
) {
  const type = getShotCellType(ch);
  const r = cellSize * 0.3;

  if (type === 'hit') {
    c.fillStyle = 'rgba(192, 57, 43, 0.15)';
    c.fillRect(cellX + 1, cellY + 1, cellSize - 2, cellSize - 2);

    const xSize = r * 0.6;
    c.strokeStyle = COLORS.shotHit;
    c.lineWidth = Math.max(2, cellSize * 0.06);
    c.beginPath();
    c.moveTo(cx - xSize, cy - xSize);
    c.lineTo(cx + xSize, cy + xSize);
    c.moveTo(cx + xSize, cy - xSize);
    c.lineTo(cx - xSize, cy + xSize);
    c.stroke();

    if (isLast) {
      drawLastShotHighlight(c, cellX, cellY, cellSize);
    }
  } else if (type === 'miss') {
    c.fillStyle = COLORS.shotMiss;
    c.beginPath();
    c.arc(cx, cy, r * 0.35, 0, Math.PI * 2);
    c.fill();

    if (isLast) {
      drawLastShotHighlight(c, cellX, cellY, cellSize);
    }
  }
}

function drawLastShotHighlight(c: CanvasRenderingContext2D, cellX: number, cellY: number, cellSize: number) {
  c.save();
  c.strokeStyle = COLORS.lastShot;
  c.lineWidth = 2;
  c.setLineDash([4, 3]);
  c.strokeRect(cellX + 2, cellY + 2, cellSize - 4, cellSize - 4);
  c.setLineDash([]);
  c.restore();
}

// ---- Main renderer ----

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header">
        <span class="player-card sketched-border" id="p1-card">Player 1</span>
        <span class="vs-label">vs</span>
        <span class="player-card sketched-border" id="p2-card">Player 2</span>
      </div>
      <div class="info-row">
        <span class="stats-info"></span>
        <span class="move-info"></span>
      </div>
      <canvas></canvas>
      <div class="status-container sketched-border"></div>
    </div>
  `;

  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const p1Card = parent.querySelector('#p1-card') as HTMLSpanElement;
  const p2Card = parent.querySelector('#p2-card') as HTMLSpanElement;
  const statsInfo = parent.querySelector('.stats-info') as HTMLSpanElement;
  const moveInfoEl = parent.querySelector('.move-info') as HTMLSpanElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;
  if (!canvas || !replay) return;

  canvas.width = 0;
  canvas.height = 0;
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = width;
  canvas.height = height;

  const c = canvas.getContext('2d');
  if (!c) return;

  const currentStep = steps[step];

  // Parse both players' observations
  const p0Obs = parseObservation(getObservationString(currentStep, 0));
  const p1Obs = parseObservation(getObservationString(currentStep, 1));

  // Transparent canvas
  c.clearRect(0, 0, width, height);

  if (!p0Obs || !p1Obs) {
    statusContainer.textContent = 'Waiting for game data...';
    return;
  }

  const terminal = isTerminal(currentStep);
  const cp = getCurrentPlayer(currentStep);
  const phase = detectPhase(p0Obs);

  // Compute stats
  const p0OwnStats = countBoardStats(p0Obs.ownBoard);
  const p1OwnStats = countBoardStats(p1Obs.ownBoard);
  const p0ShotStats = countShotStats(p0Obs.shotsBoard);
  const p1ShotStats = countShotStats(p1Obs.shotsBoard);

  // Find last shots
  let lastShotOnP0Board: { row: number; col: number } | null = null;
  let lastShotOnP1Board: { row: number; col: number } | null = null;
  let lastShotByP0: { row: number; col: number } | null = null;
  let lastShotByP1: { row: number; col: number } | null = null;
  if (step > 0 && phase === 'war') {
    const prevP0Obs = parseObservation(getObservationString(steps[step - 1], 0));
    const prevP1Obs = parseObservation(getObservationString(steps[step - 1], 1));
    if (prevP0Obs && prevP1Obs) {
      lastShotOnP0Board = findNewShot(prevP0Obs.ownBoard, p0Obs.ownBoard);
      lastShotOnP1Board = findNewShot(prevP1Obs.ownBoard, p1Obs.ownBoard);
      lastShotByP0 = findNewShot(prevP0Obs.shotsBoard, p0Obs.shotsBoard);
      lastShotByP1 = findNewShot(prevP1Obs.shotsBoard, p1Obs.shotsBoard);
    }
  }

  // =========================================================================
  //  DOM HEADER
  // =========================================================================
  p1Card.textContent = `P1: ${p0ShotStats.hits} hits`;
  p2Card.textContent = `P2: ${p1ShotStats.hits} hits`;

  if (!terminal && cp === 0) {
    p1Card.classList.add('active');
  } else {
    p1Card.classList.remove('active');
  }
  if (!terminal && cp === 1) {
    p2Card.classList.add('active');
  } else {
    p2Card.classList.remove('active');
  }

  // =========================================================================
  //  DOM INFO ROW
  // =========================================================================
  statsInfo.textContent = `P1: ${p0ShotStats.misses} misses | P2: ${p1ShotStats.misses} misses`;

  if (phase === 'war') {
    const lastShot = lastShotByP0 || lastShotByP1;
    const shooter = lastShotByP0 ? 0 : lastShotByP1 ? 1 : -1;
    if (lastShot && shooter >= 0) {
      const shotBoard = shooter === 0 ? p0Obs.shotsBoard : p1Obs.shotsBoard;
      const result = getShotCellType(shotBoard.rows[lastShot.row][lastShot.col]);
      const resultText = result === 'hit' ? 'HIT!' : 'miss';
      const coordText = `${String.fromCharCode(65 + lastShot.col)}${lastShot.row + 1}`;
      moveInfoEl.textContent = `P${shooter + 1} fires ${coordText}: ${resultText}`;
    } else {
      moveInfoEl.textContent = '';
    }
  } else {
    moveInfoEl.textContent = phase === 'placement' ? 'Ship Placement' : '';
  }

  // =========================================================================
  //  BOARD RENDERING (canvas) -- 2x2 grid
  // =========================================================================
  const boardLabelH = 18;
  const labelGap = 4;
  const margin = 12;
  const colGap = 16;

  const boardW = p0Obs.ownBoard.width;
  const boardH = p0Obs.ownBoard.height;

  const availW = width - margin * 2 - colGap;
  const availH = height - margin - (boardLabelH + labelGap) * 2 - colGap;
  const maxCellFromW = availW / (boardW * 2);
  const maxCellFromH = availH / (boardH * 2);
  const cellSize = Math.min(maxCellFromW, maxCellFromH, 48);

  const boardPxW = cellSize * boardW;
  const boardPxH = cellSize * boardH;
  const rowLabelW = 16;
  const totalGridW = (boardPxW + rowLabelW) * 2 + colGap;
  const gridStartX = (width - totalGridW) / 2 + rowLabelW;
  const gridStartY = margin;

  const col1X = gridStartX;
  const col2X = gridStartX + boardPxW + rowLabelW + colGap;
  const row1Y = gridStartY + boardLabelH + labelGap;
  const row2Y = row1Y + boardPxH + boardLabelH + labelGap + colGap + 14;

  // Board labels
  const drawBoardLabel = (text: string, x: number, y: number) => {
    c.fillStyle = COLORS.text;
    c.font = `bold ${Math.max(11, cellSize * 0.32)}px 'Inter', sans-serif`;
    c.textAlign = 'left';
    c.textBaseline = 'bottom';
    c.fillText(text, x, y);
  };

  // Row 1: Ship boards
  drawBoardLabel('Player 1 \u2014 Ships', col1X, row1Y - labelGap);
  drawBoardLabel('Player 2 \u2014 Ships', col2X, row1Y - labelGap);
  drawBoard(c, p0Obs.ownBoard, col1X, row1Y, cellSize, 'own', lastShotOnP0Board);
  drawBoard(c, p1Obs.ownBoard, col2X, row1Y, cellSize, 'own', lastShotOnP1Board);

  // Row 2: Shots boards
  drawBoardLabel('Player 1 \u2014 Shots', col1X, row2Y - labelGap);
  drawBoardLabel('Player 2 \u2014 Shots', col2X, row2Y - labelGap);
  drawBoard(c, p0Obs.shotsBoard, col1X, row2Y, cellSize, 'shots', lastShotByP0);
  drawBoard(c, p1Obs.shotsBoard, col2X, row2Y, cellSize, 'shots', lastShotByP1);

  // Damage indicators
  const drawDamageInfo = (stats: ReturnType<typeof countBoardStats>, x: number, y: number) => {
    if (stats.hitCells === 0) return;
    c.fillStyle = COLORS.shipHit;
    c.font = `bold ${Math.max(10, cellSize * 0.26)}px 'Inter', sans-serif`;
    c.textAlign = 'left';
    c.textBaseline = 'top';
    c.fillText(`${stats.hitCells} hit`, x, y);
  };

  drawDamageInfo(p0OwnStats, col1X + boardPxW + 6, row1Y + 2);
  drawDamageInfo(p1OwnStats, col2X + boardPxW + 6, row1Y + 2);

  // =========================================================================
  //  DOM STATUS CONTAINER
  // =========================================================================
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over \u2014 Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over \u2014 Player 1 wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over \u2014 Player 2 wins!';
    statusContainer.textContent = `${msg} | P1: ${p0ShotStats.hits} hits, P2: ${p1ShotStats.hits} hits`;
    statusContainer.style.fontWeight = '700';
  } else {
    const phaseLabel = phase === 'placement' ? 'Ship Placement' : 'War';
    statusContainer.textContent = `${phaseLabel} \u2014 Player ${cp + 1}'s turn`;
  }
}
