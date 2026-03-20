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

const PLAYER_COLORS = ['#4fc3f7', '#ff8a65'] as const;

const COLORS = {
  bg: '#1a1a2e',
  panelBg: '#16213e',
  water: '#1b3a5c',
  waterStroke: '#234b73',
  ship: '#6b8e9f',
  shipStroke: '#4a6b7a',
  shipHit: '#ef4444',
  shipHitStroke: '#b91c1c',
  missMarker: '#94a3b8',
  hitMarker: '#ef4444',
  shotMiss: '#475569',
  shotHit: '#ef4444',
  unknown: '#0f2942',
  unknownStroke: '#1a3a5a',
  text: '#e0e0e0',
  textDim: '#8a8aa0',
  labelBg: 'rgba(22, 33, 62, 0.9)',
  lastShot: '#ffd700',
  lastShotGlow: 'rgba(255, 215, 0, 0.4)',
  boardBorder: '#2d4a6a',
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

  // Split into own board and shots board sections
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

/** Detect the phase from the observation: placement (ships still being placed) or war (shooting). */
function detectPhase(obs: PlayerObservation): 'placement' | 'war' {
  // During war phase, the shots board has at least one @ or #
  for (const row of obs.shotsBoard.rows) {
    for (const ch of row) {
      if (ch === '@' || ch === '#') return 'war';
    }
  }
  return 'placement';
}

/** Count ships and hits for a player's own board. */
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

/** Count shots on the shots board. */
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

/** Find the cell that changed between two boards (the last shot). */
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
  lastShot: { row: number; col: number } | null,
  _playerIdx: number
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

  // Board border
  c.strokeStyle = COLORS.boardBorder;
  c.lineWidth = 2;
  c.beginPath();
  c.roundRect(x, y, boardPxW, boardPxH, 4);
  c.stroke();

  // Column labels (A, B, C, ...)
  c.fillStyle = COLORS.textDim;
  c.font = `${Math.max(9, cellSize * 0.28)}px sans-serif`;
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
    // Ship cell - filled rectangle
    const pad = cellSize * 0.08;
    c.fillStyle = COLORS.ship;
    c.fillRect(cellX + pad, cellY + pad, cellSize - pad * 2, cellSize - pad * 2);
    c.strokeStyle = COLORS.shipStroke;
    c.lineWidth = 1;
    c.strokeRect(cellX + pad, cellY + pad, cellSize - pad * 2, cellSize - pad * 2);

    // Ship letter
    c.fillStyle = '#c8dce6';
    c.font = `bold ${Math.max(9, cellSize * 0.3)}px sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(ch, cx, cy);
  } else if (type === 'ship_hit') {
    // Hit ship cell - red
    const pad = cellSize * 0.08;
    c.fillStyle = COLORS.shipHit;
    c.fillRect(cellX + pad, cellY + pad, cellSize - pad * 2, cellSize - pad * 2);
    c.strokeStyle = COLORS.shipHitStroke;
    c.lineWidth = 1;
    c.strokeRect(cellX + pad, cellY + pad, cellSize - pad * 2, cellSize - pad * 2);

    // Ship letter (uppercase)
    c.fillStyle = '#fff';
    c.font = `bold ${Math.max(9, cellSize * 0.3)}px sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(ch, cx, cy);

    // Last-shot glow
    if (isLast) {
      drawLastShotHighlight(c, cellX, cellY, cellSize);
    }
  } else if (type === 'miss') {
    // Opponent miss - small circle
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
  // 'water' = empty, nothing to draw
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
    // Hit on opponent - red X
    c.fillStyle = 'rgba(239, 68, 68, 0.2)';
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
    // Miss - small dot
    c.fillStyle = COLORS.shotMiss;
    c.beginPath();
    c.arc(cx, cy, r * 0.35, 0, Math.PI * 2);
    c.fill();

    if (isLast) {
      drawLastShotHighlight(c, cellX, cellY, cellSize);
    }
  }
  // 'unknown' = nothing to draw
}

function drawLastShotHighlight(c: CanvasRenderingContext2D, cellX: number, cellY: number, cellSize: number) {
  c.save();
  c.strokeStyle = COLORS.lastShot;
  c.lineWidth = 2.5;
  c.shadowColor = COLORS.lastShot;
  c.shadowBlur = 8;
  c.strokeRect(cellX + 2, cellY + 2, cellSize - 4, cellSize - 4);
  c.restore();
}

// ---- Main renderer ----

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  parent.innerHTML = `
    <div class="renderer-container">
      <canvas></canvas>
      <div class="status-bar"></div>
    </div>
  `;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const statusBar = parent.querySelector('.status-bar') as HTMLDivElement;
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

  // Background
  c.fillStyle = COLORS.bg;
  c.fillRect(0, 0, width, height);

  if (!p0Obs || !p1Obs) {
    c.fillStyle = '#fff';
    c.font = '16px sans-serif';
    c.textAlign = 'center';
    c.fillText('Waiting for game data...', width / 2, height / 2);
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

  // Find last shots by comparing with previous step
  let lastShotOnP0Board: { row: number; col: number } | null = null;
  let lastShotOnP1Board: { row: number; col: number } | null = null;
  let lastShotByP0: { row: number; col: number } | null = null;
  let lastShotByP1: { row: number; col: number } | null = null;
  if (step > 0 && phase === 'war') {
    const prevP0Obs = parseObservation(getObservationString(steps[step - 1], 0));
    const prevP1Obs = parseObservation(getObservationString(steps[step - 1], 1));
    if (prevP0Obs && prevP1Obs) {
      // Shots landing on P0's board (P1 shooting at P0)
      lastShotOnP0Board = findNewShot(prevP0Obs.ownBoard, p0Obs.ownBoard);
      // Shots landing on P1's board (P0 shooting at P1)
      lastShotOnP1Board = findNewShot(prevP1Obs.ownBoard, p1Obs.ownBoard);
      // P0's new shot (on the shots board)
      lastShotByP0 = findNewShot(prevP0Obs.shotsBoard, p0Obs.shotsBoard);
      // P1's new shot
      lastShotByP1 = findNewShot(prevP1Obs.shotsBoard, p1Obs.shotsBoard);
    }
  }

  // ---- Layout ----
  const infoPanelH = 52;
  const boardLabelH = 20;
  const labelGap = 4;
  const margin = 12;
  const colGap = 16;

  // We show 4 boards in a 2x2 grid:
  //   P1 Ships  |  P2 Ships
  //   P1 Shots  |  P2 Shots
  const boardW = p0Obs.ownBoard.width;
  const boardH = p0Obs.ownBoard.height;

  const availW = width - margin * 2 - colGap;
  const availH = height - infoPanelH - margin - (boardLabelH + labelGap) * 2 - colGap;
  const maxCellFromW = availW / (boardW * 2);
  const maxCellFromH = availH / (boardH * 2);
  const cellSize = Math.min(maxCellFromW, maxCellFromH, 48);

  const boardPxW = cellSize * boardW;
  const boardPxH = cellSize * boardH;
  const rowLabelW = 16; // space for row numbers
  const totalGridW = (boardPxW + rowLabelW) * 2 + colGap;
  const gridStartX = (width - totalGridW) / 2 + rowLabelW;
  const gridStartY = infoPanelH + margin;

  // ---- Info panel ----
  c.fillStyle = COLORS.panelBg;
  c.fillRect(0, 0, width, infoPanelH);
  c.strokeStyle = 'rgba(255,255,255,0.08)';
  c.lineWidth = 1;
  c.beginPath();
  c.moveTo(0, infoPanelH);
  c.lineTo(width, infoPanelH);
  c.stroke();

  // Accent bar
  const accentColor = terminal ? '#888' : (PLAYER_COLORS[cp] ?? PLAYER_COLORS[0]);
  c.fillStyle = accentColor;
  c.fillRect(0, 0, 5, infoPanelH);

  const panelFontSize = Math.max(13, Math.min(16, width * 0.028));
  const smallFont = Math.max(11, Math.min(13, width * 0.022));
  c.textBaseline = 'middle';

  // Turn / game-over text
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over -- Draw';
    let msgColor = COLORS.text;
    if (rewards[0] > rewards[1]) {
      msg = 'Game Over -- Player 1 wins!';
      msgColor = PLAYER_COLORS[0];
    } else if (rewards[1] > rewards[0]) {
      msg = 'Game Over -- Player 2 wins!';
      msgColor = PLAYER_COLORS[1];
    }
    c.fillStyle = msgColor;
    c.font = `bold ${panelFontSize}px sans-serif`;
    c.textAlign = 'left';
    c.fillText(msg, 14, infoPanelH * 0.35);
  } else {
    const phaseLabel = phase === 'placement' ? 'Ship Placement' : 'War';
    c.fillStyle = accentColor;
    c.font = `bold ${panelFontSize}px sans-serif`;
    c.textAlign = 'left';
    c.fillText(`Player ${cp + 1}'s turn -- ${phaseLabel}`, 14, infoPanelH * 0.35);
  }

  // Shot stats line
  c.font = `${smallFont}px sans-serif`;
  c.textAlign = 'left';
  c.fillStyle = PLAYER_COLORS[0];
  const p0StatsText = `P1: ${p0ShotStats.hits} hits, ${p0ShotStats.misses} misses`;
  c.fillText(p0StatsText, 14, infoPanelH * 0.72);

  const sep = '    ';
  const p0Width = c.measureText(p0StatsText + sep).width;
  c.fillStyle = PLAYER_COLORS[1];
  const p1StatsText = `P2: ${p1ShotStats.hits} hits, ${p1ShotStats.misses} misses`;
  c.fillText(p1StatsText, 14 + p0Width, infoPanelH * 0.72);

  // Last shot info on the right
  if (phase === 'war') {
    const lastShot = lastShotByP0 || lastShotByP1;
    const shooter = lastShotByP0 ? 0 : lastShotByP1 ? 1 : -1;
    if (lastShot && shooter >= 0) {
      const shotBoard = shooter === 0 ? p0Obs.shotsBoard : p1Obs.shotsBoard;
      const result = getShotCellType(shotBoard.rows[lastShot.row][lastShot.col]);
      const resultText = result === 'hit' ? 'HIT!' : 'miss';
      const coordText = `${String.fromCharCode(65 + lastShot.col)}${lastShot.row + 1}`;
      c.textAlign = 'right';
      c.fillStyle = result === 'hit' ? COLORS.shotHit : COLORS.textDim;
      c.font = `bold ${smallFont}px sans-serif`;
      c.fillText(`P${shooter + 1} fires ${coordText}: ${resultText}`, width - 14, infoPanelH * 0.5);
    }
  }

  // ---- Board labels & boards ----
  const col1X = gridStartX;
  const col2X = gridStartX + boardPxW + rowLabelW + colGap;
  const row1Y = gridStartY + boardLabelH + labelGap;
  const row2Y = row1Y + boardPxH + boardLabelH + labelGap + colGap + 14; // +14 for col labels

  // Player labels
  const drawBoardLabel = (text: string, x: number, y: number, playerIdx: number) => {
    c.fillStyle = PLAYER_COLORS[playerIdx];
    c.font = `bold ${Math.max(11, cellSize * 0.32)}px sans-serif`;
    c.textAlign = 'left';
    c.textBaseline = 'bottom';
    c.fillText(text, x, y);
  };

  // Row 1: Ship boards
  drawBoardLabel('Player 1 -- Ships', col1X, row1Y - labelGap, 0);
  drawBoardLabel('Player 2 -- Ships', col2X, row1Y - labelGap, 1);
  drawBoard(c, p0Obs.ownBoard, col1X, row1Y, cellSize, 'own', lastShotOnP0Board, 0);
  drawBoard(c, p1Obs.ownBoard, col2X, row1Y, cellSize, 'own', lastShotOnP1Board, 1);

  // Row 2: Shots boards
  drawBoardLabel('Player 1 -- Shots', col1X, row2Y - labelGap, 0);
  drawBoardLabel('Player 2 -- Shots', col2X, row2Y - labelGap, 1);
  drawBoard(c, p0Obs.shotsBoard, col1X, row2Y, cellSize, 'shots', lastShotByP0, 0);
  drawBoard(c, p1Obs.shotsBoard, col2X, row2Y, cellSize, 'shots', lastShotByP1, 1);

  // ---- Damage indicators next to ship boards ----
  // Show which ships are sunk
  const drawDamageInfo = (stats: ReturnType<typeof countBoardStats>, x: number, y: number) => {
    if (stats.hitCells === 0) return;
    c.fillStyle = COLORS.shipHit;
    c.font = `bold ${Math.max(10, cellSize * 0.26)}px sans-serif`;
    c.textAlign = 'left';
    c.textBaseline = 'top';
    c.fillText(`${stats.hitCells} hit`, x, y);
  };

  drawDamageInfo(p0OwnStats, col1X + boardPxW + 6, row1Y + 2);
  drawDamageInfo(p1OwnStats, col2X + boardPxW + 6, row1Y + 2);

  // ---- Status bar ----
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over - Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over - Player 1 wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over - Player 2 wins!';
    statusBar.textContent = `${msg} | P1 hits: ${p0ShotStats.hits}, P2 hits: ${p1ShotStats.hits}`;
  } else {
    const phaseLabel = phase === 'placement' ? 'Ship Placement' : 'War';
    statusBar.textContent = `${phaseLabel} -- Player ${cp + 1}'s turn`;
  }
}
