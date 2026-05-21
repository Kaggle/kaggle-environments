import type { RendererOptions } from '@kaggle-environments/core';
import type { HavannahBoardState, HavannahStep } from './transformers/havannahTransformer';

const PLAYER_X_COLOR = '#1f77b4';
const PLAYER_O_COLOR = '#d62728';
const CELL_FILL = '#fbf7e8';
const CORNER_FILL = '#f3e2a8';
const EDGE_FILL = '#f5ecc8';
const SECONDARY_TEXT = '#444343';
const SKETCH_STROKE = '#3c3b37';
const BORDER_STROKE = '#7c7c7c';

const SQRT3 = Math.sqrt(3);

function colLabel(col: number): string {
  return String.fromCharCode('a'.charCodeAt(0) + col);
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player X' : 'Player O';
}

// Translate the (x, y) board coordinate into the array index within row y of
// the proxy's variable-width board.
function rowOffset(y: number, boardSize: number): number {
  return y < boardSize ? 0 : y - boardSize + 1;
}

function rowLength(y: number, boardSize: number): number {
  const diameter = boardSize * 2 - 1;
  if (y < boardSize) return boardSize + y;
  return diameter - (y - boardSize + 1);
}

// Classify a board cell. Corner cells sit at the 6 extreme (x, y) positions;
// edge cells sit along the six outer sides (excluding corners).
function classifyCell(x: number, y: number, boardSize: number): 'corner' | 'edge' | 'inner' {
  const m = boardSize - 1;
  const e = boardSize * 2 - 2;
  const corners: Array<[number, number]> = [
    [0, 0],
    [m, 0],
    [e, m],
    [e, e],
    [m, e],
    [0, m],
  ];
  for (const [cx, cy] of corners) {
    if (cx === x && cy === y) return 'corner';
  }
  if (y === 0 && x > 0 && x < m) return 'edge';
  if (x - y === m && x !== m && x !== e) return 'edge';
  if (x === e && y !== m && y !== e) return 'edge';
  if (y === e && x !== e && x !== m) return 'edge';
  if (y - x === m && y !== m && y !== e) return 'edge';
  if (x === 0 && y !== 0 && y !== m) return 'edge';
  return 'inner';
}

function drawHavannahBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  board: (string | null)[][],
  boardSize: number,
  highlightCell: { x: number; y: number } | null,
  highlightColor: string
) {
  ctx.clearRect(0, 0, width, height);

  const diameter = boardSize * 2 - 1;
  const padding = 20;
  const innerW = Math.max(0, width - padding * 2);
  const innerH = Math.max(0, height - padding * 2);

  // q = x - y, r = y in pointy-top axial coords. The board fills a
  // regular flat-top hexagon of corner-to-center radius R = (boardSize - 1).
  // Total pixel bounding box (centers + half cell):
  //   width  = diameter * sqrt(3) * s
  //   height = (1.5 * (diameter - 1) + 2) * s = (3*boardSize - 1) * s
  const sizeFromW = innerW / (diameter * SQRT3);
  const sizeFromH = innerH / (3 * boardSize - 1);
  const size = Math.max(4, Math.min(sizeFromW, sizeFromH));

  // Center the board within the canvas.
  const m = boardSize - 1;
  const boardW = diameter * SQRT3 * size;
  const boardH = (3 * boardSize - 1) * size;
  const xOffset = (width - boardW) / 2 + SQRT3 * size * (m / 2) + (SQRT3 * size) / 2;
  const yOffset = (height - boardH) / 2 + size;

  const centerOf = (x: number, y: number) => ({
    px: xOffset + SQRT3 * size * (x - y / 2),
    py: yOffset + 1.5 * size * y,
  });

  const hexPath = (cx: number, cy: number, s: number) => {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i + Math.PI / 2;
      const px = cx + s * Math.cos(angle);
      const py = cy + s * Math.sin(angle);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
  };

  // Cells.
  ctx.font = `${Math.round(size * 0.75)}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let y = 0; y < diameter; y++) {
    const off = rowOffset(y, boardSize);
    const len = rowLength(y, boardSize);
    for (let i = 0; i < len; i++) {
      const x = off + i;
      const { px, py } = centerOf(x, y);
      const cell = board?.[y]?.[i] ?? null;
      const kind = classifyCell(x, y, boardSize);

      hexPath(px, py, size * 0.95);
      ctx.fillStyle = kind === 'corner' ? CORNER_FILL : kind === 'edge' ? EDGE_FILL : CELL_FILL;
      ctx.fill();
      ctx.lineWidth = 1.25;
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = SKETCH_STROKE;
      ctx.stroke();
      ctx.setLineDash([]);

      if (cell === 'x') {
        ctx.fillStyle = PLAYER_X_COLOR;
        ctx.fillText('X', px, py + size * 0.05);
      } else if (cell === 'o') {
        ctx.fillStyle = PLAYER_O_COLOR;
        ctx.fillText('O', px, py + size * 0.05);
      }
    }
  }

  // Outer border tracing the hexagonal perimeter. The board's 6 corners sit at
  // angles (idx - 2) * 60° from board center (idx=0 is top-left). Step from
  // each corner center outward to reach the outer flat of the hex board.
  const cornerCoords: Array<[number, number]> = [
    [0, 0],
    [boardSize - 1, 0],
    [diameter - 1, boardSize - 1],
    [diameter - 1, diameter - 1],
    [boardSize - 1, diameter - 1],
    [0, boardSize - 1],
  ];
  ctx.strokeStyle = BORDER_STROKE;
  ctx.lineWidth = Math.max(1.5, size * 0.08);
  ctx.setLineDash([]);
  ctx.beginPath();
  cornerCoords.forEach(([cx, cy], idx) => {
    const { px, py } = centerOf(cx, cy);
    const angle = ((idx - 2) * Math.PI) / 3;
    const ox = px + size * Math.cos(angle);
    const oy = py + size * Math.sin(angle);
    if (idx === 0) ctx.moveTo(ox, oy);
    else ctx.lineTo(ox, oy);
  });
  ctx.closePath();
  ctx.stroke();

  // Highlight the last move with a colored ring.
  if (highlightCell) {
    const { px, py } = centerOf(highlightCell.x, highlightCell.y);
    hexPath(px, py, size * 0.95);
    ctx.lineWidth = Math.max(2.5, size * 0.14);
    ctx.setLineDash([]);
    ctx.strokeStyle = highlightColor;
    ctx.stroke();
  }

  // Column labels along the top (a, b, c, ...). Each column appears in row y=0
  // for x in [0, boardSize-1] and in the right diagonal otherwise.
  ctx.font = `${Math.round(size * 0.36)}px 'Inter', sans-serif`;
  ctx.fillStyle = SECONDARY_TEXT;
  for (let col = 0; col < diameter; col++) {
    // Top-most cell with this column letter is at (col, max(0, col - boardSize + 1)).
    const topY = Math.max(0, col - boardSize + 1);
    const { px, py } = centerOf(col, topY);
    ctx.fillText(colLabel(col), px, py - size - size * 0.2);
  }
  // Row labels (1..diameter) on the left side of each row.
  for (let y = 0; y < diameter; y++) {
    const off = rowOffset(y, boardSize);
    const { px, py } = centerOf(off, y);
    ctx.fillText(`${y + 1}`, px - SQRT3 * size * 0.55 - size * 0.2, py);
  }
}

export function renderer(options: RendererOptions<HavannahStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as HavannahStep[];
  if (!steps.length) return;

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="board-wrap"><canvas></canvas></div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const wrap = parent.querySelector('.board-wrap') as HTMLDivElement;
  const canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;

  const currentStep = steps[step];
  const observation: HavannahBoardState | null = currentStep?.boardState ?? null;
  if (!observation) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const boardSize = observation.board_size;
  const diameter = boardSize * 2 - 1;
  const isTerminal = observation.is_terminal;
  const currentPlayer = observation.current_player;

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const activeIdx = isTerminal ? -1 : currentPlayer === 'x' ? 0 : currentPlayer === 'o' ? 1 : -1;

  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${PLAYER_X_COLOR};">
      ${playerNames[0]} <span style="opacity: 0.7;">(X)</span>
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${PLAYER_O_COLOR};">
      ${playerNames[1]} <span style="opacity: 0.7;">(O)</span>
    </span>
  `;

  const lastActor = observation.move_number > 0 ? (observation.move_number - 1) % 2 : null;
  let lastCell: { x: number; y: number } | null = null;
  if (observation.last_move) {
    const colChar = observation.last_move.charCodeAt(0) - 'a'.charCodeAt(0);
    const rowNum = parseInt(observation.last_move.slice(1), 10) - 1;
    if (colChar >= 0 && rowNum >= 0) lastCell = { x: colChar, y: rowNum };
  }
  const highlightColor = lastActor === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;

  // Aspect ratio of the hexagonal board bounding box.
  const aspect = (diameter * SQRT3) / (3 * boardSize - 1);

  const sizeAndDraw = () => {
    const wrapRect = wrap.getBoundingClientRect();
    const availW = wrapRect.width;
    const availH = wrapRect.height;
    if (availW <= 0 || availH <= 0) return;
    let cssW = Math.min(availW, availH * aspect);
    let cssH = cssW / aspect;
    cssW = Math.max(1, Math.floor(cssW));
    cssH = Math.max(1, Math.floor(cssH));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = cssW;
    canvas.height = cssH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawHavannahBoard(ctx, cssW, cssH, observation.board, boardSize, lastCell, highlightColor);
  };

  requestAnimationFrame(sizeAndDraw);

  let statusHTML = '';
  if (isTerminal) {
    if (observation.winner === 'x') {
      statusHTML = `<span style="color: ${PLAYER_X_COLOR};">${playerNames[0]} (X) wins!</span>`;
    } else if (observation.winner === 'o') {
      statusHTML = `<span style="color: ${PLAYER_O_COLOR};">${playerNames[1]} (O) wins!</span>`;
    } else {
      statusHTML = `<span>Draw</span>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;
  }
  if (observation.last_move) {
    const moverColor = lastActor === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">${observation.last_move}</span></span>`;
  }
  statusHTML += `<span class="annotation">move ${observation.move_number}</span>`;
  statusContainer.innerHTML = statusHTML;
}
