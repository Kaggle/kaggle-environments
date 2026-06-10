import type { RendererOptions } from '@kaggle-environments/core';
import type { QuoridorBoardState, QuoridorMove, QuoridorStep } from './transformers/quoridorTransformer';

const PLAYER_X_COLOR = '#1f77b4';
const PLAYER_O_COLOR = '#d62728';
const CELL_FILL = '#fbf7e8';
const GOAL_FILL_X = '#dfecff'; // top row, x's goal (x moves up)
const GOAL_FILL_O = '#ffe2e2'; // bottom row, o's goal
const SKETCH_STROKE = '#3c3b37';
const SECONDARY_TEXT = '#444343';
const WALL_COLOR = '#5b4636';
const HIGHLIGHT_WALL_COLOR = '#f5b400';

const COLORS_BY_CODE: Record<string, string> = {
  x: PLAYER_X_COLOR,
  o: PLAYER_O_COLOR,
};

function colLabel(col: number): string {
  return String.fromCharCode('a'.charCodeAt(0) + col);
}

function getPlayerName(replay: any, idx: number): string {
  return replay?.info?.TeamNames?.[idx] ?? (idx === 0 ? 'Player X' : 'Player O');
}

interface BoardGeom {
  originX: number;
  originY: number;
  cellSize: number;
  wallGap: number;
  boardSize: number;
}

// Solve for cellSize and wallGap so the whole board fits in (innerW, innerH).
// Layout: N cells of size C separated by (N-1) gaps of size G, plus a small
// padding for axis labels. We fix G = max(4, C * 0.18).
function computeGeom(width: number, height: number, boardSize: number): BoardGeom {
  const labelPad = 22;
  const innerW = Math.max(0, width - labelPad * 2);
  const innerH = Math.max(0, height - labelPad * 2);
  // Solve C + 0.18 * C (for each gap) per cell: total = N*C + (N-1)*G.
  // With G = 0.18 * C: total = N*C + 0.18*(N-1)*C = C * (N + 0.18*(N-1)).
  const factor = boardSize + 0.18 * (boardSize - 1);
  const cellSize = Math.max(8, Math.min(innerW, innerH) / factor);
  const wallGap = Math.max(4, cellSize * 0.18);
  const boardW = boardSize * cellSize + (boardSize - 1) * wallGap;
  const boardH = boardW;
  const originX = (width - boardW) / 2;
  const originY = (height - boardH) / 2;
  return { originX, originY, cellSize, wallGap, boardSize };
}

function cellRect(geom: BoardGeom, col: number, row: number) {
  const x = geom.originX + col * (geom.cellSize + geom.wallGap);
  const y = geom.originY + row * (geom.cellSize + geom.wallGap);
  return { x, y, w: geom.cellSize, h: geom.cellSize };
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  state: QuoridorBoardState,
  lastMove: QuoridorMove | null
) {
  ctx.clearRect(0, 0, width, height);
  const geom = computeGeom(width, height, state.board_size);

  // Cells.
  for (let r = 0; r < state.board_size; r++) {
    for (let c = 0; c < state.board_size; c++) {
      const { x, y, w, h } = cellRect(geom, c, r);
      let fill = CELL_FILL;
      if (r === 0) fill = GOAL_FILL_X;
      else if (r === state.board_size - 1) fill = GOAL_FILL_O;
      ctx.fillStyle = fill;
      ctx.fillRect(x, y, w, h);
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = SKETCH_STROKE;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    }
  }
  ctx.setLineDash([]);

  // Pawns.
  for (const [code, label] of Object.entries(state.pawns)) {
    const m = /^([a-y])(\d{1,2})$/i.exec(label);
    if (!m) continue;
    const col = m[1].toLowerCase().charCodeAt(0) - 'a'.charCodeAt(0);
    const row = parseInt(m[2], 10) - 1;
    const { x, y, w, h } = cellRect(geom, col, row);
    const cx = x + w / 2;
    const cy = y + h / 2;
    const radius = Math.min(w, h) * 0.36;
    const color = COLORS_BY_CODE[code] ?? '#333';

    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = SKETCH_STROKE;
    ctx.stroke();

    ctx.fillStyle = '#fff';
    ctx.font = `bold ${Math.round(radius * 0.95)}px 'Inter', sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(code.toUpperCase(), cx, cy + radius * 0.04);
  }

  // Walls. A vertical wall "a1v" sits in the gap to the right of column a,
  // spanning rows 1-2 (two cells). A horizontal wall "a1h" sits in the gap
  // just below row 1, spanning columns a-b.
  const drawVerticalWall = (col: number, topRow: number, color: string) => {
    const { x, y, w, h } = cellRect(geom, col, topRow);
    const wallX = x + w; // just past the cell's right edge
    const wallY = y;
    const wallH = h * 2 + geom.wallGap;
    const wallW = geom.wallGap;
    ctx.fillStyle = color;
    ctx.fillRect(wallX, wallY, wallW, wallH);
  };
  const drawHorizontalWall = (col: number, topRow: number, color: string) => {
    const { x, y, w, h } = cellRect(geom, col, topRow);
    const wallX = x;
    const wallY = y + h;
    const wallH = geom.wallGap;
    const wallW = w * 2 + geom.wallGap;
    ctx.fillStyle = color;
    ctx.fillRect(wallX, wallY, wallW, wallH);
  };

  for (const label of state.vertical_walls) {
    const m = /^([a-y])(\d{1,2})v$/i.exec(label);
    if (!m) continue;
    const col = m[1].toLowerCase().charCodeAt(0) - 'a'.charCodeAt(0);
    const row = parseInt(m[2], 10) - 1;
    const isLast = lastMove?.kind === 'wall_v' && lastMove.col === col && lastMove.row === row;
    drawVerticalWall(col, row, isLast ? HIGHLIGHT_WALL_COLOR : WALL_COLOR);
  }
  for (const label of state.horizontal_walls) {
    const m = /^([a-y])(\d{1,2})h$/i.exec(label);
    if (!m) continue;
    const col = m[1].toLowerCase().charCodeAt(0) - 'a'.charCodeAt(0);
    const row = parseInt(m[2], 10) - 1;
    const isLast = lastMove?.kind === 'wall_h' && lastMove.col === col && lastMove.row === row;
    drawHorizontalWall(col, row, isLast ? HIGHLIGHT_WALL_COLOR : WALL_COLOR);
  }

  // Pawn-move highlight: ring around the destination cell.
  if (lastMove?.kind === 'pawn') {
    const { x, y, w, h } = cellRect(geom, lastMove.col, lastMove.row);
    ctx.strokeStyle = HIGHLIGHT_WALL_COLOR;
    ctx.lineWidth = Math.max(2.5, geom.cellSize * 0.08);
    ctx.setLineDash([]);
    ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
  }

  // Axis labels.
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.font = `${Math.max(10, Math.round(geom.cellSize * 0.32))}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let c = 0; c < state.board_size; c++) {
    const { x, w } = cellRect(geom, c, 0);
    const cx = x + w / 2;
    ctx.fillText(colLabel(c), cx, geom.originY - 11);
    ctx.fillText(
      colLabel(c),
      cx,
      geom.originY + state.board_size * geom.cellSize + (state.board_size - 1) * geom.wallGap + 11
    );
  }
  for (let r = 0; r < state.board_size; r++) {
    const { y, h } = cellRect(geom, 0, r);
    const cy = y + h / 2;
    ctx.textAlign = 'center';
    ctx.fillText(String(r + 1), geom.originX - 12, cy);
    ctx.fillText(
      String(r + 1),
      geom.originX + state.board_size * geom.cellSize + (state.board_size - 1) * geom.wallGap + 12,
      cy
    );
  }
}

export function renderer(options: RendererOptions<QuoridorStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as QuoridorStep[];
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

  const currentStep = steps[step] ?? steps[steps.length - 1];
  const observation = currentStep?.boardState ?? null;
  if (!observation) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const isTerminal = observation.is_terminal;
  const currentPlayer = observation.current_player;
  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const wallsRemaining = observation.walls_remaining ?? {};
  const activeIdx = isTerminal ? -1 : currentPlayer === 'x' ? 0 : currentPlayer === 'o' ? 1 : -1;

  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${PLAYER_X_COLOR};">
      <span>${playerNames[0]} <span style="opacity: 0.7;">(X)</span></span>
      <span class="walls">walls: ${wallsRemaining.x ?? '?'}</span>
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${PLAYER_O_COLOR};">
      <span>${playerNames[1]} <span style="opacity: 0.7;">(O)</span></span>
      <span class="walls">walls: ${wallsRemaining.o ?? '?'}</span>
    </span>
  `;

  const sizeAndDraw = () => {
    const wrapRect = wrap.getBoundingClientRect();
    const availW = wrapRect.width;
    const availH = wrapRect.height;
    if (availW <= 0 || availH <= 0) return;
    const side = Math.max(1, Math.floor(Math.min(availW, availH)));
    canvas.style.width = `${side}px`;
    canvas.style.height = `${side}px`;
    canvas.width = side;
    canvas.height = side;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawBoard(ctx, side, side, observation, currentStep?.lastMove ?? null);
  };

  requestAnimationFrame(sizeAndDraw);

  // Status line.
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
  const lastMove = currentStep?.lastMove ?? null;
  if (lastMove) {
    const lastActor = currentStep?.lastActor;
    const moverColor = lastActor === 'x' ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">${lastMove.raw}</span></span>`;
  }
  statusHTML += `<span class="annotation">move ${observation.move_number}</span>`;
  statusContainer.innerHTML = statusHTML;
}
