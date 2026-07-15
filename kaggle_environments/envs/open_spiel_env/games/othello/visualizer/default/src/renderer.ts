import type { RendererOptions } from '@kaggle-environments/core';
import type { ReversiBoardState, ReversiCell, ReversiStep } from './transformers/reversiTransformer';

const INK = '#050001';
const SOFT_INK = '#3c3b37';
const SECONDARY_TEXT = '#444343';
const BOARD_TINT = '#e8e0c4'; // muted parchment-toned cell fill
const P0_COLOR = '#050001'; // Black disk (x) ring + accent
const P1_COLOR = '#c9a24a'; // White disk (o) ring + accent (warm gold on parchment)
const DISK_WHITE = '#f8f4e3'; // creamy off-white for White disks
const PLACED_RING = '#c9381c'; // strong accent for the newly-placed disk
const FLIP_RING = '#4a7f9d'; // muted blue for disks flipped this turn
const PASS_TAG = '#8a6d3b';

interface MoveHighlight {
  placed: { row: number; col: number } | null;
  flipped: { row: number; col: number }[];
  wasPass: boolean;
}

function colLabel(col: number): string {
  return String.fromCharCode('a'.charCodeAt(0) + col);
}

function rowLabel(row: number): string {
  // OpenSpiel row 0 is the top of the display (rank 1); mirror that here.
  return String(row + 1);
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Black' : 'White';
}

function diffBoards(prev: ReversiCell[][] | null, curr: ReversiCell[][], lastMove: string | null): MoveHighlight {
  const wasPass = lastMove === 'pass';
  const highlight: MoveHighlight = { placed: null, flipped: [], wasPass };
  if (!prev || wasPass) return highlight;
  for (let r = 0; r < curr.length; r++) {
    for (let c = 0; c < curr[r].length; c++) {
      const before = prev[r]?.[c] ?? '';
      const after = curr[r][c];
      if (before === after) continue;
      if (before === '' && after !== '') {
        highlight.placed = { row: r, col: c };
      } else if (before !== '' && after !== '' && before !== after) {
        highlight.flipped.push({ row: r, col: c });
      }
    }
  }
  return highlight;
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: ReversiBoardState,
  highlight: MoveHighlight
) {
  ctx.clearRect(0, 0, width, height);

  const padding = 32;
  const innerW = Math.max(0, width - padding * 2);
  const innerH = Math.max(0, height - padding * 2);
  const cellSize = Math.max(16, Math.min(innerW / obs.columns, innerH / obs.rows));
  const boardW = cellSize * obs.columns;
  const boardH = cellSize * obs.rows;
  const originX = (width - boardW) / 2;
  const originY = (height - boardH) / 2;

  // Solid board tint (no checkerboard — traditional Reversi has a uniform surface).
  ctx.fillStyle = BOARD_TINT;
  ctx.fillRect(originX, originY, boardW, boardH);

  // Sketched grid lines.
  ctx.strokeStyle = SOFT_INK;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  for (let i = 0; i <= obs.columns; i++) {
    ctx.beginPath();
    ctx.moveTo(originX + i * cellSize, originY);
    ctx.lineTo(originX + i * cellSize, originY + boardH);
    ctx.stroke();
  }
  for (let i = 0; i <= obs.rows; i++) {
    ctx.beginPath();
    ctx.moveTo(originX, originY + i * cellSize);
    ctx.lineTo(originX + boardW, originY + i * cellSize);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Disks.
  const radius = cellSize * 0.4;
  const flippedSet = new Set(highlight.flipped.map((p) => `${p.row},${p.col}`));

  for (let r = 0; r < obs.rows; r++) {
    for (let c = 0; c < obs.columns; c++) {
      const cell = obs.board?.[r]?.[c];
      if (cell !== 'x' && cell !== 'o') continue;
      const cx = originX + (c + 0.5) * cellSize;
      const cy = originY + (r + 0.5) * cellSize;
      const isBlack = cell === 'x';
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = isBlack ? INK : DISK_WHITE;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = isBlack ? P0_COLOR : P1_COLOR;
      ctx.stroke();

      // Flipped-this-turn overlay: dashed blue ring.
      if (flippedSet.has(`${r},${c}`)) {
        ctx.beginPath();
        ctx.arc(cx, cy, radius + 3.5, 0, Math.PI * 2);
        ctx.lineWidth = 2.4;
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = FLIP_RING;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  // Newly-placed disk gets a bold red ring on top so it's unambiguously the move.
  if (highlight.placed) {
    const cx = originX + (highlight.placed.col + 0.5) * cellSize;
    const cy = originY + (highlight.placed.row + 0.5) * cellSize;
    ctx.beginPath();
    ctx.arc(cx, cy, radius + 5, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = PLACED_RING;
    ctx.stroke();
  }

  // Coordinate labels (files a..h below, ranks 1..8 to the left).
  ctx.font = `${Math.round(cellSize * 0.28)}px 'Inter', sans-serif`;
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.textBaseline = 'top';
  ctx.textAlign = 'center';
  for (let c = 0; c < obs.columns; c++) {
    ctx.fillText(colLabel(c), originX + (c + 0.5) * cellSize, originY + boardH + 6);
  }
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (let r = 0; r < obs.rows; r++) {
    ctx.fillText(rowLabel(r), originX - 6, originY + (r + 0.5) * cellSize);
  }
}

export function renderer(options: RendererOptions<ReversiStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as ReversiStep[];
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
  const obs: ReversiBoardState | null = currentStep?.boardState ?? null;
  if (!obs) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const prevStep = step > 0 ? steps[step - 1] : null;
  const prevBoard = prevStep?.boardState?.board ?? null;
  const highlight = diffBoards(prevBoard, obs.board, obs.last_move);

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const isTerminal = obs.is_terminal;
  const activeIdx = isTerminal ? -1 : obs.current_player === 'x' ? 0 : obs.current_player === 'o' ? 1 : -1;
  const disks = obs.disks ?? { x: 0, o: 0 };

  header.innerHTML = `
    <span class="player p0 sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${P0_COLOR};">
      <span class="glyph"></span>${playerNames[0]} <span class="count">${disks.x}</span>
    </span>
    <span class="vs">vs</span>
    <span class="player p1 sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${P1_COLOR};">
      <span class="glyph"></span>${playerNames[1]} <span class="count">${disks.o}</span>
    </span>
  `;

  // last-move parity: move history length equals total actions so far;
  // even-index action (0-based) was Black's, so the LAST action was Black
  // if move_number is odd.
  const lastActor: 0 | 1 | null = obs.move_number > 0 ? (((obs.move_number - 1) % 2) as 0 | 1) : null;

  const sizeAndDraw = () => {
    const wrapRect = wrap.getBoundingClientRect();
    const availW = wrapRect.width;
    const availH = wrapRect.height;
    if (availW <= 0 || availH <= 0) return;
    const cssW = Math.max(1, Math.floor(availW));
    const cssH = Math.max(1, Math.floor(availH));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = cssW;
    canvas.height = cssH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawBoard(ctx, cssW, cssH, obs, highlight);
  };

  requestAnimationFrame(sizeAndDraw);

  let statusHTML = '';
  if (isTerminal) {
    if (obs.winner === 'x') {
      statusHTML = `<span style="color: ${P0_COLOR};">${playerNames[0]} wins ${disks.x}–${disks.o}</span>`;
    } else if (obs.winner === 'o') {
      statusHTML = `<span style="color: ${P1_COLOR};">${playerNames[1]} wins ${disks.o}–${disks.x}</span>`;
    } else {
      statusHTML = `<span>Draw ${disks.x}–${disks.o}</span>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? P0_COLOR : P1_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    const passTag = obs.must_pass ? ` <span style="color: ${PASS_TAG}; font-weight: 700;">(must pass)</span>` : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span>${passTag}</span>`;
  }
  if (obs.last_move) {
    const moverColor = lastActor === 0 ? P0_COLOR : P1_COLOR;
    const flipCount = highlight.flipped.length;
    if (highlight.wasPass) {
      statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">pass</span></span>`;
    } else {
      const flipTag =
        flipCount > 0 ? ` <span style="color: ${FLIP_RING}; font-weight: 700;">+${flipCount} flipped</span>` : '';
      statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">${obs.last_move}</span>${flipTag}</span>`;
    }
  }
  statusHTML += `<span class="annotation">move ${obs.move_number}</span>`;
  statusContainer.innerHTML = statusHTML;
}
