import type { RendererOptions } from '@kaggle-environments/core';
import type { BreakthroughBoardState, BreakthroughStep } from './transformers/breakthroughTransformer';

const INK = '#050001';
const SOFT_INK = '#3c3b37';
const SECONDARY_TEXT = '#444343';
const P0_COLOR = '#222222'; // Black ('b') ring + accent
const P1_COLOR = '#b8801f'; // White ('w') ring + accent
const CELL_LIGHT = '#fbf7e8';
const CELL_DARK = '#e8dfc4';
const HIGHLIGHT_FROM = 'rgba(60, 59, 55, 0.55)';
const CAPTURE_RING = '#c9381c';

interface ParsedMove {
  from: { row: number; col: number };
  to: { row: number; col: number };
  capture: boolean;
}

function colLabel(col: number): string {
  return String.fromCharCode('a'.charCodeAt(0) + col);
}

function rowLabel(row: number, totalRows: number): string {
  // OpenSpiel rows are top-to-bottom; visible label is bottom-up (1..N).
  return String(totalRows - row);
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player 1' : 'Player 2';
}

function parseMove(move: string | null, rows: number): ParsedMove | null {
  if (!move) return null;
  const capture = move.endsWith('*');
  const core = capture ? move.slice(0, -1) : move;
  // Format: <col1><row1><col2><row2>  (row labels are 1..rows; can be multi-digit on big boards)
  if (core.length < 4) return null;
  // Find the split point between (col1,row1) and (col2,row2): the first
  // character that is a letter after we've consumed at least one digit.
  let split = -1;
  for (let i = 2; i < core.length - 1; i++) {
    const ch = core.charCodeAt(i);
    if (ch >= 'a'.charCodeAt(0) && ch <= 'z'.charCodeAt(0)) {
      split = i;
      break;
    }
  }
  if (split < 0) return null;
  const fromCol = core.charCodeAt(0) - 'a'.charCodeAt(0);
  const fromRowLabel = parseInt(core.slice(1, split), 10);
  const toCol = core.charCodeAt(split) - 'a'.charCodeAt(0);
  const toRowLabel = parseInt(core.slice(split + 1), 10);
  if (Number.isNaN(fromRowLabel) || Number.isNaN(toRowLabel)) return null;
  return {
    from: { row: rows - fromRowLabel, col: fromCol },
    to: { row: rows - toRowLabel, col: toCol },
    capture,
  };
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: BreakthroughBoardState,
  highlight: ParsedMove | null,
  lastActor: 0 | 1 | null
) {
  ctx.clearRect(0, 0, width, height);

  const padding = 36;
  const innerW = Math.max(0, width - padding * 2);
  const innerH = Math.max(0, height - padding * 2);
  const cellSize = Math.max(16, Math.min(innerW / obs.columns, innerH / obs.rows));
  const boardW = cellSize * obs.columns;
  const boardH = cellSize * obs.rows;
  const originX = (width - boardW) / 2;
  const originY = (height - boardH) / 2;

  // Checkerboard cells.
  for (let r = 0; r < obs.rows; r++) {
    for (let c = 0; c < obs.columns; c++) {
      const x = originX + c * cellSize;
      const y = originY + r * cellSize;
      ctx.fillStyle = (r + c) % 2 === 0 ? CELL_LIGHT : CELL_DARK;
      ctx.fillRect(x, y, cellSize, cellSize);
    }
  }

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

  // Tint home rows so it's clear which direction each side is pushing.
  ctx.fillStyle = 'rgba(34, 34, 34, 0.06)';
  ctx.fillRect(originX, originY, boardW, cellSize);
  ctx.fillStyle = 'rgba(184, 128, 31, 0.08)';
  ctx.fillRect(originX, originY + boardH - cellSize, boardW, cellSize);

  // Pieces.
  const radius = cellSize * 0.36;
  ctx.font = `700 ${Math.round(cellSize * 0.42)}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < obs.rows; r++) {
    for (let c = 0; c < obs.columns; c++) {
      const cell = obs.board?.[r]?.[c];
      if (cell !== 'b' && cell !== 'w') continue;
      const cx = originX + (c + 0.5) * cellSize;
      const cy = originY + (r + 0.5) * cellSize;
      const isBlack = cell === 'b';
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = isBlack ? INK : '#f8f4e3';
      ctx.fill();
      ctx.lineWidth = 1.75;
      ctx.strokeStyle = isBlack ? P0_COLOR : P1_COLOR;
      ctx.stroke();
    }
  }

  // Last-move highlight: dashed origin ring + arrow to destination.
  if (highlight) {
    const fromX = originX + (highlight.from.col + 0.5) * cellSize;
    const fromY = originY + (highlight.from.row + 0.5) * cellSize;
    const toX = originX + (highlight.to.col + 0.5) * cellSize;
    const toY = originY + (highlight.to.row + 0.5) * cellSize;
    const actorColor = lastActor === 0 ? P0_COLOR : P1_COLOR;

    // Ghost on FROM (square is now empty).
    ctx.beginPath();
    ctx.arc(fromX, fromY, radius * 0.85, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(245, 241, 226, 0.5)';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = HIGHLIGHT_FROM;
    ctx.stroke();
    ctx.setLineDash([]);

    // Ring on TO. Red ring if capture, actor color otherwise.
    ctx.beginPath();
    ctx.arc(toX, toY, radius + 3.5, 0, Math.PI * 2);
    ctx.lineWidth = 2.8;
    ctx.strokeStyle = highlight.capture ? CAPTURE_RING : actorColor;
    ctx.stroke();

    if (highlight.capture) {
      // Small "x" mark to flag the captured opponent piece.
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = CAPTURE_RING;
      const off = radius * 0.55;
      ctx.beginPath();
      ctx.moveTo(toX - off, toY - off);
      ctx.lineTo(toX + off, toY + off);
      ctx.moveTo(toX + off, toY - off);
      ctx.lineTo(toX - off, toY + off);
      ctx.stroke();
    }
  }

  // Coordinate labels.
  ctx.font = `${Math.round(cellSize * 0.28)}px 'Inter', sans-serif`;
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.textBaseline = 'top';
  for (let c = 0; c < obs.columns; c++) {
    ctx.fillText(colLabel(c), originX + (c + 0.5) * cellSize, originY + boardH + 6);
  }
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (let r = 0; r < obs.rows; r++) {
    ctx.fillText(rowLabel(r, obs.rows), originX - 6, originY + (r + 0.5) * cellSize);
  }
}

export function renderer(options: RendererOptions<BreakthroughStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as BreakthroughStep[];
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
  const obs: BreakthroughBoardState | null = currentStep?.boardState ?? null;
  if (!obs) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const isTerminal = obs.is_terminal;
  const activeIdx = isTerminal ? -1 : obs.current_player === 'b' ? 0 : obs.current_player === 'w' ? 1 : -1;
  const pieceCounts = obs.pieces ?? { b: 0, w: 0 };

  header.innerHTML = `
    <span class="player p0 sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${P0_COLOR};">
      <span class="glyph"></span>${playerNames[0]} <span class="count">${pieceCounts.b}</span>
    </span>
    <span class="vs">vs</span>
    <span class="player p1 sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${P1_COLOR};">
      <span class="glyph"></span>${playerNames[1]} <span class="count">${pieceCounts.w}</span>
    </span>
  `;

  // last move parity: even-index move number => move was by P0 (b).
  const lastActor: 0 | 1 | null = obs.move_number > 0 ? (((obs.move_number - 1) % 2) as 0 | 1) : null;
  const highlight = parseMove(obs.last_move, obs.rows);

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
    drawBoard(ctx, cssW, cssH, obs, highlight, lastActor);
  };

  requestAnimationFrame(sizeAndDraw);

  let statusHTML = '';
  if (isTerminal) {
    if (obs.winner === 'b') {
      statusHTML = `<span style="color: ${P0_COLOR};">${playerNames[0]} (b) wins!</span>`;
    } else if (obs.winner === 'w') {
      statusHTML = `<span style="color: ${P1_COLOR};">${playerNames[1]} (w) wins!</span>`;
    } else {
      statusHTML = `<span>Game over: ${obs.winner ?? 'finished'}</span>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? P0_COLOR : P1_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;
  }
  if (obs.last_move) {
    const moverColor = lastActor === 0 ? P0_COLOR : P1_COLOR;
    const captureTag = obs.last_move.endsWith('*')
      ? `<span style="color: ${CAPTURE_RING}; font-weight: 700;"> capture</span>`
      : '';
    statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">${obs.last_move}</span>${captureTag}</span>`;
  }
  statusHTML += `<span class="annotation">move ${obs.move_number}</span>`;
  statusContainer.innerHTML = statusHTML;
}
