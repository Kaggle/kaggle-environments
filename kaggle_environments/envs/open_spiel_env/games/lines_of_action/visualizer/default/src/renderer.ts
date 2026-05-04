import type { RendererOptions } from '@kaggle-environments/core';
import type { LoaBoardState, LoaCell, LoaMove, LoaStep } from './transformers/loaTransformer';

const PLAYER_X_COLOR = '#1f77b4';
const PLAYER_O_COLOR = '#d62728';
const CELL_LIGHT = '#fbf7e8';
const CELL_DARK = '#e7dfc1';
const SECONDARY_TEXT = '#444343';
const SKETCH_STROKE = '#3c3b37';

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

function countPieces(board: LoaCell[][]): { x: number; o: number } {
  let x = 0;
  let o = 0;
  for (const row of board) {
    for (const cell of row) {
      if (cell === 'x') x++;
      else if (cell === 'o') o++;
    }
  }
  return { x, o };
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  size: number
) {
  const headLen = size * 0.35;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  // Pull the tip back so the arrow lands on the destination piece edge.
  const inset = size * 0.45;
  const tx = x2 - inset * Math.cos(angle);
  const ty = y2 - inset * Math.sin(angle);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = Math.max(2, size * 0.08);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(tx - headLen * Math.cos(angle - Math.PI / 6), ty - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(tx - headLen * Math.cos(angle + Math.PI / 6), ty - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  board: LoaCell[][],
  lastMove: LoaMove | null,
  lastActor: number | null
) {
  ctx.clearRect(0, 0, width, height);

  const padding = 24;
  const innerSize = Math.max(0, Math.min(width, height) - padding * 2);
  const cellSize = innerSize / 8;
  const xOffset = (width - cellSize * 8) / 2;
  const yOffset = (height - cellSize * 8) / 2;

  // Cells. board[0] is rank 1 (bottom); we draw rank 8 at the top so flip the
  // row index when computing the screen position.
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const screenRow = 7 - r;
      const x = xOffset + c * cellSize;
      const y = yOffset + screenRow * cellSize;
      ctx.fillStyle = (r + c) % 2 === 0 ? CELL_LIGHT : CELL_DARK;
      ctx.fillRect(x, y, cellSize, cellSize);
    }
  }

  // Sketched outer border + grid lines.
  ctx.strokeStyle = SKETCH_STROKE;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([]);
  ctx.strokeRect(xOffset, yOffset, cellSize * 8, cellSize * 8);
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  for (let i = 1; i < 8; i++) {
    ctx.beginPath();
    ctx.moveTo(xOffset + i * cellSize, yOffset);
    ctx.lineTo(xOffset + i * cellSize, yOffset + cellSize * 8);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xOffset, yOffset + i * cellSize);
    ctx.lineTo(xOffset + cellSize * 8, yOffset + i * cellSize);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Highlight the move's source and destination squares before drawing pieces
  // so the cell tint sits beneath the piece.
  if (lastMove) {
    const moverColor = lastActor === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    const fromX = xOffset + lastMove.fromCol * cellSize;
    const fromY = yOffset + (7 - lastMove.fromRow) * cellSize;
    const toX = xOffset + lastMove.toCol * cellSize;
    const toY = yOffset + (7 - lastMove.toRow) * cellSize;
    ctx.save();
    ctx.fillStyle = moverColor;
    ctx.globalAlpha = 0.18;
    ctx.fillRect(fromX, fromY, cellSize, cellSize);
    ctx.fillRect(toX, toY, cellSize, cellSize);
    ctx.restore();
    ctx.save();
    ctx.strokeStyle = moverColor;
    ctx.lineWidth = Math.max(2, cellSize * 0.08);
    ctx.strokeRect(toX + 1, toY + 1, cellSize - 2, cellSize - 2);
    ctx.restore();
  }

  // Pieces.
  const pieceRadius = cellSize * 0.36;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = board[r]?.[c];
      if (cell !== 'x' && cell !== 'o') continue;
      const cx = xOffset + c * cellSize + cellSize / 2;
      const cy = yOffset + (7 - r) * cellSize + cellSize / 2;
      const color = cell === 'x' ? PLAYER_X_COLOR : PLAYER_O_COLOR;
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, pieceRadius, 0, Math.PI * 2);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = Math.max(1, cellSize * 0.05);
      ctx.strokeStyle = SKETCH_STROKE;
      ctx.stroke();
      // Bright "checker" highlight for legibility.
      ctx.beginPath();
      ctx.arc(cx - pieceRadius * 0.3, cy - pieceRadius * 0.3, pieceRadius * 0.35, 0, Math.PI * 2);
      ctx.closePath();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.fill();
      ctx.restore();
    }
  }

  // Arrow over the move (drawn after pieces so it's on top).
  if (lastMove) {
    const moverColor = lastActor === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    const fromCx = xOffset + lastMove.fromCol * cellSize + cellSize / 2;
    const fromCy = yOffset + (7 - lastMove.fromRow) * cellSize + cellSize / 2;
    const toCx = xOffset + lastMove.toCol * cellSize + cellSize / 2;
    const toCy = yOffset + (7 - lastMove.toRow) * cellSize + cellSize / 2;
    drawArrow(ctx, fromCx, fromCy, toCx, toCy, moverColor, cellSize);
    if (lastMove.capture) {
      // Mark the capture with a small gold "x" badge in the destination corner.
      ctx.save();
      ctx.fillStyle = '#c89b1e';
      ctx.strokeStyle = SKETCH_STROKE;
      const badgeR = cellSize * 0.18;
      const bx = xOffset + lastMove.toCol * cellSize + cellSize - badgeR - 2;
      const by = yOffset + (7 - lastMove.toRow) * cellSize + badgeR + 2;
      ctx.beginPath();
      ctx.arc(bx, by, badgeR, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = `700 ${Math.round(badgeR * 1.4)}px 'Inter', sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('×', bx, by + badgeR * 0.05);
      ctx.restore();
    }
  }

  // Coordinate labels.
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.font = `${Math.round(cellSize * 0.28)}px 'Inter', sans-serif`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  for (let c = 0; c < 8; c++) {
    ctx.fillText(colLabel(c), xOffset + c * cellSize + cellSize / 2, yOffset + cellSize * 8 + cellSize * 0.32);
  }
  ctx.textAlign = 'right';
  for (let r = 0; r < 8; r++) {
    ctx.fillText(`${r + 1}`, xOffset - cellSize * 0.12, yOffset + (7 - r) * cellSize + cellSize / 2);
  }
}

export function renderer(options: RendererOptions<LoaStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as LoaStep[];
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
  const observation: LoaBoardState | null = currentStep?.boardState ?? null;
  if (!observation) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const isTerminal = observation.is_terminal;
  const currentPlayer = observation.current_player;
  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const activeIdx = isTerminal ? -1 : currentPlayer === 'x' ? 0 : currentPlayer === 'o' ? 1 : -1;
  const counts = countPieces(observation.board);

  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${PLAYER_X_COLOR};">
      ${playerNames[0]} <span style="opacity: 0.7;">(X)</span>
      <span class="count">${counts.x}</span>
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${PLAYER_O_COLOR};">
      ${playerNames[1]} <span style="opacity: 0.7;">(O)</span>
      <span class="count">${counts.o}</span>
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
    drawBoard(ctx, side, side, observation.board, currentStep.lastMove, currentStep.lastActor);
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
    const moverColor = currentStep.lastActor === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    const verb = currentStep.lastMove?.capture ? 'capture' : 'move';
    statusHTML += `<span class="annotation">${verb}: <span style="color: ${moverColor}; font-weight: 600;">${observation.last_move}</span></span>`;
  }
  statusHTML += `<span class="annotation">move ${observation.move_number}</span>`;
  statusContainer.innerHTML = statusHTML;
}
