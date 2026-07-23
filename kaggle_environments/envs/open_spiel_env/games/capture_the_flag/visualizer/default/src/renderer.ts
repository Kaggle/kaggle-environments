import type { RendererOptions } from '@kaggle-environments/core';
import type { CtfBoardState, CtfStep } from './transformers/captureTheFlagTransformer';

const SOFT_INK = '#3c3b37';
const SECONDARY_TEXT = '#444343';
const P0_COLOR = '#1f4f8b'; // Player A
const P1_COLOR = '#9a3324'; // Player B
const P0_TERRITORY = 'rgba(31, 79, 139, 0.08)';
const P1_TERRITORY = 'rgba(154, 51, 36, 0.08)';
const OBSTACLE_COLOR = '#3c3b37';

function escapeHtml(s: string): string {
  return s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player A' : 'Player B';
}

function drawFlag(ctx: CanvasRenderingContext2D, cx: number, cy: number, size: number, color: string) {
  // Pennant on a pole. Pole is vertical; flag triangles out to the right.
  const poleTop = cy - size * 0.55;
  const poleBottom = cy + size * 0.4;
  ctx.strokeStyle = SOFT_INK;
  ctx.lineWidth = Math.max(1, size * 0.05);
  ctx.beginPath();
  ctx.moveTo(cx, poleTop);
  ctx.lineTo(cx, poleBottom);
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(cx, poleTop);
  ctx.lineTo(cx + size * 0.55, poleTop + size * 0.18);
  ctx.lineTo(cx, poleTop + size * 0.4);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = SOFT_INK;
  ctx.lineWidth = Math.max(1, size * 0.04);
  ctx.stroke();
}

function drawUnit(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  cellSize: number,
  color: string,
  label: string,
  carriedFlagColor: string | null
) {
  const radius = cellSize * 0.34;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 1.75;
  ctx.strokeStyle = SOFT_INK;
  ctx.stroke();

  ctx.font = `700 ${Math.round(cellSize * 0.38)}px 'Inter', sans-serif`;
  ctx.fillStyle = '#f5f1e2';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, cx, cy + 1);

  if (carriedFlagColor) {
    // Little pennant riding on the top-right of the unit.
    drawFlag(ctx, cx + radius * 0.7, cy - radius * 0.55, cellSize * 0.55, carriedFlagColor);
  }
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  color: string
) {
  const dx = toX - fromX;
  const dy = toY - fromY;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 0.5) return;
  const nx = dx / dist;
  const ny = dy / dist;
  // Shorten the arrow slightly so it doesn't overlap the unit circle.
  const trimTail = Math.min(dist * 0.25, 14);
  const trimHead = Math.min(dist * 0.35, 20);
  const sx = fromX + nx * trimTail;
  const sy = fromY + ny * trimTail;
  const ex = toX - nx * trimHead;
  const ey = toY - ny * trimHead;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.setLineDash([]);

  // Arrowhead.
  const headLen = 8;
  const headAngle = Math.PI / 6;
  const angle = Math.atan2(ny, nx);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(ex + Math.cos(angle) * headLen * 0.4, ey + Math.sin(angle) * headLen * 0.4);
  ctx.lineTo(ex - Math.cos(angle - headAngle) * headLen, ey - Math.sin(angle - headAngle) * headLen);
  ctx.lineTo(ex - Math.cos(angle + headAngle) * headLen, ey - Math.sin(angle + headAngle) * headLen);
  ctx.closePath();
  ctx.fill();
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: CtfBoardState,
  prevObs: CtfBoardState | null
) {
  ctx.clearRect(0, 0, width, height);

  const rows = obs.num_rows;
  const cols = obs.num_cols;
  // Reserve space around the board so row/column labels can render without
  // being clipped by the canvas edge. Bottom needs the most room since
  // column labels sit below the board.
  const padLeft = 24;
  const padRight = 8;
  const padTop = 8;
  const padBottom = 44;
  const availW = Math.max(0, width - padLeft - padRight);
  const availH = Math.max(0, height - padTop - padBottom);
  const cellSize = Math.max(20, Math.min(availW / cols, availH / rows));
  const boardW = cellSize * cols;
  const boardH = cellSize * rows;
  const originX = padLeft + Math.max(0, (availW - boardW) / 2);
  const originY = padTop + Math.max(0, (availH - boardH) / 2);

  // Territory shading: split by column midpoint. Cells left of the midpoint
  // are Player A's home; right are Player B's; the middle column is neutral.
  const midCol = (cols - 1) / 2;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = originX + c * cellSize;
      const y = originY + r * cellSize;
      if (c < midCol) ctx.fillStyle = P0_TERRITORY;
      else if (c > midCol) ctx.fillStyle = P1_TERRITORY;
      else ctx.fillStyle = 'transparent';
      if (ctx.fillStyle !== 'transparent') ctx.fillRect(x, y, cellSize, cellSize);
    }
  }

  // Grid lines (sketched dashed).
  ctx.strokeStyle = SOFT_INK;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  for (let i = 0; i <= cols; i++) {
    ctx.beginPath();
    ctx.moveTo(originX + i * cellSize, originY);
    ctx.lineTo(originX + i * cellSize, originY + boardH);
    ctx.stroke();
  }
  for (let i = 0; i <= rows; i++) {
    ctx.beginPath();
    ctx.moveTo(originX, originY + i * cellSize);
    ctx.lineTo(originX + boardW, originY + i * cellSize);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Obstacles.
  for (const [r, c] of obs.obstacles ?? []) {
    const x = originX + c * cellSize;
    const y = originY + r * cellSize;
    ctx.fillStyle = OBSTACLE_COLOR;
    ctx.fillRect(x + cellSize * 0.12, y + cellSize * 0.12, cellSize * 0.76, cellSize * 0.76);
  }

  // Base squares: player-colored tint + outline. Label is drawn later on top
  // of units/flags so it stays readable.
  const drawBaseCell = (base: [number, number] | undefined, color: string, tint: string) => {
    if (!base) return;
    const [r, c] = base;
    const x = originX + c * cellSize;
    const y = originY + r * cellSize;
    ctx.fillStyle = tint;
    ctx.fillRect(x, y, cellSize, cellSize);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x + 2, y + 2, cellSize - 4, cellSize - 4);
  };
  drawBaseCell(obs.a_base, P0_COLOR, 'rgba(31, 79, 139, 0.16)');
  drawBaseCell(obs.b_base, P1_COLOR, 'rgba(154, 51, 36, 0.16)');

  // Loose flags (only draw when nobody is carrying them).
  const cellCenter = (r: number, c: number) => ({
    x: originX + (c + 0.5) * cellSize,
    y: originY + (r + 0.5) * cellSize,
  });
  if (obs.carrier_a === null && obs.flag_a_pos) {
    const { x, y } = cellCenter(obs.flag_a_pos[0], obs.flag_a_pos[1]);
    drawFlag(ctx, x - cellSize * 0.15, y, cellSize * 0.75, P0_COLOR);
  }
  if (obs.carrier_b === null && obs.flag_b_pos) {
    const { x, y } = cellCenter(obs.flag_b_pos[0], obs.flag_b_pos[1]);
    drawFlag(ctx, x - cellSize * 0.15, y, cellSize * 0.75, P1_COLOR);
  }

  // Last-move arrows: compare current positions with previous step.
  if (prevObs) {
    if (prevObs.a_pos && obs.a_pos && (prevObs.a_pos[0] !== obs.a_pos[0] || prevObs.a_pos[1] !== obs.a_pos[1])) {
      const from = cellCenter(prevObs.a_pos[0], prevObs.a_pos[1]);
      const to = cellCenter(obs.a_pos[0], obs.a_pos[1]);
      drawArrow(ctx, from.x, from.y, to.x, to.y, P0_COLOR);
    }
    if (prevObs.b_pos && obs.b_pos && (prevObs.b_pos[0] !== obs.b_pos[0] || prevObs.b_pos[1] !== obs.b_pos[1])) {
      const from = cellCenter(prevObs.b_pos[0], prevObs.b_pos[1]);
      const to = cellCenter(obs.b_pos[0], obs.b_pos[1]);
      drawArrow(ctx, from.x, from.y, to.x, to.y, P1_COLOR);
    }
  }

  // Player units.
  if (obs.a_pos) {
    const { x, y } = cellCenter(obs.a_pos[0], obs.a_pos[1]);
    // Player A carries B's flag when carrier_b === 0.
    const carrying = obs.carrier_b === 0 ? P1_COLOR : null;
    drawUnit(ctx, x, y, cellSize, P0_COLOR, 'A', carrying);
  }
  if (obs.b_pos) {
    const { x, y } = cellCenter(obs.b_pos[0], obs.b_pos[1]);
    const carrying = obs.carrier_a === 1 ? P0_COLOR : null;
    drawUnit(ctx, x, y, cellSize, P1_COLOR, 'B', carrying);
  }

  // Coordinate labels around the board.
  ctx.font = `${Math.round(cellSize * 0.26)}px 'Inter', sans-serif`;
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.textBaseline = 'top';
  ctx.textAlign = 'center';
  for (let c = 0; c < cols; c++) {
    ctx.fillText(String(c), originX + (c + 0.5) * cellSize, originY + boardH + 4);
  }
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (let r = 0; r < rows; r++) {
    ctx.fillText(String(r), originX - 6, originY + (r + 0.5) * cellSize);
  }
}

export function renderer(options: RendererOptions<CtfStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as CtfStep[];
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
  const obs = currentStep?.boardState ?? null;
  if (!obs) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const prevObs = step > 0 ? (steps[step - 1]?.boardState ?? null) : null;

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const isTerminal = !!currentStep.isTerminal || !!obs.is_terminal;
  const forfeitReason = currentStep.forfeitReason;
  const forfeiterIdx = currentStep.players.findIndex((p) => p.forfeited);

  // Both players act on every non-terminal simultaneous step; highlight both.
  const bothActive = !isTerminal;

  header.innerHTML = `
    <span class="player p0 sketched-border ${bothActive ? 'active' : ''}" style="color: ${P0_COLOR};">
      <span class="glyph"></span>${escapeHtml(playerNames[0])}
    </span>
    <span class="vs">vs</span>
    <span class="player p1 sketched-border ${bothActive ? 'active' : ''}" style="color: ${P1_COLOR};">
      <span class="glyph"></span>${escapeHtml(playerNames[1])}
    </span>
  `;

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
    drawBoard(ctx, cssW, cssH, obs, prevObs);
  };
  requestAnimationFrame(sizeAndDraw);

  // --- Status line ---
  let winnerLabel = '';
  if (isTerminal) {
    if (obs.winner === 0) winnerLabel = `${playerNames[0]} wins!`;
    else if (obs.winner === 1) winnerLabel = `${playerNames[1]} wins!`;
    else if (obs.winner === 'draw') winnerLabel = 'Draw';
    else if (forfeitReason && forfeiterIdx >= 0) {
      const winnerIdx = 1 - forfeiterIdx;
      winnerLabel = `${playerNames[winnerIdx]} wins!`;
    } else {
      winnerLabel = currentStep.winner ?? 'Game over';
    }
  }

  const lastA = currentStep.players[0]?.actionDisplayText;
  const lastB = currentStep.players[1]?.actionDisplayText;
  const moveNum = obs.move_number ?? 0;

  let statusHTML = '';
  if (isTerminal) {
    const winColor =
      obs.winner === 0 || forfeiterIdx === 1 ? P0_COLOR : obs.winner === 1 || forfeiterIdx === 0 ? P1_COLOR : SOFT_INK;
    statusHTML += `<div class="status-line"><span style="color: ${winColor};">${escapeHtml(winnerLabel)}</span></div>`;
    if (forfeitReason) {
      statusHTML += `<div class="status-line"><span class="annotation forfeit-reason">${escapeHtml(forfeitReason)}</span></div>`;
    }
  } else {
    statusHTML += `<div class="status-line"><span>Turn ${moveNum}</span></div>`;
    const parts: string[] = [];
    if (lastA) parts.push(`<span style="color: ${P0_COLOR}; font-weight: 600;">A: ${escapeHtml(lastA)}</span>`);
    if (lastB) parts.push(`<span style="color: ${P1_COLOR}; font-weight: 600;">B: ${escapeHtml(lastB)}</span>`);
    if (parts.length) {
      statusHTML += `<div class="status-line"><span class="annotation">last moves:</span>${parts.join('<span class="annotation">/</span>')}</div>`;
    }
  }
  statusContainer.innerHTML = statusHTML;
}
