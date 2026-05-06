import type { RendererOptions } from '@kaggle-environments/core';
import type { OshiZumoBoardState, OshiZumoStep } from './transformers/oshiZumoTransformer';

const INK = '#050001';
const SOFT_INK = '#3c3b37';
const P0_COLOR = '#1f4f8b';
const P1_COLOR = '#9a3324';
const WRESTLER_FILL = '#f5f1e2';
const WRESTLER_STROKE = '#050001';
const TERRITORY_P0 = 'rgba(31, 79, 139, 0.10)';
const TERRITORY_P1 = 'rgba(154, 51, 36, 0.10)';
const PREV_GHOST = 'rgba(60, 59, 55, 0.35)';
const ARROW_COLOR = '#050001';

function makePlayerCard(
  label: string,
  pclass: 'p0' | 'p1',
  coins: number,
  delta: number | null,
  isActive: boolean
): string {
  const deltaHtml = delta !== null && delta < 0 ? `<span class="oshi-coin-delta">${delta}</span>` : '';
  return `
    <div class="oshi-player-card sketched-border ${pclass} ${isActive ? 'active' : ''}">
      <div class="oshi-player-name">
        <span class="oshi-glyph"></span>${label}
      </div>
      <div class="oshi-coins">
        ${coins}<span style="opacity:0.7;font-weight:400;">coins</span>${deltaHtml}
      </div>
    </div>
  `;
}

function findPrevBoard(steps: OshiZumoStep[], step: number): OshiZumoBoardState | null {
  for (let i = step - 1; i >= 0; i--) {
    if (steps[i]?.boardState) return steps[i].boardState;
  }
  return null;
}

function drawField(
  c: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: OshiZumoBoardState,
  prev: OshiZumoBoardState | null
) {
  c.clearRect(0, 0, width, height);

  const fieldLen = obs.field_size;
  const marginX = Math.max(28, width * 0.06);
  const marginY = Math.max(40, height * 0.18);
  const usableW = width - marginX * 2;
  const usableH = height - marginY * 2;
  const cellSize = Math.min(usableW / fieldLen, usableH * 0.6);
  const fieldW = cellSize * fieldLen;
  const originX = (width - fieldW) / 2;
  const originY = (height - cellSize) / 2;

  // Territory bands behind the cells.
  const center = (fieldLen - 1) / 2;
  for (let i = 0; i < fieldLen; i++) {
    const x = originX + i * cellSize;
    if (i === 0 || i === fieldLen - 1) continue; // boundary cells get their own treatment
    if (i < center) c.fillStyle = TERRITORY_P0;
    else if (i > center) c.fillStyle = TERRITORY_P1;
    else c.fillStyle = 'rgba(255, 255, 255, 0.4)';
    c.fillRect(x, originY, cellSize, cellSize);
  }

  // Boundary cells (the edges off which the wrestler is pushed).
  const boundaryFill = 'rgba(60, 59, 55, 0.18)';
  c.fillStyle = boundaryFill;
  c.fillRect(originX, originY, cellSize, cellSize);
  c.fillRect(originX + (fieldLen - 1) * cellSize, originY, cellSize, cellSize);

  // Sketched cell borders.
  c.strokeStyle = SOFT_INK;
  c.lineWidth = 1;
  c.setLineDash([3, 3]);
  for (let i = 0; i <= fieldLen; i++) {
    const x = originX + i * cellSize;
    c.beginPath();
    c.moveTo(x, originY);
    c.lineTo(x, originY + cellSize);
    c.stroke();
  }
  c.beginPath();
  c.moveTo(originX, originY);
  c.lineTo(originX + fieldW, originY);
  c.stroke();
  c.beginPath();
  c.moveTo(originX, originY + cellSize);
  c.lineTo(originX + fieldW, originY + cellSize);
  c.stroke();
  c.setLineDash([]);

  // Edge labels (player territory).
  c.font = `600 ${Math.round(cellSize * 0.32)}px 'Inter', sans-serif`;
  c.fillStyle = P0_COLOR;
  c.textAlign = 'center';
  c.textBaseline = 'alphabetic';
  c.fillText('Player 1 →', originX + fieldW * 0.25, originY - 10);
  c.fillStyle = P1_COLOR;
  c.fillText('← Player 2', originX + fieldW * 0.75, originY - 10);

  // "Push off" markers under the boundaries.
  c.font = `400 ${Math.round(cellSize * 0.28)}px 'Mynerve', cursive`;
  c.fillStyle = SOFT_INK;
  c.textBaseline = 'top';
  c.fillText('off edge', originX + cellSize / 2, originY + cellSize + 8);
  c.fillText('off edge', originX + (fieldLen - 0.5) * cellSize, originY + cellSize + 8);

  // Previous wrestler position (ghost).
  const cellCenterX = (idx: number) => originX + (idx + 0.5) * cellSize;
  const cellCenterY = originY + cellSize / 2;
  const radius = cellSize * 0.34;

  if (prev && prev.wrestler_position !== obs.wrestler_position) {
    const px = cellCenterX(prev.wrestler_position);
    c.beginPath();
    c.arc(px, cellCenterY, radius, 0, Math.PI * 2);
    c.fillStyle = 'rgba(245, 241, 226, 0.85)';
    c.fill();
    c.lineWidth = 1.5;
    c.setLineDash([4, 3]);
    c.strokeStyle = PREV_GHOST;
    c.stroke();
    c.setLineDash([]);

    // Arrow from prev to current.
    const cx = cellCenterX(obs.wrestler_position);
    drawArrow(c, px, cellCenterY - radius - 6, cx, cellCenterY - radius - 6);
  }

  // Current wrestler.
  const wx = cellCenterX(obs.wrestler_position);
  c.beginPath();
  c.arc(wx, cellCenterY, radius, 0, Math.PI * 2);
  c.fillStyle = WRESTLER_FILL;
  c.fill();
  c.lineWidth = 2;
  c.strokeStyle = WRESTLER_STROKE;
  c.stroke();

  // Wrestler glyph: "W"
  c.fillStyle = INK;
  c.font = `700 ${Math.round(cellSize * 0.42)}px 'Inter', sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.fillText('W', wx, cellCenterY + 1);
}

function drawArrow(c: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
  const head = 8;
  c.strokeStyle = ARROW_COLOR;
  c.fillStyle = ARROW_COLOR;
  c.lineWidth = 1.5;
  c.beginPath();
  c.moveTo(x1, y1);
  c.lineTo(x2, y2);
  c.stroke();
  const angle = Math.atan2(y2 - y1, x2 - x1);
  c.beginPath();
  c.moveTo(x2, y2);
  c.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
  c.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
  c.closePath();
  c.fill();
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = (replay.steps as unknown as OshiZumoStep[]) ?? [];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="oshi-header"></div>
      <canvas></canvas>
      <div class="oshi-status sketched-border"></div>
    </div>
  `;
  const root = parent.querySelector('.renderer-container') as HTMLDivElement;
  const header = parent.querySelector('.oshi-header') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const statusEl = parent.querySelector('.oshi-status') as HTMLDivElement;
  if (!canvas || !root) return;

  const safeStep = Math.max(0, Math.min(step, steps.length - 1));
  const current = steps[safeStep];
  const obs = current?.boardState ?? null;
  const prev = obs ? findPrevBoard(steps, safeStep) : null;

  // Header: player cards with coin counts and last-round bid as a delta.
  const teamA = current?.players?.[0]?.name ?? 'Player 1';
  const teamB = current?.players?.[1]?.name ?? 'Player 2';
  const coins = obs?.coins ?? [0, 0];
  const bids = current?.bids ?? [null, null];
  const deltas: [number | null, number | null] = [
    bids[0] !== null ? -bids[0]! : null,
    bids[1] !== null ? -bids[1]! : null,
  ];
  // Active = has coins left and game not terminal.
  const isTerm = !!obs?.is_terminal;
  const activeP0 = !isTerm && coins[0] > 0;
  const activeP1 = !isTerm && coins[1] > 0;

  header.innerHTML = `
    ${makePlayerCard(teamA, 'p0', coins[0], deltas[0], activeP0)}
    <span class="oshi-vs">vs</span>
    ${makePlayerCard(teamB, 'p1', coins[1], deltas[1], activeP1)}
  `;

  // Resize canvas to its actual flex slot.
  canvas.width = 0;
  canvas.height = 0;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width));
  canvas.height = Math.max(1, Math.floor(rect.height));

  const c = canvas.getContext('2d');
  if (c && obs) {
    drawField(c, canvas.width, canvas.height, obs, prev);
  }

  // Status line.
  if (!obs) {
    statusEl.textContent = 'Waiting for replay…';
  } else if (isTerm) {
    const winnerLine = current?.winner ?? 'Game over';
    const movesLine = `<span class="oshi-annotation">${obs.move_number} bids · field of ${obs.field_size}</span>`;
    statusEl.innerHTML = `<div>${winnerLine}</div>${movesLine}`;
  } else {
    const bidLine =
      bids[0] !== null && bids[1] !== null
        ? `<span class="oshi-annotation">last bids: ${teamA} ${bids[0]} · ${teamB} ${bids[1]}</span>`
        : `<span class="oshi-annotation">place your bids</span>`;
    statusEl.innerHTML = `<div>Round ${obs.move_number}</div>${bidLine}`;
  }
}
