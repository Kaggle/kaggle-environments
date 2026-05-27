import type { RendererOptions } from '@kaggle-environments/core';
import type { BackgammonBoardState, BackgammonPoint, BackgammonStep } from './transformers/backgammonTransformer';

const INK = '#050001';
const SOFT_INK = '#3c3b37';
const SECONDARY_TEXT = '#444343';
const P0_COLOR = '#1f4f8b'; // Player X (0) -- blue
const P1_COLOR = '#9a3324'; // Player O (1) -- red
const POINT_LIGHT = '#fbf7e8';
const POINT_DARK = '#c4a66a';
const BAR_FILL = '#d9c89a';

const NUM_POINTS = 24;
const COLS_PER_HALF = 6;
const TOTAL_COLS = COLS_PER_HALF * 2; // 12 columns of points

// OpenSpiel point index for a given (row, col) on the rendered board.
//   row = 0 -> top row, point indices 12..23 (left to right)
//   row = 1 -> bottom row, point indices 11..0 (left to right)
function pointForCell(row: 0 | 1, col: number): number {
  if (row === 0) return 12 + col;
  return 11 - col;
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player 1' : 'Player 2';
}

function playerToIndex(p: 'x' | 'o' | string | undefined): 0 | 1 | -1 {
  if (p === 'x') return 0;
  if (p === 'o') return 1;
  return -1;
}

function colorFor(idx: 0 | 1): string {
  return idx === 0 ? P0_COLOR : P1_COLOR;
}

// Compute the set of points whose contents changed between two boards.  Used
// to draw "from" (ghost) and "to" (ring) highlights for the most recent move.
interface MoveHighlights {
  decreased: Set<number>;
  increased: Set<number>;
}

function diffBoards(prev: BackgammonBoardState | null, curr: BackgammonBoardState | null): MoveHighlights {
  const decreased = new Set<number>();
  const increased = new Set<number>();
  if (!prev || !curr) return { decreased, increased };
  for (let i = 0; i < NUM_POINTS; i++) {
    const a = prev.board?.[i] ?? null;
    const b = curr.board?.[i] ?? null;
    const ac = a?.count ?? 0;
    const bc = b?.count ?? 0;
    if (!a && !b) continue;
    if (!a && b) {
      increased.add(i);
    } else if (a && !b) {
      decreased.add(i);
    } else if (a && b) {
      if (a.player !== b.player) {
        // hit: previous player lost, new player gained
        decreased.add(i);
        increased.add(i);
      } else if (bc > ac) {
        increased.add(i);
      } else if (bc < ac) {
        decreased.add(i);
      }
    }
  }
  return { decreased, increased };
}

interface BoardGeom {
  originX: number;
  originY: number;
  boardW: number;
  boardH: number;
  pointW: number;
  pointH: number;
  barW: number;
  trayW: number;
  barX: number;
  trayX: number;
}

function layout(width: number, height: number): BoardGeom {
  // Reserve a small margin around the playable region.
  const margin = 12;
  const labelPad = 18; // room for point labels above/below

  const availW = Math.max(60, width - margin * 2);
  const availH = Math.max(60, height - margin * 2 - labelPad * 2);

  // Width = 12 point cols + bar (≈ 1 col) + bear-off tray (≈ 1 col).
  // We'll allocate: pointW per col, barW = pointW, trayW = pointW * 1.0.
  const unitW = availW / (TOTAL_COLS + 2);
  const pointW = unitW;
  const barW = unitW;
  const trayW = unitW;

  const boardW = pointW * TOTAL_COLS + barW + trayW;
  // Triangle (point) height: roughly 45% of the board height per half.
  const pointH = Math.min(availH * 0.45, pointW * 4.5);
  const boardH = pointH * 2 + 8; // 8px gap between halves

  const originX = (width - boardW) / 2;
  const originY = (height - boardH) / 2;

  const barX = originX + pointW * COLS_PER_HALF;
  const trayX = originX + pointW * TOTAL_COLS + barW;

  return { originX, originY, boardW, boardH, pointW, pointH, barW, trayW, barX, trayX };
}

// X coordinate of the left edge of a point column.
function pointLeft(geom: BoardGeom, col: number): number {
  // After the first 6 columns, skip past the bar.
  const offset = col < COLS_PER_HALF ? 0 : geom.barW;
  return geom.originX + col * geom.pointW + offset;
}

function drawFrame(ctx: CanvasRenderingContext2D, geom: BoardGeom) {
  // Outer board background.
  ctx.fillStyle = '#efe6c8';
  ctx.fillRect(geom.originX, geom.originY, geom.boardW, geom.boardH);

  // Bar.
  ctx.fillStyle = BAR_FILL;
  ctx.fillRect(geom.barX, geom.originY, geom.barW, geom.boardH);

  // Bear-off tray.
  ctx.fillStyle = '#e8dcb2';
  ctx.fillRect(geom.trayX, geom.originY, geom.trayW, geom.boardH);

  // Outer sketched border.
  ctx.strokeStyle = SOFT_INK;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(geom.originX, geom.originY, geom.boardW, geom.boardH);
  ctx.strokeRect(geom.barX, geom.originY, geom.barW, geom.boardH);
  ctx.strokeRect(geom.trayX, geom.originY, geom.trayW, geom.boardH);
  ctx.setLineDash([]);
}

function drawPoint(ctx: CanvasRenderingContext2D, geom: BoardGeom, row: 0 | 1, col: number, light: boolean) {
  const x = pointLeft(geom, col);
  const w = geom.pointW;
  const h = geom.pointH;
  const cx = x + w / 2;

  ctx.beginPath();
  if (row === 0) {
    // Triangle pointing down from top edge.
    const yTop = geom.originY;
    const yTip = geom.originY + h;
    ctx.moveTo(x, yTop);
    ctx.lineTo(x + w, yTop);
    ctx.lineTo(cx, yTip);
  } else {
    // Triangle pointing up from bottom edge.
    const yBot = geom.originY + geom.boardH;
    const yTip = geom.originY + geom.boardH - h;
    ctx.moveTo(x, yBot);
    ctx.lineTo(x + w, yBot);
    ctx.lineTo(cx, yTip);
  }
  ctx.closePath();
  ctx.fillStyle = light ? POINT_LIGHT : POINT_DARK;
  ctx.fill();
  ctx.strokeStyle = SOFT_INK;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawCheckers(
  ctx: CanvasRenderingContext2D,
  geom: BoardGeom,
  row: 0 | 1,
  col: number,
  point: BackgammonPoint | null
) {
  if (!point) return;
  const x = pointLeft(geom, col);
  const cx = x + geom.pointW / 2;
  const radius = Math.max(4, geom.pointW * 0.42);
  const stride = radius * 2 * 0.95;

  // Max stack of distinct checkers we'll draw before collapsing to a label.
  const maxVisible = Math.min(point.count, 5);
  const playerIdx = point.player === 'x' ? 0 : 1;
  const color = colorFor(playerIdx);
  const fill = playerIdx === 0 ? '#f5f1e2' : INK;
  const textColor = playerIdx === 0 ? color : '#f5f1e2';

  for (let i = 0; i < maxVisible; i++) {
    const cy = row === 0 ? geom.originY + radius + i * stride : geom.originY + geom.boardH - radius - i * stride;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = color;
    ctx.stroke();
  }

  if (point.count > 5) {
    // Label the topmost (innermost) checker with the total count.
    const cy =
      row === 0
        ? geom.originY + radius + (maxVisible - 1) * stride
        : geom.originY + geom.boardH - radius - (maxVisible - 1) * stride;
    ctx.font = `700 ${Math.round(radius * 1.1)}px 'Inter', sans-serif`;
    ctx.fillStyle = textColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(point.count), cx, cy + 1);
  }
}

function drawHighlight(ctx: CanvasRenderingContext2D, geom: BoardGeom, row: 0 | 1, col: number, kind: 'from' | 'to') {
  const x = pointLeft(geom, col);
  const w = geom.pointW;
  const h = geom.pointH;
  const cx = x + w / 2;
  ctx.beginPath();
  if (row === 0) {
    ctx.moveTo(x, geom.originY);
    ctx.lineTo(x + w, geom.originY);
    ctx.lineTo(cx, geom.originY + h);
  } else {
    ctx.moveTo(x, geom.originY + geom.boardH);
    ctx.lineTo(x + w, geom.originY + geom.boardH);
    ctx.lineTo(cx, geom.originY + geom.boardH - h);
  }
  ctx.closePath();
  ctx.strokeStyle = kind === 'to' ? '#1f6b1f' : 'rgba(60, 59, 55, 0.55)';
  ctx.lineWidth = kind === 'to' ? 2.5 : 1.5;
  ctx.setLineDash(kind === 'from' ? [4, 3] : []);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawBar(ctx: CanvasRenderingContext2D, geom: BoardGeom, bar: { x: number; o: number }) {
  const cx = geom.barX + geom.barW / 2;
  const radius = Math.max(4, geom.barW * 0.32);
  const stride = radius * 2 * 0.95;

  // X's bar checkers stack from the BOTTOM half upward.
  for (let i = 0; i < bar.x; i++) {
    const cy = geom.originY + geom.boardH - radius - i * stride;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#f5f1e2';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = P0_COLOR;
    ctx.stroke();
  }

  // O's bar checkers stack from the TOP half downward.
  for (let i = 0; i < bar.o; i++) {
    const cy = geom.originY + radius + i * stride;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = INK;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = P1_COLOR;
    ctx.stroke();
  }
}

function drawTray(ctx: CanvasRenderingContext2D, geom: BoardGeom, off: { x: number; o: number }) {
  // X borne-off: top tray slab.  O borne-off: bottom tray slab.
  const slabH = geom.boardH / 2 - 4;
  const slabW = geom.trayW - 6;
  const slabX = geom.trayX + 3;

  const drawSlab = (yTop: number, count: number, idx: 0 | 1) => {
    const color = colorFor(idx);
    // Background fill, slightly inset.
    ctx.fillStyle = idx === 0 ? '#f5f1e2' : INK;
    ctx.fillRect(slabX, yTop, slabW, slabH);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(slabX, yTop, slabW, slabH);
    ctx.setLineDash([]);
    // Count label.
    ctx.font = `700 ${Math.round(slabW * 0.4)}px 'Inter', sans-serif`;
    ctx.fillStyle = idx === 0 ? color : '#f5f1e2';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(count), slabX + slabW / 2, yTop + slabH / 2);
    // Tiny "off" annotation.
    ctx.font = `${Math.round(slabW * 0.22)}px 'Inter', sans-serif`;
    ctx.fillStyle = idx === 0 ? SECONDARY_TEXT : 'rgba(245, 241, 226, 0.7)';
    ctx.fillText('off', slabX + slabW / 2, yTop + slabH - slabW * 0.22);
  };

  drawSlab(geom.originY + 2, off.x, 0);
  drawSlab(geom.originY + slabH + 6, off.o, 1);
}

function drawLabels(ctx: CanvasRenderingContext2D, geom: BoardGeom) {
  ctx.font = `${Math.max(10, Math.round(geom.pointW * 0.32))}px 'Inter', sans-serif`;
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.textAlign = 'center';

  // Top labels above each point (point indices 12..23 left→right).
  ctx.textBaseline = 'bottom';
  for (let col = 0; col < TOTAL_COLS; col++) {
    const idx = pointForCell(0, col);
    const x = pointLeft(geom, col) + geom.pointW / 2;
    ctx.fillText(String(idx), x, geom.originY - 4);
  }

  // Bottom labels below each point (point indices 11..0 left→right).
  ctx.textBaseline = 'top';
  for (let col = 0; col < TOTAL_COLS; col++) {
    const idx = pointForCell(1, col);
    const x = pointLeft(geom, col) + geom.pointW / 2;
    ctx.fillText(String(idx), x, geom.originY + geom.boardH + 4);
  }
}

function draw(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: BackgammonBoardState,
  highlights: MoveHighlights
) {
  ctx.clearRect(0, 0, width, height);
  const geom = layout(width, height);

  drawFrame(ctx, geom);

  // Point triangles.  Alternate light/dark, offset between rows so colors
  // line up vertically the way a real board does.
  for (let row of [0, 1] as const) {
    for (let col = 0; col < TOTAL_COLS; col++) {
      const light = (col + row) % 2 === 0;
      drawPoint(ctx, geom, row, col, light);
    }
  }

  // Move-from highlights (drawn over empty triangles).
  for (const idx of highlights.decreased) {
    const row: 0 | 1 = idx >= 12 ? 0 : 1;
    const col = row === 0 ? idx - 12 : 11 - idx;
    drawHighlight(ctx, geom, row, col, 'from');
  }

  // Checkers per point.
  for (let row of [0, 1] as const) {
    for (let col = 0; col < TOTAL_COLS; col++) {
      const idx = pointForCell(row, col);
      drawCheckers(ctx, geom, row, col, obs.board?.[idx] ?? null);
    }
  }

  // Move-to highlights drawn over the checkers so the ring stays visible.
  for (const idx of highlights.increased) {
    const row: 0 | 1 = idx >= 12 ? 0 : 1;
    const col = row === 0 ? idx - 12 : 11 - idx;
    drawHighlight(ctx, geom, row, col, 'to');
  }

  drawBar(ctx, geom, obs.bar ?? { x: 0, o: 0 });
  drawTray(ctx, geom, obs.off ?? { x: 0, o: 0 });
  drawLabels(ctx, geom);
}

export function renderer(options: RendererOptions<BackgammonStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as BackgammonStep[];
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
  const obs: BackgammonBoardState | null = currentStep?.boardState ?? null;
  if (!obs) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const prevObs: BackgammonBoardState | null = step > 0 ? (steps[step - 1]?.boardState ?? null) : null;
  const highlights = diffBoards(prevObs, obs);

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const isTerminal = obs.is_terminal;
  const activeIdx = isTerminal ? -1 : playerToIndex(obs.current_player as any);

  header.innerHTML = `
    <span class="player p0 sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${P0_COLOR};">
      <span class="glyph"></span>${playerNames[0]} <span style="opacity:0.7;">(x)</span>
    </span>
    <span class="vs">vs</span>
    <span class="player p1 sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${P1_COLOR};">
      <span class="glyph"></span>${playerNames[1]} <span style="opacity:0.7;">(o)</span>
    </span>
  `;

  const sizeAndDraw = () => {
    const r = wrap.getBoundingClientRect();
    const cssW = Math.max(1, Math.floor(r.width));
    const cssH = Math.max(1, Math.floor(r.height));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = cssW;
    canvas.height = cssH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    draw(ctx, cssW, cssH, obs, highlights);
  };
  requestAnimationFrame(sizeAndDraw);

  // Status bar.
  let statusHTML = '';
  if (isTerminal) {
    if (obs.winner === 'x') {
      statusHTML = `<p style="margin: 0;"><span style="color: ${P0_COLOR};">${playerNames[0]} (x) Wins!</span></p>`;
    } else if (obs.winner === 'o') {
      statusHTML = `<p style="margin: 0;"><span style="color: ${P1_COLOR};">${playerNames[1]} (o) Wins!</span></p>`;
    } else {
      statusHTML = `<p style="margin: 0;">Draw</p>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? P0_COLOR : P1_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;
    if (obs.dice && obs.dice.length) {
      const diceHTML = obs.dice.map((d) => `<span class="die ${d.used ? 'used' : ''}">${d.value}</span>`).join(' ');
      statusHTML += `<span class="annotation">dice:</span> ${diceHTML}`;
    }
  }
  statusHTML += `<span class="annotation">move ${obs.move_number}</span>`;
  if (obs.bar.x + obs.bar.o > 0) {
    statusHTML += `<span class="annotation">bar:</span>`;
    if (obs.bar.x > 0) statusHTML += `<span style="color:${P0_COLOR}; font-weight:700;">x×${obs.bar.x}</span>`;
    if (obs.bar.o > 0) statusHTML += `<span style="color:${P1_COLOR}; font-weight:700;">o×${obs.bar.o}</span>`;
  }
  statusContainer.innerHTML = statusHTML;
}
