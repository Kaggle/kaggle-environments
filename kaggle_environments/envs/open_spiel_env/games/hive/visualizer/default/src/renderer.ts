import type { RendererOptions } from '@kaggle-environments/core';
import type { HiveBoardState, HivePieces, HiveStep } from './transformers/hiveTransformer';

const WHITE_FILL = '#fbf3d8';
const WHITE_STROKE = '#7a5b00';
const BLACK_FILL = '#2c2a26';
const BLACK_STROKE = '#000000';
const WHITE_LABEL = '#1d1300';
const BLACK_LABEL = '#fff7d8';

const HIGHLIGHT_LAST_MOVE = '#ff9416';
const HIGHLIGHT_SURROUNDED = '#e63946';

const SECONDARY_TEXT = '#444343';
const SKETCH_STROKE = '#3c3b37';
const PAGE_BG = '#f5f1e2';

const SQRT3 = Math.sqrt(3);

// Bug-type colours; used to tint the letter inside each tile so the bug type
// is easy to recognise at a glance.
const BUG_COLOR: Record<string, string> = {
  Q: '#d4a017', // Queen - gold
  A: '#2e6cb6', // Ant - blue
  G: '#3b8c3b', // Grasshopper - green
  S: '#7a4a1f', // Spider - brown
  B: '#7a3fa8', // Beetle - purple
  M: '#8d8d8d', // Mosquito - grey
  L: '#c1352b', // Ladybug - red
  P: '#2fa3a3', // Pillbug - teal
};

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'White' : 'Black';
}

// "wA1" -> { color: 'w', bug: 'A', ordinal: '1' }
function parseTile(tile: string): { color: 'w' | 'b'; bug: string; ordinal: string } {
  const color = tile[0] === 'b' ? 'b' : 'w';
  const bug = tile[1] ?? '';
  const ordinal = tile.slice(2);
  return { color, bug, ordinal };
}

// Axial (q, r) -> pixel (pointy-top hex).
function axialToPixel(q: number, r: number, size: number): { x: number; y: number } {
  return {
    x: size * SQRT3 * (q + r / 2),
    y: size * 1.5 * r,
  };
}

function drawHexPath(ctx: CanvasRenderingContext2D, cx: number, cy: number, s: number) {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 2;
    const px = cx + s * Math.cos(angle);
    const py = cy + s * Math.sin(angle);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

// "bA1 wA1/" -> "bA1"; "wG2" -> "wG2"; "pass" -> null
function fromTileOfMove(move: string | null | undefined): string | null {
  if (!move || move === 'pass') return null;
  return move.split(/\s+/)[0] ?? null;
}

// Detect whether a queen is fully surrounded (game-ending condition for that
// player). A queen counts as surrounded when all 6 of its neighbouring (q, r)
// coordinates contain at least one piece.
function isQueenSurrounded(pieces: HivePieces, color: 'w' | 'b'): boolean {
  const queen = pieces[`${color}Q`];
  if (!queen) return false;
  const [q, r] = queen;
  const neighbours: Array<[number, number]> = [
    [q + 1, r - 1],
    [q + 1, r],
    [q, r + 1],
    [q - 1, r + 1],
    [q - 1, r],
    [q, r - 1],
  ];
  // Build a set of occupied (q, r) ground positions for quick lookup.
  const occupied = new Set<string>();
  for (const [, [pq, pr]] of Object.entries(pieces)) {
    occupied.add(`${pq},${pr}`);
  }
  return neighbours.every(([nq, nr]) => occupied.has(`${nq},${nr}`));
}

function drawHiveBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  pieces: HivePieces,
  highlightTile: string | null
) {
  ctx.clearRect(0, 0, width, height);

  const entries = Object.entries(pieces);

  // Bounding box of all played pieces (in axial coords).
  let minQ = 0;
  let maxQ = 0;
  let minR = 0;
  let maxR = 0;
  if (entries.length === 0) {
    // Empty board placeholder; show a small area around origin.
    minQ = -3;
    maxQ = 3;
    minR = -3;
    maxR = 3;
  } else {
    for (const [, [q, r]] of entries) {
      if (q < minQ) minQ = q;
      if (q > maxQ) maxQ = q;
      if (r < minR) minR = r;
      if (r > maxR) maxR = r;
    }
    // Add 1 hex of margin so the outer pieces don't touch the canvas edge.
    minQ -= 1;
    maxQ += 1;
    minR -= 1;
    maxR += 1;
  }

  // Compute the pixel bounding box at size=1, then derive the hex size that
  // fits the canvas.
  const xs: number[] = [];
  const ys: number[] = [];
  for (let q = minQ; q <= maxQ; q++) {
    for (let r = minR; r <= maxR; r++) {
      const { x, y } = axialToPixel(q, r, 1);
      xs.push(x);
      ys.push(y);
    }
  }
  const minPx = Math.min(...xs) - SQRT3;
  const maxPx = Math.max(...xs) + SQRT3;
  const minPy = Math.min(...ys) - 1;
  const maxPy = Math.max(...ys) + 1;
  const boardPxW = maxPx - minPx;
  const boardPxH = maxPy - minPy;

  const padding = 16;
  const innerW = Math.max(0, width - padding * 2);
  const innerH = Math.max(0, height - padding * 2);
  const sizeFromW = innerW / boardPxW;
  const sizeFromH = innerH / boardPxH;
  const size = Math.max(8, Math.min(sizeFromW, sizeFromH, 64));

  // Centre the board.
  const drawnW = boardPxW * size;
  const drawnH = boardPxH * size;
  const xOffset = (width - drawnW) / 2 - minPx * size;
  const yOffset = (height - drawnH) / 2 - minPy * size;

  const centerOf = (q: number, r: number) => {
    const p = axialToPixel(q, r, size);
    return { x: p.x + xOffset, y: p.y + yOffset };
  };

  // Draw light "ghost" cells for the bounding area so it's clear what space
  // is on the board.
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 3]);
  ctx.strokeStyle = '#d6cfb1';
  for (let q = minQ; q <= maxQ; q++) {
    for (let r = minR; r <= maxR; r++) {
      const { x, y } = centerOf(q, r);
      drawHexPath(ctx, x, y, size * 0.94);
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);

  // Bucket pieces by (q, r) so we can render stacks bottom-up.
  const stacks = new Map<string, Array<{ tile: string; h: number }>>();
  for (const [tile, [q, r, h]] of entries) {
    const key = `${q},${r}`;
    if (!stacks.has(key)) stacks.set(key, []);
    stacks.get(key)!.push({ tile, h });
  }
  for (const stack of stacks.values()) {
    stack.sort((a, b) => a.h - b.h);
  }

  const fontSize = Math.round(size * 0.55);
  const ordinalFontSize = Math.round(size * 0.3);

  for (const [key, stack] of stacks.entries()) {
    const [qStr, rStr] = key.split(',');
    const q = parseInt(qStr, 10);
    const r = parseInt(rStr, 10);
    const { x, y } = centerOf(q, r);

    stack.forEach(({ tile }, idx) => {
      // Render each layer with a slight NE offset so the stack reads visibly.
      const offset = idx * size * 0.16;
      const cx = x + offset;
      const cy = y - offset;
      const { color, bug, ordinal } = parseTile(tile);

      drawHexPath(ctx, cx, cy, size * 0.94);
      ctx.fillStyle = color === 'w' ? WHITE_FILL : BLACK_FILL;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = color === 'w' ? WHITE_STROKE : BLACK_STROKE;
      ctx.stroke();

      // Bug-type letter.
      ctx.font = `700 ${fontSize}px 'Inter', sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = BUG_COLOR[bug] ?? (color === 'w' ? WHITE_LABEL : BLACK_LABEL);
      ctx.fillText(bug, cx, cy + size * 0.02);

      // Ordinal (e.g. the "2" in "wA2") tucked in the bottom-right corner.
      if (ordinal) {
        ctx.font = `600 ${ordinalFontSize}px 'Inter', sans-serif`;
        ctx.fillStyle = color === 'w' ? SECONDARY_TEXT : BLACK_LABEL;
        ctx.fillText(ordinal, cx + size * 0.35, cy + size * 0.42);
      }

      // Last-move highlight ring around the top tile only.
      if (highlightTile === tile && idx === stack.length - 1) {
        drawHexPath(ctx, cx, cy, size * 0.94);
        ctx.lineWidth = Math.max(2.5, size * 0.12);
        ctx.strokeStyle = HIGHLIGHT_LAST_MOVE;
        ctx.stroke();
      }
    });
  }

  // Surrounded-queen highlight. Once a queen is surrounded the game ends, so
  // call this out with a strong red ring.
  for (const color of ['w', 'b'] as const) {
    const queenTile = `${color}Q`;
    const queen = pieces[queenTile];
    if (!queen) continue;
    if (!isQueenSurrounded(pieces, color)) continue;
    const [q, r] = queen;
    const { x, y } = centerOf(q, r);
    drawHexPath(ctx, x, y, size * 0.94);
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = Math.max(2.5, size * 0.16);
    ctx.strokeStyle = HIGHLIGHT_SURROUNDED;
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

export function renderer(options: RendererOptions<HiveStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as HiveStep[];
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

  // Use the parchment colour for the page so the canvas blends with the page.
  parent.style.backgroundColor = PAGE_BG;

  const currentStep = steps[step];
  const board: HiveBoardState | null = currentStep?.boardState ?? null;
  if (!board) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const isTerminal = board.is_terminal;
  const activeIdx = isTerminal ? -1 : board.current_player === 'white' ? 0 : board.current_player === 'black' ? 1 : -1;

  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${WHITE_STROKE};">
      ${playerNames[0]} <span style="opacity: 0.7;">(W)</span>
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${BLACK_STROKE};">
      ${playerNames[1]} <span style="opacity: 0.7;">(B)</span>
    </span>
  `;

  const highlightTile = fromTileOfMove(board.last_move);

  const sizeAndDraw = () => {
    const wrapRect = wrap.getBoundingClientRect();
    const cssW = Math.max(1, Math.floor(wrapRect.width));
    const cssH = Math.max(1, Math.floor(wrapRect.height));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = cssW;
    canvas.height = cssH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawHiveBoard(ctx, cssW, cssH, board.pieces, highlightTile);
  };

  requestAnimationFrame(sizeAndDraw);

  // Status row: turn / winner / last move / move count.
  let statusHTML = '';
  if (isTerminal) {
    if (board.winner === 'white') {
      statusHTML = `<span style="color: ${WHITE_STROKE};">${playerNames[0]} (W) wins!</span>`;
    } else if (board.winner === 'black') {
      statusHTML = `<span style="color: ${BLACK_STROKE};">${playerNames[1]} (B) wins!</span>`;
    } else {
      statusHTML = `<span>Draw</span>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? WHITE_STROKE : BLACK_STROKE;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;
  }
  if (board.last_move) {
    const lastActor = board.move_number > 0 ? (board.move_number - 1) % 2 : null;
    const moverColor = lastActor === 0 ? WHITE_STROKE : BLACK_STROKE;
    statusHTML += `<span class="annotation">last: <span style="color: ${moverColor}; font-weight: 600;">${board.last_move}</span></span>`;
  }
  statusHTML += `<span class="annotation">move ${board.move_number}</span>`;
  statusContainer.innerHTML = statusHTML;

  // Hide the unused dashed-border outline lint warning.
  void SKETCH_STROKE;
}
