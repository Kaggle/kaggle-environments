import type { RendererOptions } from '@kaggle-environments/core';

// Player colors match the light/dark contrast of white/black pieces while
// staying legible against the warm parchment background.
const PLAYER_W_COLOR = '#e9d59a'; // "white" piece fill (warm ivory)
const PLAYER_W_STROKE = '#8a6b1e';
const PLAYER_B_COLOR = '#2a2418'; // "black" piece fill
const PLAYER_B_STROKE = '#050001';
const HIGHLIGHT_ADD = '#2f8f3b'; // ring: piece placed / arrived here
const HIGHLIGHT_MOVE_FROM = '#3878c8'; // faded ring: piece moved from here
const HIGHLIGHT_CAPTURE = '#c93b3b'; // "X" overlay: piece captured here
const SKETCH_STROKE = '#3c3b37';
const BOARD_LINE = '#3c3b37';
const POINT_FILL = '#fbf7e8';
const MILL_COLOR = 'rgba(213, 158, 43, 0.55)';

// Board layout: each of the 24 points has a (col, row) coordinate in a 7x7
// grid (0..6). Mirrors the ASCII rendering in OpenSpiel's nine_mens_morris.cc,
// squashed to a unit grid.
const POINT_GRID: Array<[number, number]> = [
  [0, 0],
  [3, 0],
  [6, 0], // 0..2  (outer top)
  [1, 1],
  [3, 1],
  [5, 1], // 3..5  (middle top)
  [2, 2],
  [3, 2],
  [4, 2], // 6..8  (inner top)
  [0, 3],
  [1, 3],
  [2, 3], // 9..11 (left cross)
  [4, 3],
  [5, 3],
  [6, 3], // 12..14 (right cross)
  [2, 4],
  [3, 4],
  [4, 4], // 15..17 (inner bottom)
  [1, 5],
  [3, 5],
  [5, 5], // 18..20 (middle bottom)
  [0, 6],
  [3, 6],
  [6, 6], // 21..23 (outer bottom)
];

// Board segments (start, end) drawn as connecting lines. Together these form
// the three nested squares plus four midpoint crosses.
const BOARD_SEGMENTS: Array<[number, number]> = [
  // Outer square
  [0, 2],
  [2, 23],
  [23, 21],
  [21, 0],
  // Middle square
  [3, 5],
  [5, 20],
  [20, 18],
  [18, 3],
  // Inner square
  [6, 8],
  [8, 17],
  [17, 15],
  [15, 6],
  // Cross lines (midpoint to midpoint through each square edge)
  [1, 7],
  [9, 11],
  [12, 14],
  [16, 22],
];

// All 16 possible mills (three-in-a-row combinations).
const MILLS: Array<[number, number, number]> = [
  // Horizontal rows (top/bottom of each square).
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [15, 16, 17],
  [18, 19, 20],
  [21, 22, 23],
  // Vertical columns (left/right of each square).
  [0, 9, 21],
  [3, 10, 18],
  [6, 11, 15],
  [8, 12, 17],
  [5, 13, 20],
  [2, 14, 23],
  // Cross-midpoint lines through each side of the board.
  [1, 4, 7],
  [9, 10, 11],
  [12, 13, 14],
  [16, 19, 22],
];

interface Obs {
  board: string[]; // length 24, each 'W' | 'B' | '.'
  current_player: string; // 'W' | 'B' | 'terminal' | 'invalid' | ...
  phase: string; // 'placement' | 'movement' | 'flying' | 'capture' | 'terminal'
  men_to_deploy: Record<string, number>;
  num_men: Record<string, number>;
  turn_number: number;
  is_terminal: boolean;
  winner: string | null;
  last_action: string | null;
}

function getObservation(step: any, idx: number = 0): Obs | null {
  const raw = step?.[idx]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Obs;
  } catch {
    return null;
  }
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player W' : 'Player B';
}

interface MoveDiff {
  added: { point: number; piece: string } | null; // piece placed / arrived here this step
  removed: { point: number; piece: string } | null; // piece removed here this step
  isCapture: boolean; // removed piece was opponent's after a mill
}

function diffBoards(prev: Obs | null, curr: Obs): MoveDiff {
  const diff: MoveDiff = { added: null, removed: null, isCapture: false };
  if (!prev) return diff;
  for (let p = 0; p < 24; p++) {
    const a = prev.board[p];
    const b = curr.board[p];
    if (a === b) continue;
    if (a === '.' && b !== '.') {
      diff.added = { point: p, piece: b };
    } else if (a !== '.' && b === '.') {
      diff.removed = { point: p, piece: a };
    }
  }
  // A capture removes an opponent's piece with no corresponding placement of
  // the same color (i.e. a move that lands on an empty point and removes a
  // different-colored piece elsewhere is a mill capture, not a phase-2 move).
  if (diff.removed && diff.added) {
    diff.isCapture = diff.removed.piece !== diff.added.piece;
    if (diff.isCapture) {
      // If the added piece is the OTHER color, this was a capture (the
      // "added" piece belongs to the mill-forming player, "removed" is the
      // opponent). But there's no genuine placement in a phase-2 capture
      // (the mover doesn't move again -- they only remove). So diff.added
      // is only set when the previous action was the mill-forming placement
      // itself. We keep both markers.
    }
  } else if (diff.removed && !diff.added) {
    // Only a removal -- this is definitively a capture step.
    diff.isCapture = true;
  }
  return diff;
}

function millsFor(board: string[], player: string): Array<[number, number, number]> {
  return MILLS.filter(([a, b, c]) => board[a] === player && board[b] === player && board[c] === player);
}

function drawBoard(ctx: CanvasRenderingContext2D, size: number, observation: Obs, diff: MoveDiff) {
  ctx.clearRect(0, 0, size, size);

  const padding = size * 0.08;
  const grid = size - padding * 2;
  const step = grid / 6; // 7x7 grid indexed 0..6

  const px = (gx: number) => padding + gx * step;
  const py = (gy: number) => padding + gy * step;
  const pointOf = (p: number) => {
    const [gx, gy] = POINT_GRID[p];
    return { x: px(gx), y: py(gy) };
  };

  // Draw board line segments with a soft double stroke to look hand-inked.
  ctx.strokeStyle = BOARD_LINE;
  ctx.lineWidth = Math.max(1.5, size * 0.006);
  ctx.setLineDash([]);
  for (const [a, b] of BOARD_SEGMENTS) {
    const p1 = pointOf(a);
    const p2 = pointOf(b);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  // Highlight active mills for each color as translucent gold overlays.
  // Mills are always three collinear points, so a single line from the
  // outer endpoint to the other outer endpoint covers all three pieces.
  ctx.lineWidth = Math.max(6, size * 0.025);
  ctx.strokeStyle = MILL_COLOR;
  for (const player of ['W', 'B']) {
    for (const [a, , c] of millsFor(observation.board, player)) {
      const p1 = pointOf(a);
      const p3 = pointOf(c);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p3.x, p3.y);
      ctx.stroke();
    }
  }

  // Draw the 24 point disks (empty markers) and pieces on top.
  const pointR = step * 0.14;
  const pieceR = step * 0.3;

  for (let p = 0; p < 24; p++) {
    const { x, y } = pointOf(p);
    // Empty-point marker (small filled circle so intersections are visible).
    ctx.beginPath();
    ctx.arc(x, y, pointR, 0, Math.PI * 2);
    ctx.fillStyle = POINT_FILL;
    ctx.fill();
    ctx.strokeStyle = SKETCH_STROKE;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    const cell = observation.board[p];
    if (cell === 'W' || cell === 'B') {
      ctx.beginPath();
      ctx.arc(x, y, pieceR, 0, Math.PI * 2);
      ctx.fillStyle = cell === 'W' ? PLAYER_W_COLOR : PLAYER_B_COLOR;
      ctx.fill();
      ctx.lineWidth = Math.max(1.5, size * 0.005);
      ctx.strokeStyle = cell === 'W' ? PLAYER_W_STROKE : PLAYER_B_STROKE;
      ctx.stroke();
    }
  }

  // Move highlights: ring around a piece that arrived / moved from.
  if (diff.added) {
    const { x, y } = pointOf(diff.added.point);
    ctx.beginPath();
    ctx.arc(x, y, pieceR + Math.max(3, size * 0.012), 0, Math.PI * 2);
    ctx.lineWidth = Math.max(2.5, size * 0.01);
    ctx.strokeStyle = HIGHLIGHT_ADD;
    ctx.setLineDash([]);
    ctx.stroke();
  }
  if (diff.removed) {
    const { x, y } = pointOf(diff.removed.point);
    if (diff.isCapture) {
      // Draw a red "X" where the captured piece stood.
      ctx.strokeStyle = HIGHLIGHT_CAPTURE;
      ctx.lineWidth = Math.max(2.5, size * 0.012);
      const r = pieceR * 0.9;
      ctx.beginPath();
      ctx.moveTo(x - r, y - r);
      ctx.lineTo(x + r, y + r);
      ctx.moveTo(x - r, y + r);
      ctx.lineTo(x + r, y - r);
      ctx.stroke();
    } else {
      // A regular movement: mark the origin with a faded blue ring.
      ctx.beginPath();
      ctx.arc(x, y, pieceR + Math.max(2, size * 0.008), 0, Math.PI * 2);
      ctx.lineWidth = Math.max(2, size * 0.008);
      ctx.strokeStyle = HIGHLIGHT_MOVE_FROM;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

function playerColorFor(label: string): { fill: string; stroke: string } {
  return label === 'W'
    ? { fill: PLAYER_W_COLOR, stroke: PLAYER_W_STROKE }
    : { fill: PLAYER_B_COLOR, stroke: PLAYER_B_STROKE };
}

function playerChip(label: string): string {
  const c = playerColorFor(label);
  return `<span style="display:inline-block;width:0.9em;height:0.9em;border-radius:50%;background:${c.fill};border:1px solid ${c.stroke};vertical-align:middle;margin-right:6px;"></span>`;
}

export function renderer(options: RendererOptions) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as any[];
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
  const observation = getObservation(currentStep, 0);
  if (!observation) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const prev = step > 0 ? getObservation(steps[step - 1], 0) : null;
  const diff = diffBoards(prev, observation);

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const activeIdx = observation.is_terminal
    ? -1
    : observation.current_player === 'W'
      ? 0
      : observation.current_player === 'B'
        ? 1
        : -1;

  const wColor = playerColorFor('W');
  const bColor = playerColorFor('B');

  // OpenSpiel's `num_men` tracks "not-yet-captured" (starts at 9 and only
  // drops on capture); actual pieces on the board is num_men minus the
  // reserve still waiting to be dropped.
  const menSummary = (label: 'W' | 'B') => {
    const reserve = observation.men_to_deploy[label] ?? 0;
    const alive = observation.num_men[label] ?? 0;
    const onBoard = Math.max(0, alive - reserve);
    if (reserve > 0) {
      return `${onBoard} on board, ${reserve} to place`;
    }
    return `${onBoard} on board`;
  };

  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color:${wColor.stroke};">
      ${playerChip('W')}${playerNames[0]}
      <span class="men">${menSummary('W')}</span>
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color:${bColor.stroke};">
      ${playerChip('B')}${playerNames[1]}
      <span class="men">${menSummary('B')}</span>
    </span>
  `;

  const sizeAndDraw = () => {
    const wrapRect = wrap.getBoundingClientRect();
    const avail = Math.min(wrapRect.width, wrapRect.height);
    if (avail <= 0) return;
    const cssSize = Math.max(1, Math.floor(avail));
    canvas.style.width = `${cssSize}px`;
    canvas.style.height = `${cssSize}px`;
    canvas.width = cssSize;
    canvas.height = cssSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawBoard(ctx, cssSize, observation, diff);
  };
  requestAnimationFrame(sizeAndDraw);

  // Status line: phase / turn indicator, plus move annotation.
  let statusHTML = '';
  if (observation.is_terminal) {
    if (observation.winner === 'W') {
      statusHTML = `<span style="color:${wColor.stroke};">${playerNames[0]} (W) wins!</span>`;
    } else if (observation.winner === 'B') {
      statusHTML = `<span style="color:${bColor.stroke};">${playerNames[1]} (B) wins!</span>`;
    } else {
      statusHTML = `<span>Draw</span>`;
    }
  } else {
    const activeColor = activeIdx === 0 ? wColor.stroke : bColor.stroke;
    const activeName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    const phaseLabel =
      observation.phase === 'capture'
        ? 'must capture a piece'
        : observation.phase === 'placement'
          ? 'placing'
          : observation.phase === 'movement'
            ? 'moving'
            : observation.phase === 'flying'
              ? 'flying'
              : observation.phase;
    statusHTML = `<span>Turn: <span style="color:${activeColor}; font-weight:700;">${activeName}</span> <span class="annotation">(${phaseLabel})</span></span>`;
  }

  // Annotate what just happened this step from the board diff.
  const annotations: string[] = [];
  if (diff.isCapture && diff.removed) {
    annotations.push(`captured ${diff.removed.piece} at point ${diff.removed.point}`);
  } else if (diff.added && diff.removed) {
    annotations.push(`moved ${diff.added.piece}: point ${diff.removed.point} → ${diff.added.point}`);
  } else if (diff.added) {
    annotations.push(`placed ${diff.added.piece} at point ${diff.added.point}`);
  }
  annotations.push(`turn ${observation.turn_number}`);
  statusHTML += annotations.map((a) => `<span class="annotation">${a}</span>`).join('');

  statusContainer.innerHTML = statusHTML;
}
