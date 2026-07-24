import { escapeHtml, type RendererOptions } from '@kaggle-environments/core';
import type { ReversiBoardState, ReversiCell, ReversiStep } from './transformers/reversiTransformer';

const INK = '#050001';
const SOFT_INK = '#3c3b37';
const SECONDARY_TEXT = '#444343';
const BOARD_TINT = '#e8e0c4'; // muted parchment-toned cell fill
const P0_COLOR = '#050001'; // Black disk (x) ring + accent
const P1_COLOR = '#c9a24a'; // White disk (o) ring + accent (warm gold on parchment)
const DISK_WHITE = '#f8f4e3'; // creamy off-white for White disks
const PLACED_RING = '#c9381c'; // strong accent for the newly-placed disk
const FLIP_RING = '#4a7f9d'; // muted blue used for the flipped-cell wash
const PLACED_WASH = 'rgba(201, 56, 28, 0.22)';
const FLIP_WASH = 'rgba(74, 127, 157, 0.28)';
const BEAM_COLOR = 'rgba(74, 127, 157, 0.42)';
const PASS_TAG = '#8a6d3b';

// Timing (ms). Each captured disc flips in FLIP_DISC_MS, and successive discs
// along the same ray start FLIP_STAGGER_MS apart — so the flip "cascades"
// outward from the placed disc along each direction. Two rays from the same
// placed disc animate in parallel because both start their distance-1 discs
// at t=0; they simply point different ways, which is what visually
// distinguishes them.
const FLIP_DISC_MS = 320;
const FLIP_STAGGER_MS = 90;
const PLACED_POP_MS = 220;
const RING_FADE_IN_MS = 380;
const STATE_KEY = '__reversiRenderState';

interface FlippedCell {
  row: number;
  col: number;
  oldCell: ReversiCell;
  newCell: ReversiCell;
  // Populated by diffBoards once the placed disc is known.
  rayIndex: number; // -1 if unassigned
  distance: number; // Chebyshev distance from placed disc (1 = adjacent)
}

interface FlipRay {
  dr: number; // -1 | 0 | 1
  dc: number; // -1 | 0 | 1
  cells: FlippedCell[]; // sorted by distance ascending
  anchor: { row: number; col: number }; // same-color disc closing the ray
}

interface MoveHighlight {
  placed: { row: number; col: number } | null;
  flipped: FlippedCell[];
  rays: FlipRay[];
  wasPass: boolean;
}

interface RendererState {
  lastStep: number;
  rafId: number | null;
}

function getState(parent: HTMLElement): RendererState {
  const holder = parent as unknown as Record<string, RendererState | undefined>;
  let state = holder[STATE_KEY];
  if (!state) {
    state = { lastStep: -1, rafId: null };
    holder[STATE_KEY] = state;
  }
  return state;
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
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
  const highlight: MoveHighlight = { placed: null, flipped: [], rays: [], wasPass };
  if (!prev || wasPass) return highlight;
  for (let r = 0; r < curr.length; r++) {
    for (let c = 0; c < curr[r].length; c++) {
      const before = prev[r]?.[c] ?? '';
      const after = curr[r][c];
      if (before === after) continue;
      if (before === '' && after !== '') {
        highlight.placed = { row: r, col: c };
      } else if (before !== '' && after !== '' && before !== after) {
        highlight.flipped.push({ row: r, col: c, oldCell: before, newCell: after, rayIndex: -1, distance: 0 });
      }
    }
  }
  // Group flipped discs by direction from the placed disc. Othello captures
  // along the 8 straight rays, so sign(dr), sign(dc) uniquely identifies a
  // ray, and Chebyshev distance orders discs along it.
  if (highlight.placed) {
    const pr = highlight.placed.row;
    const pc = highlight.placed.col;
    const groups = new Map<string, FlippedCell[]>();
    for (const cell of highlight.flipped) {
      const dr = Math.sign(cell.row - pr);
      const dc = Math.sign(cell.col - pc);
      cell.distance = Math.max(Math.abs(cell.row - pr), Math.abs(cell.col - pc));
      const key = `${dr},${dc}`;
      let list = groups.get(key);
      if (!list) {
        list = [];
        groups.set(key, list);
      }
      list.push(cell);
    }
    let rayIndex = 0;
    for (const [key, cells] of groups) {
      cells.sort((a, b) => a.distance - b.distance);
      for (const c of cells) c.rayIndex = rayIndex;
      const [dr, dc] = key.split(',').map(Number);
      const maxDist = cells[cells.length - 1].distance;
      highlight.rays.push({
        dr,
        dc,
        cells,
        anchor: { row: pr + dr * (maxDist + 1), col: pc + dc * (maxDist + 1) },
      });
      rayIndex++;
    }
  }
  return highlight;
}

function drawDisk(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  cell: ReversiCell,
  scaleX: number = 1,
  scaleY: number = 1
) {
  if (cell !== 'x' && cell !== 'o') return;
  if (scaleX <= 0 || scaleY <= 0) return;
  const isBlack = cell === 'x';
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(scaleX, scaleY);
  ctx.beginPath();
  ctx.arc(0, 0, radius, 0, Math.PI * 2);
  ctx.fillStyle = isBlack ? INK : DISK_WHITE;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = isBlack ? P0_COLOR : P1_COLOR;
  ctx.stroke();
  ctx.restore();
}

function getTotalAnimMs(highlight: MoveHighlight): number {
  let maxDist = 0;
  for (const cell of highlight.flipped) {
    if (cell.distance > maxDist) maxDist = cell.distance;
  }
  const flipTotal = FLIP_DISC_MS + Math.max(0, maxDist - 1) * FLIP_STAGGER_MS;
  return Math.max(PLACED_POP_MS, RING_FADE_IN_MS, flipTotal);
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: ReversiBoardState,
  highlight: MoveHighlight,
  elapsedMs: number = Number.POSITIVE_INFINITY
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

  // Cell washes: peripheral cue that survives fast playback and static scrubbing.
  // Placed cell gets a red wash; each flipped cell gets a blue wash. Drawn under
  // the grid lines so the sketched grid still reads on top.
  if (highlight.placed) {
    ctx.fillStyle = PLACED_WASH;
    ctx.fillRect(
      originX + highlight.placed.col * cellSize,
      originY + highlight.placed.row * cellSize,
      cellSize,
      cellSize
    );
  }
  for (const f of highlight.flipped) {
    ctx.fillStyle = FLIP_WASH;
    ctx.fillRect(originX + f.col * cellSize, originY + f.row * cellSize, cellSize, cellSize);
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

  // Ray beams: one line per ray, from placed-disc center to the anchor cell
  // center. Sits under the disks so mid-flip (scaleX≈0) the beam shows through.
  // Beam grows as its cascade advances, then holds full length; on scrub /
  // static frames it renders full-length immediately.
  const radius = cellSize * 0.4;
  if (highlight.placed && highlight.rays.length) {
    const pxC = originX + (highlight.placed.col + 0.5) * cellSize;
    const pyC = originY + (highlight.placed.row + 0.5) * cellSize;
    ctx.save();
    ctx.strokeStyle = BEAM_COLOR;
    ctx.lineWidth = Math.max(2.5, cellSize * 0.13);
    ctx.lineCap = 'round';
    for (const ray of highlight.rays) {
      const ax = originX + (ray.anchor.col + 0.5) * cellSize;
      const ay = originY + (ray.anchor.row + 0.5) * cellSize;
      // Ray "length progress": how far along its cells the cascade has reached.
      // Uses each cell's own local flip progress so the beam tip tracks the
      // currently-flipping disc.
      let reachedIdx = -1;
      let tipFrac = 0;
      for (let i = 0; i < ray.cells.length; i++) {
        const cell = ray.cells[i];
        const startMs = (cell.distance - 1) * FLIP_STAGGER_MS;
        const local = Math.max(0, Math.min(1, (elapsedMs - startMs) / FLIP_DISC_MS));
        if (local > 0) {
          reachedIdx = i;
          tipFrac = local;
        } else {
          break;
        }
      }
      // Interpolate the beam endpoint along the ray. Reaches the anchor cell
      // center when the last disc's local progress hits 1.
      const totalSegments = ray.cells.length; // discs, not counting anchor
      let beamFrac: number;
      if (reachedIdx < 0) {
        beamFrac = 0;
      } else {
        beamFrac = (reachedIdx + tipFrac) / totalSegments;
      }
      if (!isFinite(elapsedMs) || elapsedMs >= getTotalAnimMs(highlight)) beamFrac = 1;
      if (beamFrac <= 0) continue;
      const endX = pxC + (ax - pxC) * beamFrac;
      const endY = pyC + (ay - pyC) * beamFrac;
      ctx.beginPath();
      ctx.moveTo(pxC, pyC);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Disks — cascading animation for flipped cells (staggered by distance),
  // pop for placed, static for the rest.
  const flippedByKey = new Map(highlight.flipped.map((f) => [`${f.row},${f.col}`, f]));
  const placedKey = highlight.placed ? `${highlight.placed.row},${highlight.placed.col}` : null;
  const placedProgress = Math.max(0, Math.min(1, elapsedMs / PLACED_POP_MS));

  for (let r = 0; r < obs.rows; r++) {
    for (let c = 0; c < obs.columns; c++) {
      const key = `${r},${c}`;
      const cx = originX + (c + 0.5) * cellSize;
      const cy = originY + (r + 0.5) * cellSize;
      const flip = flippedByKey.get(key);

      if (key === placedKey) {
        // Placed disc "pops" in from scale 0 → 1.
        const eased = easeInOutCubic(placedProgress);
        drawDisk(ctx, cx, cy, radius, obs.board[r][c], eased, eased);
      } else if (flip) {
        // Per-disc local progress derived from its own start offset. Discs
        // that haven't started yet render statically in their OLD color at
        // full scale, so the whole board reads correctly at t=0.
        const startMs = (flip.distance - 1) * FLIP_STAGGER_MS;
        const local = Math.max(0, Math.min(1, (elapsedMs - startMs) / FLIP_DISC_MS));
        const eased = easeInOutCubic(local);
        let displayCell: ReversiCell;
        let scaleX: number;
        if (eased < 0.5) {
          displayCell = flip.oldCell;
          scaleX = 1 - eased * 2;
        } else {
          displayCell = flip.newCell;
          scaleX = eased * 2 - 1;
        }
        const scaleY = 1 - 0.08 * (1 - Math.abs(eased * 2 - 1));
        drawDisk(ctx, cx, cy, radius, displayCell, scaleX, scaleY);
      } else {
        drawDisk(ctx, cx, cy, radius, obs.board[r][c]);
      }
    }
  }

  // Placed-disc red ring on top: fades in as the disc finishes its pop.
  if (highlight.placed) {
    const ringAlpha = Math.max(
      0,
      Math.min(1, (elapsedMs - PLACED_POP_MS * 0.4) / (RING_FADE_IN_MS - PLACED_POP_MS * 0.4))
    );
    if (ringAlpha > 0) {
      const cx = originX + (highlight.placed.col + 0.5) * cellSize;
      const cy = originY + (highlight.placed.row + 0.5) * cellSize;
      ctx.save();
      ctx.globalAlpha = ringAlpha;
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 5, 0, Math.PI * 2);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = PLACED_RING;
      ctx.stroke();
      ctx.restore();
    }
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

  const state = getState(parent);
  if (state.rafId != null) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }

  // Reuse DOM across renders. Wiping innerHTML on every step change destroys
  // the canvas element and, for the ~1 frame between wipe and redraw, lets
  // the browser paint a blank canvas — that was the inter-step flicker.
  let root = parent.querySelector('.renderer-container') as HTMLDivElement | null;
  if (!root) {
    parent.innerHTML = `
      <div class="renderer-container">
        <div class="header"></div>
        <div class="board-wrap"><canvas></canvas></div>
        <div class="status-container sketched-border"></div>
      </div>
    `;
    root = parent.querySelector('.renderer-container') as HTMLDivElement;
  }
  const header = root.querySelector('.header') as HTMLDivElement;
  const wrap = root.querySelector('.board-wrap') as HTMLDivElement;
  const canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
  const statusContainer = root.querySelector('.status-container') as HTMLDivElement;

  const currentStep = steps[step];
  const obs: ReversiBoardState | null = currentStep?.boardState ?? null;
  if (!obs) {
    statusContainer.textContent = 'Waiting for first observation...';
    state.lastStep = step;
    return;
  }

  const prevStep = step > 0 ? steps[step - 1] : null;
  const prevBoard = prevStep?.boardState?.board ?? null;
  const highlight = diffBoards(prevBoard, obs.board, obs.last_move);

  // Only animate on natural forward steps (auto-play or "next" click). Scrubbing
  // backward or jumping renders a static final frame — no animation churn.
  const isForwardStep = step === state.lastStep + 1;
  const hasChange = highlight.placed !== null || highlight.flipped.length > 0;
  const shouldAnimate = isForwardStep && hasChange;
  state.lastStep = step;

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  // Prefer currentStep.isTerminal — it also fires on forfeits, which the raw
  // OpenSpiel observation.is_terminal does not.
  const isTerminal = !!currentStep?.isTerminal || obs.is_terminal;
  const forfeitReason = currentStep?.forfeitReason ?? null;
  const forfeiterIdx = currentStep?.players?.findIndex((p) => p.forfeited) ?? -1;
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

  let cssW = 0;
  let cssH = 0;
  let ctx: CanvasRenderingContext2D | null = null;

  const resize = () => {
    const wrapRect = wrap.getBoundingClientRect();
    const availW = wrapRect.width;
    const availH = wrapRect.height;
    if (availW <= 0 || availH <= 0) return false;
    cssW = Math.max(1, Math.floor(availW));
    cssH = Math.max(1, Math.floor(availH));
    // Only touch the canvas backing store when dimensions actually change —
    // assigning to canvas.width/height *always* clears the canvas, even when
    // the new value equals the old one. Skipping the write preserves the
    // previous frame's pixels across step changes so the transition is
    // seamless while the new frame is being drawn.
    if (canvas.width !== cssW || canvas.height !== cssH) {
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
      canvas.width = cssW;
      canvas.height = cssH;
    }
    ctx = canvas.getContext('2d');
    return ctx !== null;
  };

  const runAnimation = () => {
    if (!resize() || !ctx) return false;
    if (!shouldAnimate) {
      drawBoard(ctx, cssW, cssH, obs, highlight);
      return true;
    }
    const totalMs = getTotalAnimMs(highlight);
    const start = performance.now();
    // Draw the first animation frame synchronously so the browser never gets
    // a chance to paint an in-between state (blank canvas / stale disks).
    drawBoard(ctx, cssW, cssH, obs, highlight, 0);
    const tick = (now: number) => {
      if (!ctx) return;
      const elapsed = now - start;
      drawBoard(ctx, cssW, cssH, obs, highlight, elapsed);
      if (elapsed < totalMs) {
        state.rafId = requestAnimationFrame(tick);
      } else {
        state.rafId = null;
      }
    };
    state.rafId = requestAnimationFrame(tick);
    return true;
  };

  // Draw synchronously in-line. If layout isn't ready (very first mount before
  // the flex container has been sized), fall back to a single RAF retry.
  if (!runAnimation()) {
    requestAnimationFrame(runAnimation);
  }

  let statusHTML = '';
  if (isTerminal) {
    if (obs.winner === 'x') {
      statusHTML = `<span style="color: ${P0_COLOR};">${playerNames[0]} wins ${disks.x}–${disks.o}</span>`;
    } else if (obs.winner === 'o') {
      statusHTML = `<span style="color: ${P1_COLOR};">${playerNames[1]} wins ${disks.o}–${disks.x}</span>`;
    } else if (forfeitReason && forfeiterIdx >= 0) {
      const winnerIdx = 1 - forfeiterIdx;
      const winnerColor = winnerIdx === 0 ? P0_COLOR : P1_COLOR;
      statusHTML = `<span style="color: ${winnerColor};">${playerNames[winnerIdx]} wins by default</span>`;
    } else {
      statusHTML = `<span>Draw ${disks.x}–${disks.o}</span>`;
    }
    if (forfeitReason) {
      statusHTML += `<span class="annotation forfeit-reason">${escapeHtml(forfeitReason)}</span>`;
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
