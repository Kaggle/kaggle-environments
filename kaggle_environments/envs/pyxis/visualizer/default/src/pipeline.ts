import { ASSET_STATES, THERAPEUTIC_AREAS, TRIAL_PHASES, type AssetView, type PlayerView } from './types';

/**
 * The pipeline board: a drug's journey left to right.
 *
 * Columns are the trial phases plus a market column; rows are therapeutic
 * areas. The two players share the column layout but occupy opposite halves of
 * each row, so a lead in one area reads as an imbalance across the divider.
 */
const COLUMNS = [...TRIAL_PHASES, 'On Market'];

const INK = '#050001';
const RULE = '#3c3b37';
const MUTED = '#8a8880';

/** Per-player chip fill, kept distinct at any PTRS shading. */
const PLAYER_HUES = [205, 22];

const PAD = { top: 26, left: 92, right: 8, bottom: 8 };

function dashedLine(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, dash = [4, 4]) {
  ctx.save();
  ctx.setLineDash(dash);
  ctx.strokeStyle = RULE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

/** Column a live asset belongs in; -1 for anything no longer in play. */
function columnOf(asset: AssetView): number {
  const state = ASSET_STATES[asset.state];
  if (state === 'On Market') return COLUMNS.length - 1;
  if (state === 'Idle' || state === 'In Development') return asset.phase >= 0 ? asset.phase : 0;
  return -1;
}

/**
 * Chip radius by peak revenue, on a square-root scale so a £10B asset reads as
 * bigger than a £1B one without swamping the cell.
 */
function radiusOf(asset: AssetView, cell: number): number {
  const scale = Math.sqrt(Math.min(1, Math.max(0, asset.maxRevenue / 1e10)));
  return Math.max(3, Math.min(cell * 0.34, 3 + scale * cell * 0.3));
}

function chipStyle(asset: AssetView, playerIdx: number): { fill: string; stroke: string } {
  const hue = PLAYER_HUES[playerIdx % PLAYER_HUES.length];
  const state = ASSET_STATES[asset.state];
  if (state === 'Idle') {
    // Not yet funded: hollow, so the eye skips it.
    return { fill: 'rgba(255,255,255,0.85)', stroke: MUTED };
  }
  // Probability of technical and regulatory success drives saturation.
  const confidence = Math.min(1, Math.max(0, asset.ptrs));
  const light = 84 - confidence * 40;
  return { fill: `hsl(${hue} 62% ${light}%)`, stroke: RULE };
}

function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const gridW = w - PAD.left - PAD.right;
  const gridH = h - PAD.top - PAD.bottom;
  const colW = gridW / COLUMNS.length;
  const rowH = gridH / THERAPEUTIC_AREAS.length;

  ctx.font = "600 10px 'Inter', sans-serif";
  ctx.fillStyle = INK;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  COLUMNS.forEach((label, c) => {
    ctx.fillText(label, PAD.left + colW * (c + 0.5), PAD.top / 2);
    if (c > 0) dashedLine(ctx, PAD.left + colW * c, PAD.top, PAD.left + colW * c, h - PAD.bottom);
  });

  ctx.textAlign = 'right';
  ctx.font = "500 9px 'Inter', sans-serif";
  THERAPEUTIC_AREAS.forEach((label, r) => {
    const y = PAD.top + rowH * (r + 0.5);
    // Wrap the long TA names onto two lines rather than clipping them.
    const words = label.split(' ');
    const mid = Math.ceil(words.length / 2);
    const lines = words.length > 1 ? [words.slice(0, mid).join(' '), words.slice(mid).join(' ')] : [label];
    lines.forEach((line, i) => {
      ctx.fillText(line, PAD.left - 8, y + (i - (lines.length - 1) / 2) * 11);
    });
    if (r > 0) dashedLine(ctx, PAD.left, PAD.top + rowH * r, w - PAD.right, PAD.top + rowH * r);
  });

  // Mid-row divider: player 0 above, player 1 below.
  THERAPEUTIC_AREAS.forEach((_, r) => {
    const y = PAD.top + rowH * (r + 0.5);
    dashedLine(ctx, PAD.left, y, w - PAD.right, y, [2, 5]);
  });

  ctx.strokeStyle = RULE;
  ctx.lineWidth = 1;
  ctx.strokeRect(PAD.left + 0.5, PAD.top + 0.5, gridW - 1, gridH - 1);
  return { colW, rowH };
}

function drawPlayerAssets(
  ctx: CanvasRenderingContext2D,
  player: PlayerView,
  playerIdx: number,
  colW: number,
  rowH: number
) {
  // Bucket by cell first so chips within a cell can be packed rather than
  // drawn on top of each other.
  const cells = new Map<string, AssetView[]>();
  for (const asset of player.assets) {
    const col = columnOf(asset);
    if (col < 0) continue;
    const key = `${asset.therapeuticArea}:${col}`;
    const bucket = cells.get(key);
    if (bucket) bucket.push(asset);
    else cells.set(key, [asset]);
  }

  const half = rowH / 2;
  for (const [key, assets] of cells) {
    const [therapeuticArea, col] = key.split(':').map(Number);
    const x0 = PAD.left + colW * col;
    // Player 0 takes the top half of the row, player 1 the bottom.
    const y0 = PAD.top + rowH * therapeuticArea + (playerIdx === 0 ? 0 : half);

    assets.sort((a, b) => b.maxRevenue - a.maxRevenue);
    const perRow = Math.max(1, Math.floor(colW / 16));
    assets.forEach((asset, i) => {
      const gx = i % perRow;
      const gy = Math.floor(i / perRow);
      const cx = x0 + colW * ((gx + 0.5) / perRow);
      const cy = y0 + 10 + gy * 13;
      if (cy > y0 + half - 3) return; // overflow guard; count is shown in the panels
      const r = radiusOf(asset, Math.min(colW / perRow, half));
      const { fill, stroke } = chipStyle(asset, playerIdx);

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.stroke();

      // An accelerated asset gets a ring; it is the loudest thing a player does.
      if (asset.investmentLevel >= 3) {
        ctx.beginPath();
        ctx.arc(cx, cy, r + 2.5, 0, Math.PI * 2);
        ctx.strokeStyle = INK;
        ctx.stroke();
      }
    });
  }
}

export function drawPipeline(canvas: HTMLCanvasElement, players: PlayerView[]) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w <= 0 || h <= 0) return;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const { colW, rowH } = drawGrid(ctx, w, h);
  players.forEach((player, i) => drawPlayerAssets(ctx, player, i, colW, rowH));
}
