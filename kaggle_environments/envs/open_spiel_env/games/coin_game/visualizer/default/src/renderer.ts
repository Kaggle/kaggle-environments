import type { RendererOptions } from '@kaggle-environments/core';
import type { CoinBoardState, CoinStep } from './transformers/coinGameTransformer';

const PLAYER_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e'];
const COIN_COLORS = ['#c89b1e', '#7d3fa0', '#1aa07a', '#cc4477', '#557acc'];
const CELL_LIGHT = '#fbf7e8';
const CELL_DARK = '#e7dfc1';
const SECONDARY_TEXT = '#444343';
const SKETCH_STROKE = '#3c3b37';

function coinColorFor(idx: number): string {
  return COIN_COLORS[idx % COIN_COLORS.length];
}

function playerColorFor(idx: number): string {
  return PLAYER_COLORS[idx % PLAYER_COLORS.length];
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return `Player ${idx}`;
}

function drawCell(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, fill: string) {
  ctx.fillStyle = fill;
  ctx.fillRect(x, y, size, size);
}

function drawCoin(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, fill: string, label: string) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = SKETCH_STROKE;
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = `700 ${Math.round(r * 1.1)}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, cx, cy + r * 0.05);
  ctx.restore();
}

function drawPlayer(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  fill: string,
  label: string,
  highlight: boolean
) {
  ctx.save();
  if (highlight) {
    ctx.beginPath();
    ctx.arc(cx, cy, r * 1.25, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(189, 238, 255, 0.7)';
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = Math.max(1, r * 0.12);
  ctx.strokeStyle = SKETCH_STROKE;
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = `700 ${Math.round(r * 1.2)}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, cx, cy + r * 0.05);
  ctx.restore();
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
  if (x1 === x2 && y1 === y2) return;
  const headLen = size * 0.35;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const inset = size * 0.4;
  const tx = x2 - inset * Math.cos(angle);
  const ty = y2 - inset * Math.sin(angle);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = Math.max(2, size * 0.07);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
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
  state: CoinBoardState,
  prevState: CoinBoardState | null,
  lastActor: number | null,
  coinIndexFor: (label: string) => number
) {
  const rows = state.num_rows;
  const cols = state.num_columns;
  ctx.clearRect(0, 0, width, height);

  const padding = 24;
  const innerSize = Math.max(0, Math.min(width, height) - padding * 2);
  const cellSize = innerSize / Math.max(rows, cols);
  const xOffset = (width - cellSize * cols) / 2;
  const yOffset = (height - cellSize * rows) / 2;

  // Cells.
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = xOffset + c * cellSize;
      const y = yOffset + r * cellSize;
      drawCell(ctx, x, y, cellSize, (r + c) % 2 === 0 ? CELL_LIGHT : CELL_DARK);
    }
  }

  // Highlight previous and current cell of the player who just moved.
  if (prevState && lastActor !== null) {
    const prev = prevState.player_positions?.[String(lastActor)];
    const cur = state.player_positions?.[String(lastActor)];
    if (prev && cur) {
      const moverColor = playerColorFor(lastActor);
      const px = xOffset + prev[1] * cellSize;
      const py = yOffset + prev[0] * cellSize;
      const cx = xOffset + cur[1] * cellSize;
      const cy = yOffset + cur[0] * cellSize;
      ctx.save();
      ctx.fillStyle = moverColor;
      ctx.globalAlpha = 0.18;
      ctx.fillRect(px, py, cellSize, cellSize);
      ctx.fillRect(cx, cy, cellSize, cellSize);
      ctx.restore();
      ctx.save();
      ctx.strokeStyle = moverColor;
      ctx.lineWidth = Math.max(2, cellSize * 0.08);
      ctx.strokeRect(cx + 1, cy + 1, cellSize - 2, cellSize - 2);
      ctx.restore();
    }
  }

  // Sketched outer border + grid lines.
  ctx.strokeStyle = SKETCH_STROKE;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([]);
  ctx.strokeRect(xOffset, yOffset, cellSize * cols, cellSize * rows);
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  for (let i = 1; i < cols; i++) {
    ctx.beginPath();
    ctx.moveTo(xOffset + i * cellSize, yOffset);
    ctx.lineTo(xOffset + i * cellSize, yOffset + cellSize * rows);
    ctx.stroke();
  }
  for (let i = 1; i < rows; i++) {
    ctx.beginPath();
    ctx.moveTo(xOffset, yOffset + i * cellSize);
    ctx.lineTo(xOffset + cellSize * cols, yOffset + i * cellSize);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Pieces (coins, then players on top).
  const coinR = cellSize * 0.32;
  const playerR = cellSize * 0.36;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = state.board?.[r]?.[c];
      if (!cell || cell === '.') continue;
      const cx = xOffset + c * cellSize + cellSize / 2;
      const cy = yOffset + r * cellSize + cellSize / 2;
      if (/^[a-z]$/.test(cell)) {
        const idx = coinIndexFor(cell);
        drawCoin(ctx, cx, cy, coinR, coinColorFor(idx), cell.toUpperCase());
      }
    }
  }
  // Players drawn last so they sit above coins (e.g., the cell they just
  // moved onto).
  const numPlayers = Object.keys(state.player_positions ?? {}).length;
  for (let p = 0; p < numPlayers; p++) {
    const pos = state.player_positions[String(p)];
    if (!pos) continue;
    const cx = xOffset + pos[1] * cellSize + cellSize / 2;
    const cy = yOffset + pos[0] * cellSize + cellSize / 2;
    drawPlayer(ctx, cx, cy, playerR, playerColorFor(p), `${p}`, p === state.current_player && !state.is_terminal);
  }

  // Arrow from previous to current cell of the player who just moved.
  if (prevState && lastActor !== null) {
    const prev = prevState.player_positions?.[String(lastActor)];
    const cur = state.player_positions?.[String(lastActor)];
    if (prev && cur && (prev[0] !== cur[0] || prev[1] !== cur[1])) {
      const px = xOffset + prev[1] * cellSize + cellSize / 2;
      const py = yOffset + prev[0] * cellSize + cellSize / 2;
      const cx = xOffset + cur[1] * cellSize + cellSize / 2;
      const cy = yOffset + cur[0] * cellSize + cellSize / 2;
      drawArrow(ctx, px, py, cx, cy, playerColorFor(lastActor), cellSize);
    }
  }

  // Coordinate labels.
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.font = `${Math.round(cellSize * 0.28)}px 'Inter', sans-serif`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  for (let c = 0; c < cols; c++) {
    ctx.fillText(`${c}`, xOffset + c * cellSize + cellSize / 2, yOffset + cellSize * rows + cellSize * 0.32);
  }
  ctx.textAlign = 'right';
  for (let r = 0; r < rows; r++) {
    ctx.fillText(`${r}`, xOffset - cellSize * 0.12, yOffset + r * cellSize + cellSize / 2);
  }
}

export function renderer(options: RendererOptions<CoinStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as CoinStep[];
  if (!steps.length) return;

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="board-wrap"><canvas></canvas></div>
      <div class="scoreboard"></div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const wrap = parent.querySelector('.board-wrap') as HTMLDivElement;
  const canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
  const scoreboard = parent.querySelector('.scoreboard') as HTMLDivElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;

  const currentStep = steps[step] ?? steps[steps.length - 1];
  const prevStep = step > 0 ? steps[step - 1] : null;
  const state = currentStep?.boardState;
  if (!state) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const numPlayers = Object.keys(state.player_positions ?? {}).length;
  const playerNames = Array.from({ length: numPlayers }, (_, i) => getPlayerName(replay, i));
  const coinIndexFor = (label: string) => Math.max(0, (state.coin_colors ?? []).indexOf(label));

  // Discover each player's preference if revealed (terminal state) or
  // available from any per-step private observation we've seen so far.
  const preferences: Record<number, string | null> = {};
  for (let p = 0; p < numPlayers; p++) preferences[p] = null;
  if (state.preferences) {
    for (const [k, v] of Object.entries(state.preferences)) {
      preferences[Number(k)] = v;
    }
  }
  for (let i = 0; i <= step; i++) {
    const s = steps[i];
    s?.privateObs?.forEach((obs, idx) => {
      if (obs?.your_preference) preferences[idx] = obs.your_preference;
    });
  }

  // --- Header ---
  header.innerHTML = playerNames
    .map((name, i) => {
      const active = !state.is_terminal && state.current_player === i;
      const pref = preferences[i];
      const prefHtml = pref
        ? `<span class="pref">pref <span style="color: ${coinColorFor(coinIndexFor(pref))}; font-weight: 700;">${pref.toUpperCase()}</span></span>`
        : `<span class="pref">pref ?</span>`;
      return `
        <span class="player sketched-border ${active ? 'active' : ''}" style="color: ${playerColorFor(i)};">
          <span style="display:inline-flex;align-items:center;gap:6px;">
            <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:${playerColorFor(i)};border:1px solid ${SKETCH_STROKE};"></span>
            ${name}
          </span>
          ${prefHtml}
        </span>
      `;
    })
    .join('<span class="vs">vs</span>');

  // --- Scoreboard: per-player coin counts by colour ---
  const coinColors = state.coin_colors ?? [];
  scoreboard.innerHTML = playerNames
    .map((name, p) => {
      const counts = state.coins_collected?.[String(p)] ?? {};
      const items = coinColors
        .map(
          (c) => `
            <span style="display:inline-flex;align-items:center;gap:4px;">
              <span class="swatch" style="background:${coinColorFor(coinIndexFor(c))};"></span>
              ${counts[c] ?? 0}
            </span>
          `
        )
        .join('&nbsp;');
      return `
        <span class="row sketched-border">
          <span style="color:${playerColorFor(p)};font-weight:700;">${name}</span>
          ${items}
        </span>
      `;
    })
    .join('');

  // --- Canvas ---
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
    drawBoard(ctx, side, side, state, prevStep?.boardState ?? null, currentStep.lastActor, coinIndexFor);
  };
  requestAnimationFrame(sizeAndDraw);

  // --- Status ---
  let statusHTML = '';
  if (state.is_terminal) {
    if (state.winner === 'draw') {
      statusHTML = `<span>Draw &mdash; returns ${state.returns?.map((r) => r.toFixed(0)).join(' / ') ?? '?'}</span>`;
    } else if (state.winner !== null && state.winner !== undefined) {
      const widx = Number(state.winner);
      statusHTML = `<span style="color: ${playerColorFor(widx)};">${playerNames[widx] ?? `Player ${widx}`} wins (${state.returns?.[widx]?.toFixed(0) ?? '?'} pts)</span>`;
    } else {
      statusHTML = '<span>Game over</span>';
    }
  } else {
    const turnColor = playerColorFor(state.current_player);
    const turnName = playerNames[state.current_player] ?? `Player ${state.current_player}`;
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;
  }
  if (currentStep.lastAction) {
    const actorColor = currentStep.lastActor !== null ? playerColorFor(currentStep.lastActor) : SECONDARY_TEXT;
    statusHTML += `<span class="annotation">last: <span style="color: ${actorColor}; font-weight: 600;">${currentStep.lastAction}</span></span>`;
  }
  statusHTML += `<span class="annotation">move ${state.move_number}/${state.episode_length}</span>`;
  statusContainer.innerHTML = statusHTML;
}
