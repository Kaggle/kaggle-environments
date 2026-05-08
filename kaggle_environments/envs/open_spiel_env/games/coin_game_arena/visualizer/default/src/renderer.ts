import type { RendererOptions } from '@kaggle-environments/core';
import type { ArenaBoardView, ArenaStep } from './transformers/coinGameArenaTransformer';

const PLAYER_COLORS = ['#1f77b4', '#3a9af0', '#d62728', '#ff7f6e'];
const TEAM_COLORS = ['#1f77b4', '#d62728'];
const COIN_COLORS = ['#c89b1e', '#7d3fa0', '#1aa07a', '#cc4477', '#557acc'];
const CELL_LIGHT = '#fbf7e8';
const CELL_DARK = '#e7dfc1';
const SECONDARY_TEXT = '#444343';
const SKETCH_STROKE = '#3c3b37';
const PLAYERS_PER_TEAM = 2;

function coinColorFor(idx: number): string {
  return COIN_COLORS[Math.max(0, idx) % COIN_COLORS.length];
}

function playerColorFor(pid: number): string {
  return PLAYER_COLORS[pid % PLAYER_COLORS.length];
}

function teamColorFor(team: number): string {
  return TEAM_COLORS[team % TEAM_COLORS.length];
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
  state: ArenaBoardView,
  prevState: ArenaBoardView | null,
  lastActor: number | null,
  coinIndexFor: (label: string) => number
) {
  const rows = state.num_rows;
  const cols = state.num_columns;
  ctx.clearRect(0, 0, width, height);

  const padding = 18;
  const innerSize = Math.max(0, Math.min(width, height) - padding * 2);
  const cellSize = innerSize / Math.max(rows, cols);
  const xOffset = (width - cellSize * cols) / 2;
  const yOffset = (height - cellSize * rows) / 2;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = xOffset + c * cellSize;
      const y = yOffset + r * cellSize;
      drawCell(ctx, x, y, cellSize, (r + c) % 2 === 0 ? CELL_LIGHT : CELL_DARK);
    }
  }

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

  for (const [pidStr, pos] of Object.entries(state.player_positions ?? {})) {
    if (!pos) continue;
    const pid = Number(pidStr);
    const cx = xOffset + pos[1] * cellSize + cellSize / 2;
    const cy = yOffset + pos[0] * cellSize + cellSize / 2;
    drawPlayer(ctx, cx, cy, playerR, playerColorFor(pid), `${pid}`, lastActor === pid);
  }

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

  ctx.fillStyle = SECONDARY_TEXT;
  ctx.font = `${Math.round(cellSize * 0.26)}px 'Inter', sans-serif`;
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

function renderTeamHeader(
  team: number,
  playerNames: string[],
  preferences: (string | null)[],
  activeSeat: number | null,
  isTerminal: boolean,
  coinIndexFor: (label: string) => number
): string {
  const baseId = team * PLAYERS_PER_TEAM;
  const pids = [baseId, baseId + 1];
  const seatHTML = pids
    .map((pid) => {
      const seat = pid % PLAYERS_PER_TEAM;
      const isActive = !isTerminal && activeSeat === seat;
      const pref = preferences[pid];
      const prefHtml = pref
        ? `<span class="pref">pref <span style="color: ${coinColorFor(coinIndexFor(pref))}; font-weight: 700;">${pref.toUpperCase()}</span></span>`
        : `<span class="pref">pref ?</span>`;
      return `
        <span class="seat ${isActive ? 'active' : ''}" style="color: ${playerColorFor(pid)};">
          <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${playerColorFor(pid)};border:1px solid ${SKETCH_STROKE};"></span>
          ${playerNames[pid]}
          ${prefHtml}
        </span>
      `;
    })
    .join('');
  return `
    <span class="team-block sketched-border">
      <span class="team-label" style="color:${teamColorFor(team)};">Team ${team === 0 ? 'A' : 'B'}</span>
      <span class="seats">${seatHTML}</span>
    </span>
  `;
}

function renderBoardScore(
  board: ArenaBoardView,
  playerNames: string[],
  coinIndexFor: (label: string) => number
): string {
  const coinColors = board.coin_colors ?? [];
  return Object.entries(board.coins_collected ?? {})
    .map(([pidStr, counts]) => {
      const pid = Number(pidStr);
      const items = coinColors
        .map(
          (c) => `
            <span style="display:inline-flex;align-items:center;gap:3px;">
              <span class="swatch" style="background:${coinColorFor(coinIndexFor(c))};"></span>
              ${counts[c] ?? 0}
            </span>
          `
        )
        .join('&nbsp;');
      return `
        <span class="row">
          <span style="color:${playerColorFor(pid)};font-weight:700;">${playerNames[pid]}</span>
          ${items}
        </span>
      `;
    })
    .join('');
}

export function renderer(options: RendererOptions<ArenaStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as ArenaStep[];
  if (!steps.length) return;

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="boards-row">
        <div class="board-column" data-team="0">
          <div class="team-banner" style="color:${teamColorFor(0)};">Team A board</div>
          <div class="board-canvas-wrap"><canvas></canvas></div>
          <div class="board-score"></div>
        </div>
        <div class="board-column" data-team="1">
          <div class="team-banner" style="color:${teamColorFor(1)};">Team B board</div>
          <div class="board-canvas-wrap"><canvas></canvas></div>
          <div class="board-score"></div>
        </div>
      </div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;
  const teamColumns = Array.from(parent.querySelectorAll('.board-column')) as HTMLDivElement[];

  const currentStep = steps[step] ?? steps[steps.length - 1];
  const prevStep = step > 0 ? steps[step - 1] : null;

  const playerNames = Array.from({ length: 4 }, (_, i) => getPlayerName(replay, i));
  const allCoinColors = Array.from(
    new Set(currentStep.boards.filter((b): b is ArenaBoardView => b !== null).flatMap((b) => b.coin_colors ?? []))
  );
  const coinIndexFor = (label: string) => Math.max(0, allCoinColors.indexOf(label));

  // Header.
  const headerHTML = [0, 1]
    .map((team) =>
      renderTeamHeader(
        team,
        playerNames,
        currentStep.preferences,
        currentStep.activeSeat,
        currentStep.isTerminal,
        coinIndexFor
      )
    )
    .join('<span class="vs">vs</span>');
  header.innerHTML = headerHTML;

  // Boards.
  teamColumns.forEach((col) => {
    const team = Number(col.dataset.team ?? '0');
    const board = currentStep.boards[team];
    const prevBoard = prevStep?.boards[team] ?? null;
    const lastAction = currentStep.lastActionPerBoard[team];
    const lastActor = lastAction?.actor ?? null;
    const wrap = col.querySelector('.board-canvas-wrap') as HTMLDivElement;
    const canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
    const score = col.querySelector('.board-score') as HTMLDivElement;

    if (!board) {
      score.innerHTML = '<span class="annotation">no board data</span>';
      return;
    }

    score.innerHTML = renderBoardScore(board, playerNames, coinIndexFor);

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
      drawBoard(ctx, side, side, board, prevBoard, lastActor, coinIndexFor);
    };
    requestAnimationFrame(sizeAndDraw);
  });

  // Status bar.
  let statusHTML = '';
  if (currentStep.isTerminal) {
    const totals = currentStep.teamTotals;
    if (currentStep.winningTeam === 'draw') {
      statusHTML = `<span>Draw &mdash; team totals ${totals?.map((t) => t.toFixed(0)).join(' / ') ?? '?'}</span>`;
    } else if (currentStep.winningTeam !== null && currentStep.winningTeam !== undefined) {
      const wt = Number(currentStep.winningTeam);
      const label = wt === 0 ? 'Team A' : 'Team B';
      statusHTML = `<span style="color: ${teamColorFor(wt)};">${label} wins &mdash; ${totals?.[wt]?.toFixed(0) ?? '?'} pts (vs ${totals?.[1 - wt]?.toFixed(0) ?? '?'})</span>`;
    } else {
      statusHTML = '<span>Game over</span>';
    }
  } else if (currentStep.activeSeat !== null && currentStep.activeSeat !== undefined) {
    const seat = currentStep.activeSeat;
    statusHTML = `<span>Active seat: <span style="font-weight:700;">${seat}</span> &mdash; ${playerNames[seat]} &amp; ${playerNames[seat + PLAYERS_PER_TEAM]} play simultaneously</span>`;
  } else {
    statusHTML = '<span>Setting up...</span>';
  }
  if (currentStep.moveNumber !== null && currentStep.episodeLength !== null) {
    statusHTML += `<span class="annotation">move ${currentStep.moveNumber}/${currentStep.episodeLength}</span>`;
  }
  statusContainer.innerHTML = statusHTML;
}
