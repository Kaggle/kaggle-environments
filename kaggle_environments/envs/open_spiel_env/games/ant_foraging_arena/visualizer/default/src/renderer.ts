import type { RendererOptions } from '@kaggle-environments/core';
import type { AntBoardView, ArenaStep } from './transformers/antForagingArenaTransformer';

const ANT_COLORS = ['#9a3324', '#1f4f8b', '#2e7d32', '#7b1fa2'];
const TEAM_COLORS = ['#1f4f8b', '#9a3324'];
const INK = '#050001';
const GRID_LINE = '#3c3b37';
const CELL_FILL = '#fbf7e8';
const NEST_FILL = '#e8d8a8';
const FOOD_COLOR = '#7a4d20';
const SECONDARY_TEXT = '#444343';
const PHEROMONE_FOOD = '218, 90, 60'; // rusty red
const PHEROMONE_NEST = '60, 110, 180'; // muted blue
const PLAYERS_PER_TEAM = 2;

function antColorFor(pid: number): string {
  return ANT_COLORS[pid % ANT_COLORS.length];
}

function teamColorFor(team: number): string {
  return TEAM_COLORS[team % TEAM_COLORS.length];
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return `Ant ${idx}`;
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  board: AntBoardView,
  prevBoard: AntBoardView | null,
  lastActor: number | null
) {
  const size = board.grid_size;
  ctx.clearRect(0, 0, width, height);

  const padding = 8;
  const cellSize = Math.floor(Math.min((width - padding * 2) / size, (height - padding * 2) / size));
  const boardW = cellSize * size;
  const boardH = cellSize * size;
  const xOff = Math.floor((width - boardW) / 2);
  const yOff = Math.floor((height - boardH) / 2);

  // Base cells.
  ctx.fillStyle = CELL_FILL;
  ctx.fillRect(xOff, yOff, boardW, boardH);

  // Pheromone overlays — to-food (red) + to-nest (blue).
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const food = board.pheromone_to_food?.[r]?.[c] ?? 0;
      const nest = board.pheromone_to_nest?.[r]?.[c] ?? 0;
      const x = xOff + c * cellSize;
      const y = yOff + r * cellSize;
      if (food > 0.02) {
        ctx.fillStyle = `rgba(${PHEROMONE_FOOD}, ${Math.min(0.55, food * 0.6)})`;
        ctx.fillRect(x, y, cellSize, cellSize);
      }
      if (nest > 0.02) {
        ctx.fillStyle = `rgba(${PHEROMONE_NEST}, ${Math.min(0.55, nest * 0.6)})`;
        ctx.fillRect(x, y, cellSize, cellSize);
      }
    }
  }

  // Nest cell highlight + crosshatch.
  const [nr, nc] = board.nest_position;
  ctx.fillStyle = NEST_FILL;
  ctx.fillRect(xOff + nc * cellSize, yOff + nr * cellSize, cellSize, cellSize);
  ctx.save();
  ctx.beginPath();
  ctx.rect(xOff + nc * cellSize, yOff + nr * cellSize, cellSize, cellSize);
  ctx.clip();
  ctx.strokeStyle = 'rgba(60, 59, 55, 0.35)';
  ctx.lineWidth = 1;
  const nx = xOff + nc * cellSize;
  const ny = yOff + nr * cellSize;
  for (let k = -cellSize; k < cellSize * 2; k += 4) {
    ctx.beginPath();
    ctx.moveTo(nx + k, ny);
    ctx.lineTo(nx + k + cellSize, ny + cellSize);
    ctx.stroke();
  }
  ctx.restore();

  // Dashed grid lines.
  ctx.strokeStyle = GRID_LINE;
  ctx.setLineDash([2, 3]);
  ctx.lineWidth = 1;
  for (let i = 0; i <= size; i++) {
    const y = yOff + i * cellSize + 0.5;
    ctx.beginPath();
    ctx.moveTo(xOff, y);
    ctx.lineTo(xOff + boardW, y);
    ctx.stroke();
    const x = xOff + i * cellSize + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, yOff);
    ctx.lineTo(x, yOff + boardH);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Nest "N" label.
  ctx.fillStyle = INK;
  ctx.font = `${Math.floor(cellSize * 0.42)}px Inter, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('N', xOff + nc * cellSize + cellSize / 2, yOff + nr * cellSize + cellSize / 2);

  // Food dots.
  for (const [fr, fc] of board.food_positions) {
    const cx = xOff + fc * cellSize + cellSize / 2;
    const cy = yOff + fr * cellSize + cellSize / 2;
    ctx.fillStyle = FOOD_COLOR;
    ctx.beginPath();
    ctx.arc(cx, cy, cellSize * 0.18, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Ants. Offset stacked ants slightly so both remain visible.
  const occupancy: Record<string, number> = {};
  const antPositions = board.ant_positions ?? {};
  const carrying = board.carrying_food ?? {};
  const pids = Object.keys(antPositions)
    .map(Number)
    .sort((a, b) => a - b);
  for (const pid of pids) {
    const pos = antPositions[String(pid)];
    if (!pos) continue;
    const [r, c] = pos;
    const key = `${r},${c}`;
    const stack = occupancy[key] ?? 0;
    occupancy[key] = stack + 1;
    const color = antColorFor(pid);
    const baseX = xOff + c * cellSize + cellSize / 2;
    const baseY = yOff + r * cellSize + cellSize / 2;
    const offset = cellSize * 0.18;
    const cx = baseX + (stack === 0 ? 0 : -offset + (stack - 1) * offset);
    const cy = baseY + (stack === 0 ? 0 : -offset + (stack - 1) * offset);
    const radius = cellSize * 0.28;

    // Movement trail from previous step.
    const prev = prevBoard?.ant_positions?.[String(pid)];
    if (prev && (prev[0] !== r || prev[1] !== c)) {
      const px = xOff + prev[1] * cellSize + cellSize / 2;
      const py = yOff + prev[0] * cellSize + cellSize / 2;
      ctx.strokeStyle = color;
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(cx, cy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Ant body.
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.4;
    ctx.stroke();

    // Carrying-food marker.
    if (carrying[String(pid)]) {
      ctx.fillStyle = FOOD_COLOR;
      ctx.beginPath();
      ctx.arc(cx + radius * 0.6, cy - radius * 0.6, cellSize * 0.12, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Player id label.
    ctx.fillStyle = 'white';
    ctx.font = `700 ${Math.floor(cellSize * 0.28)}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(pid), cx, cy);
  }

  // Highlight ring around the last actor.
  if (lastActor !== null && lastActor !== undefined) {
    const pos = board.ant_positions?.[String(lastActor)];
    if (pos) {
      const [r, c] = pos;
      const x = xOff + c * cellSize;
      const y = yOff + r * cellSize;
      ctx.strokeStyle = antColorFor(lastActor);
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
      ctx.setLineDash([]);
    }
  }

  // Axis labels.
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.font = `${Math.max(9, Math.round(cellSize * 0.26))}px Inter, sans-serif`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  for (let c = 0; c < size; c++) {
    ctx.fillText(`${c}`, xOff + c * cellSize + cellSize / 2, yOff + cellSize * size + cellSize * 0.32);
  }
  ctx.textAlign = 'right';
  for (let r = 0; r < size; r++) {
    ctx.fillText(`${r}`, xOff - cellSize * 0.12, yOff + r * cellSize + cellSize / 2);
  }
}

function renderTeamHeader(
  team: number,
  playerNames: string[],
  activeSeat: number | null,
  isTerminal: boolean,
  board: AntBoardView | null
): string {
  const baseId = team * PLAYERS_PER_TEAM;
  const pids = [baseId, baseId + 1];
  const carrying = board?.carrying_food ?? {};
  const seatHTML = pids
    .map((pid) => {
      const seat = pid % PLAYERS_PER_TEAM;
      const isActive = !isTerminal && activeSeat === seat;
      const isCarrying = carrying[String(pid)];
      const carryHtml = isCarrying ? '<span class="pref">🍃 carrying</span>' : '<span class="pref">searching</span>';
      return `
        <span class="seat ${isActive ? 'active' : ''}" style="color: ${antColorFor(pid)};">
          <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${antColorFor(pid)};border:1px solid ${GRID_LINE};"></span>
          ${playerNames[pid]}
          ${carryHtml}
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

function renderBoardScore(board: AntBoardView): string {
  return `
    <span class="row">
      <span class="swatch" style="background:${FOOD_COLOR};"></span>
      <span style="font-weight:700;">Food delivered:</span>
      ${board.food_collected} / ${board.num_food}
    </span>
  `;
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

  // Header.
  const headerHTML = [0, 1]
    .map((team) =>
      renderTeamHeader(team, playerNames, currentStep.activeSeat, currentStep.isTerminal, currentStep.boards[team])
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

    score.innerHTML = renderBoardScore(board);

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
      drawBoard(ctx, side, side, board, prevBoard, lastActor);
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
      statusHTML = `<span style="color: ${teamColorFor(wt)};">${label} wins &mdash; ${totals?.[wt]?.toFixed(0) ?? '?'} food (vs ${totals?.[1 - wt]?.toFixed(0) ?? '?'})</span>`;
    } else {
      statusHTML = '<span>Game over</span>';
    }
  } else if (currentStep.activeSeat !== null && currentStep.activeSeat !== undefined) {
    const seat = currentStep.activeSeat;
    statusHTML = `<span>Active seat: <span style="font-weight:700;">${seat}</span> &mdash; ${playerNames[seat]} (Team A) &amp; ${playerNames[seat + PLAYERS_PER_TEAM]} (Team B)</span>`;
  } else {
    statusHTML = '<span>Setting up...</span>';
  }
  if (currentStep.moveNumber !== null && currentStep.maxTurns !== null) {
    statusHTML += `<span class="annotation">move ${currentStep.moveNumber}</span>`;
  }
  statusContainer.innerHTML = statusHTML;
}
