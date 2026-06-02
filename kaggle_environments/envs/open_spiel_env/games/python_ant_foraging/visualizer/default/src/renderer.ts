import type { RendererOptions } from '@kaggle-environments/core';

type AntForagingObservation = {
  grid: string[][];
  grid_size: number;
  num_ants: number;
  num_food: number;
  nest_position: [number, number];
  food_positions: [number, number][];
  ant_positions: [number, number][];
  carrying_food: boolean[];
  pheromone_to_food: number[][];
  pheromone_to_nest: number[][];
  food_collected: number;
  score: number;
  turn: number;
  max_turns: number;
  current_player: number;
  legal_actions: number[];
  action_names: Record<string, string>;
  is_terminal: boolean;
};

const ANT_COLORS = ['#9a3324', '#1f4f8b', '#2e7d32', '#7b1fa2'];
const INK = '#050001';
const GRID_LINE = '#3c3b37';
const CELL_FILL = '#fbf7e8';
const NEST_FILL = '#e8d8a8';
const FOOD_COLOR = '#7a4d20';
const PHEROMONE_FOOD = '218, 90, 60'; // rusty red
const PHEROMONE_NEST = '60, 110, 180'; // muted blue

function getPlayerName(replay: any, idx: number): string {
  const team = replay?.info?.TeamNames?.[idx];
  if (team) return team;
  const agent = replay?.agents?.[idx]?.name;
  if (agent) return agent;
  return `Ant ${idx}`;
}

function parseObservation(step: any): AntForagingObservation | null {
  const raw = step?.[0]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AntForagingObservation;
  } catch {
    return null;
  }
}

function getLastAction(
  replay: any,
  step: number,
  obs: AntForagingObservation
): { antIdx: number; name: string } | null {
  if (step <= 0) return null;
  const steps = replay?.steps ?? [];
  const prev = steps[step - 1];
  if (!prev) return null;
  // Find the player whose action.submission is a valid play action (not -1).
  for (let i = 0; i < prev.length; i++) {
    const submission = prev[i]?.action?.submission;
    if (typeof submission === 'number' && submission >= 0 && submission <= 4) {
      const name = obs.action_names?.[String(submission)] ?? `action ${submission}`;
      return { antIdx: i, name };
    }
  }
  return null;
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: AntForagingObservation,
  prevAntPositions: [number, number][] | null
) {
  ctx.clearRect(0, 0, width, height);

  const size = obs.grid_size;
  const padding = 8;
  const cellSize = Math.floor(Math.min((width - padding * 2) / size, (height - padding * 2) / size));
  const boardW = cellSize * size;
  const boardH = cellSize * size;
  const xOff = Math.floor((width - boardW) / 2);
  const yOff = Math.floor((height - boardH) / 2);

  // Base cells.
  ctx.fillStyle = CELL_FILL;
  ctx.fillRect(xOff, yOff, boardW, boardH);

  // Pheromone overlays — combine to-food (red) and to-nest (blue) per cell.
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const food = obs.pheromone_to_food?.[r]?.[c] ?? 0;
      const nest = obs.pheromone_to_nest?.[r]?.[c] ?? 0;
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

  // Nest cell highlight.
  const [nr, nc] = obs.nest_position;
  ctx.fillStyle = NEST_FILL;
  ctx.fillRect(xOff + nc * cellSize, yOff + nr * cellSize, cellSize, cellSize);
  // Crosshatch on the nest for a sketched-paper look.
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

  // Grid lines (dashed for sketched look).
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

  // Nest label.
  ctx.fillStyle = INK;
  ctx.font = `${Math.floor(cellSize * 0.42)}px Inter, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('N', xOff + nc * cellSize + cellSize / 2, yOff + nr * cellSize + cellSize / 2);

  // Food: small filled dots with a darker outline.
  for (const [fr, fc] of obs.food_positions) {
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

  // Ants. When two ants stack on the same cell, offset them slightly so
  // both remain visible.
  const occupancy: Record<string, number> = {};
  for (let i = 0; i < obs.ant_positions.length; i++) {
    const [r, c] = obs.ant_positions[i];
    const key = `${r},${c}`;
    const stack = occupancy[key] ?? 0;
    occupancy[key] = stack + 1;
    const color = ANT_COLORS[i % ANT_COLORS.length];
    const baseX = xOff + c * cellSize + cellSize / 2;
    const baseY = yOff + r * cellSize + cellSize / 2;
    const offset = cellSize * 0.18;
    const cx = baseX + (stack === 0 ? 0 : -offset + (stack - 1) * offset);
    const cy = baseY + (stack === 0 ? 0 : -offset + (stack - 1) * offset);
    const radius = cellSize * 0.28;

    // Movement trail from previous step (subtle dashed line).
    const prev = prevAntPositions?.[i];
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
    if (obs.carrying_food[i]) {
      ctx.fillStyle = FOOD_COLOR;
      ctx.beginPath();
      ctx.arc(cx + radius * 0.6, cy - radius * 0.6, cellSize * 0.12, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Ant index label inside the body.
    ctx.fillStyle = 'white';
    ctx.font = `700 ${Math.floor(cellSize * 0.28)}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i), cx, cy);
  }

  // Highlight ring around the next-to-act ant.
  if (!obs.is_terminal) {
    const acting = obs.current_player;
    const [ar, ac] = obs.ant_positions[acting];
    const x = xOff + ac * cellSize;
    const y = yOff + ar * cellSize;
    ctx.strokeStyle = ANT_COLORS[acting % ANT_COLORS.length];
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
    ctx.setLineDash([]);
  }
}

export function renderer(options: RendererOptions) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as any[];
  if (!steps.length) {
    parent.innerHTML = '<div>No replay data.</div>';
    return;
  }
  const obs = parseObservation(steps[step]);
  if (!obs) {
    parent.innerHTML = '<div>Waiting for first observation...</div>';
    return;
  }
  const prev = step > 0 ? parseObservation(steps[step - 1]) : null;
  const prevAntPositions = prev?.ant_positions ?? null;
  const lastAction = getLastAction(replay, step, obs);

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="board-wrap"><canvas></canvas></div>
      <div class="status-container"></div>
    </div>
  `;
  const container = parent.querySelector('.renderer-container') as HTMLDivElement;
  const header = container.querySelector('.header') as HTMLDivElement;
  const wrap = container.querySelector('.board-wrap') as HTMLDivElement;
  const canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
  const statusEl = container.querySelector('.status-container') as HTMLDivElement;

  // Header: per-ant cards plus a shared cooperative score pill.
  const playerCards = obs.ant_positions
    .map((_, i) => {
      const name = getPlayerName(replay, i);
      const color = ANT_COLORS[i % ANT_COLORS.length];
      const isActive = !obs.is_terminal && obs.current_player === i;
      const carrying = obs.carrying_food[i] ? '🍃 carrying' : 'searching';
      return `
        <div class="ant-player ${isActive ? 'active' : ''}" style="color: ${color};">
          <div class="player-name">${name}</div>
          <div class="player-sub">${carrying}</div>
        </div>
      `;
    })
    .join('');
  header.innerHTML = `
    ${playerCards}
    <div class="score-pill">Food: ${obs.food_collected} / ${obs.num_food}</div>
  `;

  // Status line.
  let status: string;
  if (obs.is_terminal) {
    const reason =
      obs.food_collected >= obs.num_food ? 'all food collected!' : `time up at turn ${obs.turn}/${obs.max_turns}`;
    status = `Game over — score ${obs.score} (${reason})`;
  } else {
    const actingColor = ANT_COLORS[obs.current_player % ANT_COLORS.length];
    const actingName = getPlayerName(replay, obs.current_player);
    let actionPart = '';
    if (lastAction) {
      const prevColor = ANT_COLORS[lastAction.antIdx % ANT_COLORS.length];
      const prevName = getPlayerName(replay, lastAction.antIdx);
      actionPart = ` — last: <span style="color: ${prevColor}; font-weight: 700;">${prevName}</span> ${lastAction.name}`;
    }
    status = `Turn ${obs.turn}/${obs.max_turns} — acting: <span style="color: ${actingColor}; font-weight: 700;">${actingName}</span>${actionPart}`;
  }
  statusEl.innerHTML = status;

  const sizeAndDraw = () => {
    const rect = wrap.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const size = Math.max(1, Math.floor(Math.min(rect.width, rect.height)));
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawBoard(ctx, size, size, obs, prevAntPositions);
  };
  requestAnimationFrame(sizeAndDraw);
}
