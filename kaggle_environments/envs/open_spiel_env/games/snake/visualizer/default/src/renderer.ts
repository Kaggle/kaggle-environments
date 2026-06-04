import type { RendererOptions } from '@kaggle-environments/core';

const PLAYER_COLORS = ['#1f4f8b', '#9a3324', '#2e7d32', '#7b1fa2'];
const HEAD_COLORS = ['#4a90e2', '#e74c3c', '#4caf50', '#ab47bc'];
const FOOD_COLOR = '#e6a23c';
const GRID_LINE = '#d6cfb0';
const CELL_FILL = '#fbf7e8';
const INK = '#050001';

type SnakeObservation = {
  board: string[][];
  num_rows: number;
  num_columns: number;
  num_players: number;
  foods?: [number, number][];
  food?: [number, number] | null;
  snakes: {
    player: number;
    body: [number, number][];
    alive: boolean;
    score: number;
  }[];
  scores: number[];
  is_alive: boolean[];
  current_player: number;
  pending_this_turn: number[];
  turn: number;
  is_terminal: boolean;
  winner: number | string | null;
  game_over_reason: string | null;
};

function getPlayerName(replay: any, idx: number): string {
  const team = replay?.info?.TeamNames?.[idx];
  if (team) return team;
  const agent = replay?.agents?.[idx]?.name;
  if (agent) return agent;
  return `Player ${idx}`;
}

function parseObservation(step: any): SnakeObservation | null {
  const raw = step?.[0]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as SnakeObservation;
  } catch {
    return null;
  }
}

function drawBoard(ctx: CanvasRenderingContext2D, width: number, height: number, obs: SnakeObservation) {
  ctx.clearRect(0, 0, width, height);

  const rows = obs.num_rows;
  const cols = obs.num_columns;
  const padding = 8;
  const cellSize = Math.floor(Math.min((width - padding * 2) / cols, (height - padding * 2) / rows));
  const boardW = cellSize * cols;
  const boardH = cellSize * rows;
  const xOff = Math.floor((width - boardW) / 2);
  const yOff = Math.floor((height - boardH) / 2);

  // Cells.
  ctx.fillStyle = CELL_FILL;
  ctx.fillRect(xOff, yOff, boardW, boardH);

  ctx.strokeStyle = GRID_LINE;
  ctx.lineWidth = 1;
  for (let r = 0; r <= rows; r++) {
    const y = yOff + r * cellSize + 0.5;
    ctx.beginPath();
    ctx.moveTo(xOff, y);
    ctx.lineTo(xOff + boardW, y);
    ctx.stroke();
  }
  for (let c = 0; c <= cols; c++) {
    const x = xOff + c * cellSize + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, yOff);
    ctx.lineTo(x, yOff + boardH);
    ctx.stroke();
  }

  // Food.
  const foods: [number, number][] = obs.foods ?? (obs.food ? [obs.food] : []);
  for (const [fr, fc] of foods) {
    const cx = xOff + fc * cellSize + cellSize / 2;
    const cy = yOff + fr * cellSize + cellSize / 2;
    ctx.fillStyle = FOOD_COLOR;
    ctx.beginPath();
    ctx.arc(cx, cy, cellSize * 0.32, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  // Snakes.
  for (const snake of obs.snakes) {
    if (!snake.alive || snake.body.length === 0) continue;
    const bodyColor = PLAYER_COLORS[snake.player % PLAYER_COLORS.length];
    const headColor = HEAD_COLORS[snake.player % HEAD_COLORS.length];
    // Body segments.
    for (let i = 0; i < snake.body.length; i++) {
      const [r, c] = snake.body[i];
      const x = xOff + c * cellSize;
      const y = yOff + r * cellSize;
      const inset = i === 0 ? 1 : 2;
      ctx.fillStyle = i === 0 ? headColor : bodyColor;
      ctx.fillRect(x + inset, y + inset, cellSize - inset * 2, cellSize - inset * 2);
    }
    // Head outline.
    const [hr, hc] = snake.body[0];
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(xOff + hc * cellSize + 1, yOff + hr * cellSize + 1, cellSize - 2, cellSize - 2);
  }
}

export function renderer(options: RendererOptions) {
  const { parent, replay, step } = options;
  const steps = replay?.steps ?? [];
  if (!steps.length) {
    parent.innerHTML = '<div>No replay data.</div>';
    return;
  }
  const obs = parseObservation(steps[step]);
  if (!obs) {
    parent.innerHTML = '<div>Waiting for first observation...</div>';
    return;
  }

  parent.innerHTML = `
    <div class="snake-container">
      <div class="snake-header"></div>
      <div class="snake-board-wrap"><canvas></canvas></div>
      <div class="snake-status"></div>
    </div>
  `;
  const header = parent.querySelector('.snake-header') as HTMLDivElement;
  const wrap = parent.querySelector('.snake-board-wrap') as HTMLDivElement;
  const canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
  const statusEl = parent.querySelector('.snake-status') as HTMLDivElement;

  // Header: one card per player.
  const headerHtml = obs.snakes
    .map((snake) => {
      const name = getPlayerName(replay, snake.player);
      const color = PLAYER_COLORS[snake.player % PLAYER_COLORS.length];
      const isActive = !obs.is_terminal && obs.current_player === snake.player;
      const classes = ['snake-player-card', snake.alive ? 'alive' : 'dead', isActive ? 'active' : ''].join(' ');
      return `
        <div class="${classes}" style="color: ${color};">
          <div class="snake-player-name">${name}</div>
          <div class="snake-player-score">${snake.score.toFixed(0)}</div>
        </div>
      `;
    })
    .join('');
  header.innerHTML = headerHtml;

  // Status line.
  let status = '';
  if (obs.is_terminal) {
    if (obs.winner === 'draw' || obs.winner === null) {
      status = `Game over — draw (turn ${obs.turn})`;
    } else {
      const wname = getPlayerName(replay, obs.winner as number);
      const wcolor = PLAYER_COLORS[(obs.winner as number) % PLAYER_COLORS.length];
      status = `Game over — winner: <span style="color: ${wcolor}; font-weight: 700;">${wname}</span> (turn ${obs.turn})`;
    }
    if (obs.game_over_reason) status += ` — ${obs.game_over_reason}`;
  } else {
    const aname = getPlayerName(replay, obs.current_player);
    const acolor = PLAYER_COLORS[obs.current_player % PLAYER_COLORS.length];
    status = `Turn ${obs.turn} — acting: <span style="color: ${acolor}; font-weight: 700;">${aname}</span>`;
    if (obs.pending_this_turn.length > 0) {
      status += ` (pending: ${obs.pending_this_turn.join(', ')})`;
    }
  }
  statusEl.innerHTML = status;

  // Canvas sizing + draw.
  const aspect = obs.num_columns / obs.num_rows;
  const sizeAndDraw = () => {
    const rect = wrap.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    let cssW = Math.min(rect.width, rect.height * aspect);
    let cssH = cssW / aspect;
    cssW = Math.max(1, Math.floor(cssW));
    cssH = Math.max(1, Math.floor(cssH));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = cssW;
    canvas.height = cssH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawBoard(ctx, cssW, cssH, obs);
  };
  requestAnimationFrame(sizeAndDraw);
}
