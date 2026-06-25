import type { RendererOptions } from '@kaggle-environments/core';

const COLOR_A = '#1f77b4';
const COLOR_B = '#d62728';
const BALL_FILL = '#ffe27a';
const BALL_STROKE = '#6a4b00';
const CELL_LIGHT = '#fbf7e8';
const CELL_DARK = '#e7dfc1';
// Same-lightness pastels of the team colors so both goal zones read as
// equally-saturated team territory without one looking darker than the other.
const GOAL_A_FILL = '#c8dbef';
const GOAL_B_FILL = '#efc8c8';
const SECONDARY_TEXT = '#444343';
const SKETCH_STROKE = '#3c3b37';

type Pos = [number, number] | null;

interface SoccerState {
  board: string[][];
  current_player: string;
  is_terminal: boolean;
  winner: 'A' | 'B' | 'draw' | null;
  player_a_pos: Pos;
  player_b_pos: Pos;
  ball_pos: Pos;
  ball_owner: 'A' | 'B' | null;
}

function parseObservation(step: any): SoccerState | null {
  const raw = step?.[0]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as SoccerState;
  } catch {
    return null;
  }
}

function getLastAction(step: any, playerIdx: number): string | null {
  const info = step?.[playerIdx]?.info;
  const str = info?.actionSubmittedToString;
  if (typeof str === 'string' && str !== '') return str;
  return null;
}

function getPlayerName(replay: any, idx: number): string {
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player A' : 'Player B';
}

function posEqual(a: Pos, b: Pos): boolean {
  if (!a || !b) return false;
  return a[0] === b[0] && a[1] === b[1];
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  cellSize: number
) {
  const headLen = cellSize * 0.28;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const inset = cellSize * 0.32;
  const tx = x2 - inset * Math.cos(angle);
  const ty = y2 - inset * Math.sin(angle);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = Math.max(2, cellSize * 0.06);
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

function drawStandMarker(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, color: string) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2, r * 0.18);
  ctx.setLineDash([r * 0.35, r * 0.35]);
  ctx.beginPath();
  ctx.arc(cx, cy, r * 1.45, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawPlayer(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  color: string,
  label: string,
  hasBall: boolean
) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = Math.max(1, r * 0.14);
  ctx.strokeStyle = SKETCH_STROKE;
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = `700 ${Math.round(r * 1.2)}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, cx, cy + r * 0.05);
  ctx.restore();

  if (hasBall) {
    drawBall(ctx, cx + r * 0.7, cy + r * 0.7, r * 0.45);
  }
}

function drawBall(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = BALL_FILL;
  ctx.fill();
  ctx.lineWidth = Math.max(1, r * 0.22);
  ctx.strokeStyle = BALL_STROKE;
  ctx.stroke();
  ctx.restore();
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  state: SoccerState,
  prevState: SoccerState | null
) {
  const rows = state.board.length;
  const cols = state.board[0]?.length ?? 0;
  ctx.clearRect(0, 0, width, height);

  // Reserve outer padding for grid labels and the vertical "goal zone"
  // text on each side. Horizontal padding has to be larger to fit both.
  const padX = 64;
  const padY = 28;
  const innerW = width - padX * 2;
  const innerH = height - padY * 2;
  const cellSize = Math.floor(Math.min(innerW / cols, innerH / rows));
  const xOffset = (width - cellSize * cols) / 2;
  const yOffset = (height - cellSize * rows) / 2;

  // Pitch cells (checkerboard, with goal squares painted in matched pastel
  // team colors so each goal reads as that team's territory).
  const goalFill = (r: number, c: number): string | null => {
    if (r < 1 || r > 2 || r >= rows) return null;
    if (c === 0) return GOAL_B_FILL; // B scores by stepping left off column 0.
    if (c === cols - 1) return GOAL_A_FILL; // A scores by stepping right off the rightmost column.
    return null;
  };
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = xOffset + c * cellSize;
      const y = yOffset + r * cellSize;
      ctx.fillStyle = goalFill(r, c) ?? ((r + c) % 2 === 0 ? CELL_LIGHT : CELL_DARK);
      ctx.fillRect(x, y, cellSize, cellSize);
    }
  }

  // Rotated "Player N goal zone" labels outside the pitch, beside each goal.
  const goalLabelFont = `600 ${Math.max(10, Math.round(cellSize * 0.2))}px 'Inter', sans-serif`;
  const goalLabelInset = Math.max(10, cellSize * 0.28);
  const goalCenterY = yOffset + 2 * cellSize; // boundary between rows 1 and 2
  ctx.save();
  ctx.font = goalLabelFont;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  // Left side (B / Player 2): reads bottom-to-top.
  ctx.save();
  ctx.translate(xOffset - goalLabelInset, goalCenterY);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = COLOR_B;
  ctx.fillText('goal zone', 0, 0);
  ctx.restore();
  // Right side (A / Player 1): reads top-to-bottom.
  ctx.save();
  ctx.translate(xOffset + cols * cellSize + goalLabelInset, goalCenterY);
  ctx.rotate(Math.PI / 2);
  ctx.fillStyle = COLOR_A;
  ctx.fillText('goal zone', 0, 0);
  ctx.restore();
  ctx.restore();

  // Outer sketched border + grid lines.
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

  // Centre line emphasising the pitch.
  if (cols >= 2) {
    const midX = xOffset + (cols / 2) * cellSize;
    ctx.save();
    ctx.strokeStyle = SKETCH_STROKE;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(midX, yOffset);
    ctx.lineTo(midX, yOffset + cellSize * rows);
    ctx.stroke();
    ctx.restore();
  }

  const playerR = cellSize * 0.32;

  // Loose ball (only when no owner).
  if (state.ball_owner === null && state.ball_pos) {
    const [br, bc] = state.ball_pos;
    drawBall(ctx, xOffset + bc * cellSize + cellSize / 2, yOffset + br * cellSize + cellSize / 2, cellSize * 0.22);
  }

  // Movement arrows + stand markers for each player based on previous step.
  if (prevState) {
    const moves: { prev: Pos; curr: Pos; color: string }[] = [
      { prev: prevState.player_a_pos, curr: state.player_a_pos, color: COLOR_A },
      { prev: prevState.player_b_pos, curr: state.player_b_pos, color: COLOR_B },
    ];
    for (const m of moves) {
      if (!m.prev || !m.curr) continue;
      const px = xOffset + m.prev[1] * cellSize + cellSize / 2;
      const py = yOffset + m.prev[0] * cellSize + cellSize / 2;
      const cx = xOffset + m.curr[1] * cellSize + cellSize / 2;
      const cy = yOffset + m.curr[0] * cellSize + cellSize / 2;
      if (posEqual(m.prev, m.curr)) {
        drawStandMarker(ctx, cx, cy, playerR, m.color);
      } else {
        drawArrow(ctx, px, py, cx, cy, m.color, cellSize);
      }
    }
  }

  // Players on top of everything else.
  if (state.player_a_pos) {
    const [r, c] = state.player_a_pos;
    drawPlayer(
      ctx,
      xOffset + c * cellSize + cellSize / 2,
      yOffset + r * cellSize + cellSize / 2,
      playerR,
      COLOR_A,
      'A',
      state.ball_owner === 'A'
    );
  }
  if (state.player_b_pos) {
    const [r, c] = state.player_b_pos;
    drawPlayer(
      ctx,
      xOffset + c * cellSize + cellSize / 2,
      yOffset + r * cellSize + cellSize / 2,
      playerR,
      COLOR_B,
      'B',
      state.ball_owner === 'B'
    );
  }

  // Coordinate labels — pushed outside the rotated goal-zone text so the two
  // don't visually overlap on rows 1-2.
  const labelOffset = goalLabelInset + Math.max(12, cellSize * 0.3);
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.font = `${Math.max(10, Math.round(cellSize * 0.26))}px 'Inter', sans-serif`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  for (let c = 0; c < cols; c++) {
    ctx.fillText(`${c}`, xOffset + c * cellSize + cellSize / 2, yOffset + cellSize * rows + cellSize * 0.32);
  }
  ctx.textAlign = 'right';
  for (let r = 0; r < rows; r++) {
    ctx.fillText(`${r}`, xOffset - labelOffset, yOffset + r * cellSize + cellSize / 2);
  }
}

export function renderer(options: RendererOptions) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as any[];
  if (!steps.length) return;

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="status-container sketched-border"></div>
      <div class="board-wrap"><canvas></canvas></div>
    </div>
  `;
  const wrap = parent.querySelector('.board-wrap') as HTMLDivElement;
  const canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;

  const idx = Math.max(0, Math.min(step, steps.length - 1));
  const currentStep = steps[idx];
  const prevStep = idx > 0 ? steps[idx - 1] : null;
  const state = parseObservation(currentStep);
  if (!state) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }
  const prevState = prevStep ? parseObservation(prevStep) : null;

  const nameA = getPlayerName(replay, 0);
  const nameB = getPlayerName(replay, 1);

  // --- Canvas ---
  const sizeAndDraw = () => {
    const rect = wrap.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    // 4 rows x 5 cols pitch with padding looks best at roughly 5:4 aspect.
    const aspect = 5 / 4;
    let w = rect.width;
    let h = w / aspect;
    if (h > rect.height) {
      h = rect.height;
      w = h * aspect;
    }
    w = Math.max(1, Math.floor(w));
    h = Math.max(1, Math.floor(h));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawBoard(ctx, w, h, state, prevState);
  };
  requestAnimationFrame(sizeAndDraw);

  // --- Status: round, ball ownership, last moves, terminal outcome. ---
  const round = idx; // step 0 is the setup pre-game; round N corresponds to step N.
  let statusHTML = '';
  if (state.is_terminal) {
    if (state.winner === 'draw') {
      statusHTML = `<span>Draw &mdash; horizon reached</span>`;
    } else if (state.winner === 'A') {
      statusHTML = `<span style="color: ${COLOR_A};">${nameA} wins!</span>`;
    } else if (state.winner === 'B') {
      statusHTML = `<span style="color: ${COLOR_B};">${nameB} wins!</span>`;
    } else {
      statusHTML = '<span>Game over</span>';
    }
  } else {
    let ballText: string;
    if (state.ball_owner === 'A') {
      ballText = `<span style="color: ${COLOR_A}; font-weight: 700;">${nameA}</span> has the ball`;
    } else if (state.ball_owner === 'B') {
      ballText = `<span style="color: ${COLOR_B}; font-weight: 700;">${nameB}</span> has the ball`;
    } else {
      ballText = 'Ball is loose';
    }
    const moveA = prevStep ? (getLastAction(currentStep, 0) ?? getLastAction(prevStep, 0)) : null;
    const moveB = prevStep ? (getLastAction(currentStep, 1) ?? getLastAction(prevStep, 1)) : null;
    const moveParts: string[] = [];
    if (moveA) moveParts.push(`<span style="color:${COLOR_A};font-weight:700;">A</span> ${moveA}`);
    if (moveB) moveParts.push(`<span style="color:${COLOR_B};font-weight:700;">B</span> ${moveB}`);
    const movesText = moveParts.length ? ` &middot; last: ${moveParts.join(', ')}` : '';
    statusHTML = `<span>Round ${round} &middot; ${ballText}${movesText}</span>`;
  }
  statusContainer.innerHTML = statusHTML;
}
