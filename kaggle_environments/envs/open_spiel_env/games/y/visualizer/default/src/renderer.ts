import type { RendererOptions } from '@kaggle-environments/core';
import type { YBoardState, YStep } from './transformers/yTransformer';

const PLAYER_X_COLOR = '#1f77b4';
const PLAYER_O_COLOR = '#d62728';
const CELL_FILL = '#fbf7e8';
const SECONDARY_TEXT = '#444343';
const SKETCH_STROKE = '#3c3b37';

const SQRT3 = Math.sqrt(3);

function colLabel(col: number): string {
  return String.fromCharCode('a'.charCodeAt(0) + col);
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player X' : 'Player O';
}

function drawYBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  board: (string | null)[][],
  boardSize: number,
  highlightCell: { row: number; col: number } | null,
  highlightColor: string
) {
  ctx.clearRect(0, 0, width, height);

  const padding = 28;
  const innerW = Math.max(0, width - padding * 2);
  const innerH = Math.max(0, height - padding * 2);
  // The triangle bounding box: width = boardSize * hexW, height = ((boardSize-1)*1.5 + 2) * size.
  const totalUnitsH = (boardSize - 1) * 1.5 + 2;
  const sizeFromW = innerW / (boardSize * SQRT3);
  const sizeFromH = innerH / totalUnitsH;
  const size = Math.max(6, Math.min(sizeFromW, sizeFromH));
  const hexW = SQRT3 * size;
  const boardW = boardSize * hexW;
  const boardH = (boardSize - 1) * 1.5 * size + 2 * size;
  const xOffset = (width - boardW) / 2;
  const yOffset = (height - boardH) / 2;

  const centerOf = (row: number, col: number) => ({
    x: xOffset + (col + row * 0.5 + 0.5) * hexW,
    y: yOffset + size + row * 1.5 * size,
  });

  const hexPath = (cx: number, cy: number, s: number) => {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i + Math.PI / 6;
      const x = cx + s * Math.cos(angle);
      const y = cy + s * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
  };

  // Three sides of the triangle. In Y, a player wins by linking all three.
  const edgeWidth = Math.max(3, size * 0.18);
  ctx.lineWidth = edgeWidth;
  ctx.lineCap = 'round';
  ctx.setLineDash([]);

  const topL = centerOf(0, 0);
  const topR = centerOf(0, boardSize - 1);
  ctx.strokeStyle = PLAYER_X_COLOR;
  ctx.beginPath();
  ctx.moveTo(topL.x - hexW / 2, topL.y - size * 0.6);
  ctx.lineTo(topR.x + hexW / 2, topR.y - size * 0.6);
  ctx.stroke();

  const botCorner = centerOf(boardSize - 1, 0);
  ctx.strokeStyle = PLAYER_O_COLOR;
  ctx.beginPath();
  ctx.moveTo(topL.x - hexW / 2, topL.y - size * 0.5);
  ctx.lineTo(botCorner.x - hexW / 2, botCorner.y + size * 0.5);
  ctx.stroke();

  ctx.strokeStyle = '#7c7c7c';
  ctx.beginPath();
  ctx.moveTo(topR.x + hexW / 2, topR.y - size * 0.5);
  ctx.lineTo(botCorner.x + hexW / 2, botCorner.y + size * 0.5);
  ctx.stroke();

  // Cells.
  ctx.font = `${Math.round(size * 0.7)}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let r = 0; r < boardSize; r++) {
    const rowLen = boardSize - r;
    for (let c = 0; c < rowLen; c++) {
      const { x, y } = centerOf(r, c);
      const cell = board?.[r]?.[c] ?? null;

      hexPath(x, y, size * 0.95);
      ctx.fillStyle = CELL_FILL;
      ctx.fill();
      ctx.lineWidth = 1.25;
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = SKETCH_STROKE;
      ctx.stroke();
      ctx.setLineDash([]);

      if (cell === 'x') {
        ctx.fillStyle = PLAYER_X_COLOR;
        ctx.fillText('X', x, y + size * 0.04);
      } else if (cell === 'o') {
        ctx.fillStyle = PLAYER_O_COLOR;
        ctx.fillText('O', x, y + size * 0.04);
      }
    }
  }

  if (highlightCell) {
    const { x, y } = centerOf(highlightCell.row, highlightCell.col);
    hexPath(x, y, size * 0.95);
    ctx.lineWidth = Math.max(2.5, size * 0.12);
    ctx.setLineDash([]);
    ctx.strokeStyle = highlightColor;
    ctx.stroke();
  }

  // Coordinate labels.
  ctx.font = `${Math.round(size * 0.32)}px 'Inter', sans-serif`;
  ctx.fillStyle = SECONDARY_TEXT;
  for (let c = 0; c < boardSize; c++) {
    const top = centerOf(0, c);
    ctx.fillText(colLabel(c), top.x, top.y - size - size * 0.25);
  }
  for (let r = 0; r < boardSize; r++) {
    const left = centerOf(r, 0);
    ctx.fillText(`${r + 1}`, left.x - hexW * 0.55 - size * 0.25, left.y);
  }
}

export function renderer(options: RendererOptions<YStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as YStep[];
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
  const observation: YBoardState | null = currentStep?.boardState ?? null;
  if (!observation) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const boardSize = observation.board_size;
  const isTerminal = observation.is_terminal;
  const currentPlayer = observation.current_player;

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const activeIdx = isTerminal ? -1 : currentPlayer === 'x' ? 0 : currentPlayer === 'o' ? 1 : -1;

  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${PLAYER_X_COLOR};">
      ${playerNames[0]} <span style="opacity: 0.7;">(X)</span>
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${PLAYER_O_COLOR};">
      ${playerNames[1]} <span style="opacity: 0.7;">(O)</span>
    </span>
  `;

  // The proxy puts the most recent move on the board state itself.
  // Move parity tells us who placed it: even-index → x (player 0), odd → o.
  const lastActor = observation.move_number > 0 ? (observation.move_number - 1) % 2 : null;
  let lastCell: { row: number; col: number } | null = null;
  if (observation.last_move) {
    const colChar = observation.last_move.charCodeAt(0) - 'a'.charCodeAt(0);
    const rowNum = parseInt(observation.last_move.slice(1), 10) - 1;
    if (colChar >= 0 && rowNum >= 0) lastCell = { row: rowNum, col: colChar };
  }
  const highlightColor = lastActor === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;

  const aspect = (boardSize * SQRT3) / ((boardSize - 1) * 1.5 + 2);

  const sizeAndDraw = () => {
    const wrapRect = wrap.getBoundingClientRect();
    const availW = wrapRect.width;
    const availH = wrapRect.height;
    if (availW <= 0 || availH <= 0) return;
    let cssW = Math.min(availW, availH * aspect);
    let cssH = cssW / aspect;
    cssW = Math.max(1, Math.floor(cssW));
    cssH = Math.max(1, Math.floor(cssH));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = cssW;
    canvas.height = cssH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawYBoard(ctx, cssW, cssH, observation.board, boardSize, lastCell, highlightColor);
  };

  requestAnimationFrame(sizeAndDraw);

  let statusHTML = '';
  if (isTerminal) {
    if (observation.winner === 'x') {
      statusHTML = `<span style="color: ${PLAYER_X_COLOR};">${playerNames[0]} (X) wins!</span>`;
    } else if (observation.winner === 'o') {
      statusHTML = `<span style="color: ${PLAYER_O_COLOR};">${playerNames[1]} (O) wins!</span>`;
    } else {
      statusHTML = `<span>Game over: ${observation.winner ?? 'finished'}</span>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;
  }
  if (observation.last_move) {
    const moverColor = lastActor === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">${observation.last_move}</span></span>`;
  }
  statusHTML += `<span class="annotation">move ${observation.move_number}</span>`;
  statusContainer.innerHTML = statusHTML;
}
