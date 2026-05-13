import type { RendererOptions } from '@kaggle-environments/core';
import type { CheckersBoardState, CheckersStep } from './transformers/checkersTransformer';

const INK = '#050001';
const SOFT_INK = '#3c3b37';
const SECONDARY_TEXT = '#444343';
const P0_COLOR = '#1f4f8b'; // Player 0 ('o')
const P1_COLOR = '#9a3324'; // Player 1 ('+')
const CELL_LIGHT = '#fbf7e8';
const CELL_DARK = '#c4a66a';

const ROWS = 8;
const COLS = 8;

function colLabel(col: number): string {
  return String.fromCharCode('a'.charCodeAt(0) + col);
}

function rowLabel(rank: number): string {
  return String(rank + 1);
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player 1' : 'Player 2';
}

/** Parse a move string like "a3b4" or "c5e3" into board coordinates. */
function parseMove(
  move: string | null
): { from: { rank: number; col: number }; to: { rank: number; col: number } } | null {
  if (!move || move.length < 4) return null;
  const fromCol = move.charCodeAt(0) - 'a'.charCodeAt(0);
  const fromRank = parseInt(move[1], 10) - 1;
  const toCol = move.charCodeAt(2) - 'a'.charCodeAt(0);
  const toRank = parseInt(move[3], 10) - 1;
  if (Number.isNaN(fromRank) || Number.isNaN(toRank)) return null;
  return {
    from: { rank: fromRank, col: fromCol },
    to: { rank: toRank, col: toCol },
  };
}

function isPlayerPiece(cell: string, player: 0 | 1): boolean {
  if (player === 0) return cell === 'o' || cell === 'O';
  return cell === '+' || cell === '*';
}

function isKing(cell: string): boolean {
  return cell === 'O' || cell === '*';
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: CheckersBoardState,
  highlight: ReturnType<typeof parseMove>,
  lastActor: 0 | 1 | null
) {
  ctx.clearRect(0, 0, width, height);

  const padding = 36;
  const innerW = Math.max(0, width - padding * 2);
  const innerH = Math.max(0, height - padding * 2);
  const cellSize = Math.max(16, Math.min(innerW / COLS, innerH / ROWS));
  const boardW = cellSize * COLS;
  const boardH = cellSize * ROWS;
  const originX = (width - boardW) / 2;
  const originY = (height - boardH) / 2;

  // board[0] = rank 1 (bottom), board[7] = rank 8 (top).
  // On screen: rank 8 at top (row 0), rank 1 at bottom (row 7).
  const rankToScreenRow = (rank: number) => ROWS - 1 - rank;

  // Draw checkerboard cells.
  for (let rank = 0; rank < ROWS; rank++) {
    const screenRow = rankToScreenRow(rank);
    for (let col = 0; col < COLS; col++) {
      const x = originX + col * cellSize;
      const y = originY + screenRow * cellSize;
      ctx.fillStyle = (rank + col) % 2 === 0 ? CELL_LIGHT : CELL_DARK;
      ctx.fillRect(x, y, cellSize, cellSize);
    }
  }

  // Sketched grid lines.
  ctx.strokeStyle = SOFT_INK;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  for (let i = 0; i <= COLS; i++) {
    ctx.beginPath();
    ctx.moveTo(originX + i * cellSize, originY);
    ctx.lineTo(originX + i * cellSize, originY + boardH);
    ctx.stroke();
  }
  for (let i = 0; i <= ROWS; i++) {
    ctx.beginPath();
    ctx.moveTo(originX, originY + i * cellSize);
    ctx.lineTo(originX + boardW, originY + i * cellSize);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Last-move highlight: ghost on FROM, ring on TO.
  if (highlight) {
    const fromScreenRow = rankToScreenRow(highlight.from.rank);
    const toScreenRow = rankToScreenRow(highlight.to.rank);
    const fromX = originX + (highlight.from.col + 0.5) * cellSize;
    const fromY = originY + (fromScreenRow + 0.5) * cellSize;
    const toX = originX + (highlight.to.col + 0.5) * cellSize;
    const toY = originY + (toScreenRow + 0.5) * cellSize;
    const radius = cellSize * 0.36;

    // Ghost on FROM.
    ctx.beginPath();
    ctx.arc(fromX, fromY, radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(245, 241, 226, 0.6)';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = 'rgba(60, 59, 55, 0.45)';
    ctx.stroke();
    ctx.setLineDash([]);

    // Ring on TO.
    ctx.beginPath();
    ctx.arc(toX, toY, radius + 3, 0, Math.PI * 2);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = lastActor === 0 ? P0_COLOR : P1_COLOR;
    ctx.stroke();
  }

  // Draw pieces.
  const radius = cellSize * 0.36;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let rank = 0; rank < ROWS; rank++) {
    const screenRow = rankToScreenRow(rank);
    for (let col = 0; col < COLS; col++) {
      const cell = obs.board?.[rank]?.[col];
      if (!cell || cell === '.') continue;

      const cx = originX + (col + 0.5) * cellSize;
      const cy = originY + (screenRow + 0.5) * cellSize;
      const isP0 = isPlayerPiece(cell, 0);
      const king = isKing(cell);
      const fillColor = isP0 ? '#f5f1e2' : INK;
      const strokeColor = isP0 ? P0_COLOR : P1_COLOR;
      const textColor = isP0 ? P0_COLOR : '#f5f1e2';

      // Outer circle.
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.lineWidth = 1.75;
      ctx.strokeStyle = strokeColor;
      ctx.stroke();

      if (king) {
        // Inner ring for kings.
        ctx.beginPath();
        ctx.arc(cx, cy, radius * 0.7, 0, Math.PI * 2);
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = strokeColor;
        ctx.stroke();

        // Crown marker "K".
        ctx.font = `700 ${Math.round(cellSize * 0.3)}px 'Inter', sans-serif`;
        ctx.fillStyle = textColor;
        ctx.fillText('K', cx, cy + 1);
      } else {
        // Letter for regular pieces.
        const label = isP0 ? 'o' : '+';
        ctx.font = `700 ${Math.round(cellSize * 0.36)}px 'Inter', sans-serif`;
        ctx.fillStyle = textColor;
        ctx.fillText(label, cx, cy + 1);
      }
    }
  }

  // Coordinate labels.
  ctx.font = `${Math.round(cellSize * 0.28)}px 'Inter', sans-serif`;
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.textBaseline = 'top';
  for (let c = 0; c < COLS; c++) {
    ctx.fillText(colLabel(c), originX + (c + 0.5) * cellSize, originY + boardH + 6);
  }
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (let rank = 0; rank < ROWS; rank++) {
    const screenRow = rankToScreenRow(rank);
    ctx.fillText(rowLabel(rank), originX - 6, originY + (screenRow + 0.5) * cellSize);
  }
}

export function renderer(options: RendererOptions<CheckersStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as CheckersStep[];
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
  const obs: CheckersBoardState | null = currentStep?.boardState ?? null;
  if (!obs) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const isTerminal = obs.is_terminal;
  const activeIdx = isTerminal ? -1 : obs.current_player === 'o' ? 0 : obs.current_player === '+' ? 1 : -1;

  // Piece counts.
  const pc = obs.piece_counts ?? { o: 0, '+': 0, O: 0, '*': 0 };
  const p0Total = pc.o + pc.O;
  const p1Total = pc['+'] + pc['*'];

  header.innerHTML = `
    <span class="player p0 sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${P0_COLOR};">
      <span class="glyph"></span>${playerNames[0]} <span style="opacity:0.7;">(o) ${p0Total}</span>
    </span>
    <span class="vs">vs</span>
    <span class="player p1 sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${P1_COLOR};">
      <span class="glyph"></span>${playerNames[1]} <span style="opacity:0.7;">(+) ${p1Total}</span>
    </span>
  `;

  // Determine who made the last move.
  const lastActor: 0 | 1 | null = obs.move_number > 0 ? (((obs.move_number - 1) % 2) as 0 | 1) : null;
  const highlight = parseMove(obs.last_move);

  const sizeAndDraw = () => {
    const wrapRect = wrap.getBoundingClientRect();
    const availW = wrapRect.width;
    const availH = wrapRect.height;
    if (availW <= 0 || availH <= 0) return;
    const cssW = Math.max(1, Math.floor(availW));
    const cssH = Math.max(1, Math.floor(availH));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = cssW;
    canvas.height = cssH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawBoard(ctx, cssW, cssH, obs, highlight, lastActor);
  };

  requestAnimationFrame(sizeAndDraw);

  // Status bar.
  let statusHTML = '';
  if (isTerminal) {
    if (obs.winner === 'o') {
      statusHTML = `<p style="margin: 0;"><span style="color: ${P0_COLOR};">${playerNames[0]} (o) Wins!</span></p>`;
    } else if (obs.winner === '+') {
      statusHTML = `<p style="margin: 0;"><span style="color: ${P1_COLOR};">${playerNames[1]} (+) Wins!</span></p>`;
    } else {
      statusHTML = `<p style="margin: 0;">Draw</p>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? P0_COLOR : P1_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;
  }
  if (obs.last_move) {
    const moverColor = lastActor === 0 ? P0_COLOR : P1_COLOR;
    statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">${obs.last_move}</span></span>`;
  }
  statusHTML += `<span class="annotation">move ${obs.move_number}</span>`;
  statusContainer.innerHTML = statusHTML;
}
