import type { RendererOptions } from '@kaggle-environments/core';
import type { ShogiBoardState, ShogiCell, ShogiHandCounts, ShogiStep } from './transformers/shogiTransformer';

const INK = '#050001';
const SOFT_INK = '#3c3b37';
const SECONDARY_TEXT = '#444343';
const P0_COLOR = '#222222'; // Sente accent
const P1_COLOR = '#b8801f'; // Gote accent
const BOARD_BG = '#f3e6b6'; // Warm wood tone for the board surface
const PROMOTED_COLOR = '#c9381c';
const HIGHLIGHT_FROM = 'rgba(60, 59, 55, 0.55)';
const HIGHLIGHT_TO = 'rgba(189, 238, 255, 0.85)';
const DROP_RING = '#1f6f8b';

// Order pieces are shown in each player's hand. Strongest first.
const HAND_ORDER_SENTE = ['R', 'B', 'G', 'S', 'N', 'L', 'P'];
const HAND_ORDER_GOTE = ['r', 'b', 'g', 's', 'n', 'l', 'p'];

interface ParsedMove {
  isDrop: boolean;
  promote: boolean;
  // For board moves: from/to in board indices. For drops: from is null.
  from: { row: number; col: number } | null;
  to: { row: number; col: number };
  // For drops: the piece type letter (uppercase = Sente, lowercase = Gote).
  dropPiece: string | null;
}

function fileToCol(file: number): number {
  // USI file 9..1 maps to board columns 0..8.
  return 9 - file;
}

function rankToRow(rank: string): number {
  return rank.charCodeAt(0) - 'a'.charCodeAt(0);
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Sente' : 'Gote';
}

function parseMove(move: string | null, currentPlayer: 'b' | 'w'): ParsedMove | null {
  if (!move) return null;
  // Drop: <PIECE>*<file><rank>, e.g. "P*5e".
  if (move.length >= 4 && move[1] === '*') {
    const piece = move[0];
    const file = parseInt(move[2], 10);
    const rank = move[3];
    if (Number.isNaN(file)) return null;
    // The mover is the side that just played, i.e. the OPPOSITE of the new
    // current_player. We use that to attribute uppercase/lowercase to the
    // dropped piece for rendering on the highlighted square.
    const moverIsSente = currentPlayer === 'w';
    const dropPiece = moverIsSente ? piece.toUpperCase() : piece.toLowerCase();
    return {
      isDrop: true,
      promote: false,
      from: null,
      to: { row: rankToRow(rank), col: fileToCol(file) },
      dropPiece,
    };
  }
  // Board move: <file1><rank1><file2><rank2>[+], e.g. "7g7f" or "2c2b+".
  if (move.length < 4) return null;
  const promote = move.endsWith('+');
  const core = promote ? move.slice(0, -1) : move;
  if (core.length !== 4) return null;
  const fromFile = parseInt(core[0], 10);
  const fromRank = core[1];
  const toFile = parseInt(core[2], 10);
  const toRank = core[3];
  if (Number.isNaN(fromFile) || Number.isNaN(toFile)) return null;
  return {
    isDrop: false,
    promote,
    from: { row: rankToRow(fromRank), col: fileToCol(fromFile) },
    to: { row: rankToRow(toRank), col: fileToCol(toFile) },
    dropPiece: null,
  };
}

function isSentePiece(cell: ShogiCell): boolean {
  if (!cell || cell === '.') return false;
  const letter = cell.startsWith('+') ? cell[1] : cell[0];
  return letter === letter.toUpperCase();
}

function isPromoted(cell: ShogiCell): boolean {
  return typeof cell === 'string' && cell.startsWith('+');
}

function pieceLetter(cell: ShogiCell): string {
  if (!cell || cell === '.') return '';
  return cell.startsWith('+') ? cell.slice(1).toUpperCase() : cell.toUpperCase();
}

function drawPiece(
  ctx: CanvasRenderingContext2D,
  cell: ShogiCell,
  cx: number,
  cy: number,
  size: number,
  fontPx: number,
  emphasis?: { ring?: string; ringWidth?: number }
) {
  const sente = isSentePiece(cell);
  const promoted = isPromoted(cell);
  const letter = pieceLetter(cell);
  const half = size / 2;

  // Pentagon shape pointing "forward" (up for Sente, down for Gote).
  ctx.save();
  ctx.translate(cx, cy);
  if (!sente) ctx.rotate(Math.PI);

  ctx.beginPath();
  const tipY = -half * 0.95;
  const shoulderY = -half * 0.45;
  const baseY = half * 0.85;
  const sideX = half * 0.78;
  ctx.moveTo(0, tipY);
  ctx.lineTo(sideX, shoulderY);
  ctx.lineTo(sideX * 0.85, baseY);
  ctx.lineTo(-sideX * 0.85, baseY);
  ctx.lineTo(-sideX, shoulderY);
  ctx.closePath();
  ctx.fillStyle = promoted ? '#f2d27a' : '#fbe7a4';
  ctx.fill();
  ctx.lineWidth = 1.25;
  ctx.strokeStyle = SOFT_INK;
  ctx.stroke();

  ctx.fillStyle = promoted ? PROMOTED_COLOR : INK;
  ctx.font = `700 ${Math.round(fontPx)}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(letter, 0, fontPx * 0.05);
  if (promoted) {
    ctx.font = `700 ${Math.round(fontPx * 0.55)}px 'Inter', sans-serif`;
    ctx.fillText('+', -fontPx * 0.55, -fontPx * 0.35);
  }
  ctx.restore();

  if (emphasis?.ring) {
    ctx.beginPath();
    ctx.arc(cx, cy, half * 1.05, 0, Math.PI * 2);
    ctx.lineWidth = emphasis.ringWidth ?? 2.6;
    ctx.strokeStyle = emphasis.ring;
    ctx.stroke();
  }
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: ShogiBoardState,
  highlight: ParsedMove | null,
  lastActor: 0 | 1 | null
) {
  ctx.clearRect(0, 0, width, height);

  const padding = 28;
  const innerW = Math.max(0, width - padding * 2);
  const innerH = Math.max(0, height - padding * 2);
  const cellSize = Math.max(16, Math.min(innerW / 9, innerH / 9));
  const boardSize = cellSize * 9;
  const originX = (width - boardSize) / 2;
  const originY = (height - boardSize) / 2;

  // Board surface.
  ctx.fillStyle = BOARD_BG;
  ctx.fillRect(originX, originY, boardSize, boardSize);

  // Grid lines.
  ctx.strokeStyle = SOFT_INK;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  for (let i = 1; i < 9; i++) {
    ctx.beginPath();
    ctx.moveTo(originX + i * cellSize, originY);
    ctx.lineTo(originX + i * cellSize, originY + boardSize);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(originX, originY + i * cellSize);
    ctx.lineTo(originX + boardSize, originY + i * cellSize);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  // Outer border solid.
  ctx.lineWidth = 1.5;
  ctx.strokeRect(originX, originY, boardSize, boardSize);

  // Promotion-zone divider dots (between rank c/d and rank f/g).
  ctx.fillStyle = SOFT_INK;
  const dotR = Math.max(2, cellSize * 0.06);
  const dotCols = [3, 6];
  const dotRows = [3, 6];
  for (const dc of dotCols) {
    for (const dr of dotRows) {
      ctx.beginPath();
      ctx.arc(originX + dc * cellSize, originY + dr * cellSize, dotR, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Highlight: tinted destination square + dashed origin square.
  if (highlight) {
    const toX = originX + highlight.to.col * cellSize;
    const toY = originY + highlight.to.row * cellSize;
    ctx.fillStyle = HIGHLIGHT_TO;
    ctx.fillRect(toX, toY, cellSize, cellSize);
    if (highlight.from) {
      const fromX = originX + highlight.from.col * cellSize;
      const fromY = originY + highlight.from.row * cellSize;
      ctx.lineWidth = 1.75;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = HIGHLIGHT_FROM;
      ctx.strokeRect(fromX + 2, fromY + 2, cellSize - 4, cellSize - 4);
      ctx.setLineDash([]);
    }
  }

  // Pieces.
  const pieceSize = cellSize * 0.86;
  const fontPx = cellSize * 0.46;
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const cell = obs.board?.[r]?.[c];
      if (!cell || cell === '.') continue;
      const cx = originX + (c + 0.5) * cellSize;
      const cy = originY + (r + 0.5) * cellSize;
      let emphasis: { ring: string; ringWidth?: number } | undefined;
      if (highlight && highlight.to.row === r && highlight.to.col === c) {
        const actorColor = lastActor === 0 ? P0_COLOR : lastActor === 1 ? P1_COLOR : SOFT_INK;
        emphasis = { ring: highlight.isDrop ? DROP_RING : actorColor, ringWidth: 2.8 };
      }
      drawPiece(ctx, cell, cx, cy, pieceSize, fontPx, emphasis);
    }
  }

  // Coordinate labels: files 9..1 across the top, ranks a..i down the right.
  ctx.fillStyle = SECONDARY_TEXT;
  ctx.font = `${Math.round(cellSize * 0.28)}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  for (let c = 0; c < 9; c++) {
    ctx.fillText(String(9 - c), originX + (c + 0.5) * cellSize, originY - 6);
  }
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < 9; r++) {
    ctx.fillText(String.fromCharCode('a'.charCodeAt(0) + r), originX + boardSize + 6, originY + (r + 0.5) * cellSize);
  }
}

function renderHand(container: HTMLDivElement, label: string, counts: ShogiHandCounts, order: string[]) {
  const entries = order.filter((p) => counts[p] && counts[p] > 0);
  let html = `<div class="hand-label">${label}</div>`;
  if (entries.length === 0) {
    html += `<div class="hand-empty">—</div>`;
  } else {
    for (const piece of entries) {
      const letter = piece.toUpperCase();
      html += `<div class="hand-piece"><span>${letter}</span><span class="hand-count">×${counts[piece]}</span></div>`;
    }
  }
  container.innerHTML = html;
}

export function renderer(options: RendererOptions<ShogiStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as ShogiStep[];
  if (!steps.length) return;

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="body-wrap">
        <div class="hand p1 sketched-border"></div>
        <div class="board-wrap"><canvas></canvas></div>
        <div class="hand p0 sketched-border"></div>
      </div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const wrap = parent.querySelector('.board-wrap') as HTMLDivElement;
  const canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
  const handGote = parent.querySelector('.hand.p1') as HTMLDivElement;
  const handSente = parent.querySelector('.hand.p0') as HTMLDivElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;

  const currentStep = steps[step];
  const obs: ShogiBoardState | null = currentStep?.boardState ?? null;
  if (!obs) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const isTerminal = obs.is_terminal;
  const activeIdx = isTerminal ? -1 : obs.current_player === 'b' ? 0 : obs.current_player === 'w' ? 1 : -1;

  const senteHand = obs.captured?.b ?? {};
  const goteHand = obs.captured?.w ?? {};
  const senteHandTotal = Object.values(senteHand).reduce((a, b) => a + b, 0);
  const goteHandTotal = Object.values(goteHand).reduce((a, b) => a + b, 0);

  header.innerHTML = `
    <span class="player p0 sketched-border ${activeIdx === 0 ? 'active' : ''}">
      <span class="glyph">☗</span>${playerNames[0]} <span class="count">hand ${senteHandTotal}</span>
    </span>
    <span class="vs">vs</span>
    <span class="player p1 sketched-border ${activeIdx === 1 ? 'active' : ''}">
      <span class="glyph">☖</span>${playerNames[1]} <span class="count">hand ${goteHandTotal}</span>
    </span>
  `;

  renderHand(handSente, 'Sente', senteHand, HAND_ORDER_SENTE);
  renderHand(handGote, 'Gote', goteHand, HAND_ORDER_GOTE);

  // The player who just moved is the OPPOSITE of current_player (or whoever
  // played the last move when the game is over).
  const lastActor: 0 | 1 | null = obs.last_move
    ? obs.current_player === 'b'
      ? 1
      : obs.current_player === 'w'
        ? 0
        : null
    : null;
  const highlight = parseMove(obs.last_move, obs.current_player as 'b' | 'w');

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

  let statusHTML = '';
  if (isTerminal) {
    if (obs.winner === 'b') {
      statusHTML = `<span style="color: ${P0_COLOR};">${playerNames[0]} (Sente) wins!</span>`;
    } else if (obs.winner === 'w') {
      statusHTML = `<span style="color: ${P1_COLOR};">${playerNames[1]} (Gote) wins!</span>`;
    } else {
      statusHTML = `<span>Game over: ${obs.winner ?? 'finished'}</span>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? P0_COLOR : P1_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    const turnSide = activeIdx === 0 ? 'Sente' : activeIdx === 1 ? 'Gote' : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span> <span class="annotation">(${turnSide})</span></span>`;
  }
  if (obs.last_move) {
    const moverColor = lastActor === 0 ? P0_COLOR : P1_COLOR;
    const tag = highlight?.isDrop
      ? `<span style="color: ${DROP_RING}; font-weight: 700;"> drop</span>`
      : highlight?.promote
        ? `<span style="color: ${PROMOTED_COLOR}; font-weight: 700;"> promote</span>`
        : '';
    statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">${obs.last_move}</span>${tag}</span>`;
  }
  statusHTML += `<span class="annotation">move ${obs.move_number}</span>`;
  statusContainer.innerHTML = statusHTML;
}
