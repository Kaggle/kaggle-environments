import type { RendererOptions } from '@kaggle-environments/core';

// Breakthrough observation string format:
// 8bbbbbbbb
// 7bb.bbbbb
// ...
// 1wwwwwwww
//  abcdefgh

interface BreakthroughState {
  board: string[][]; // board[row][col], row 0 = top (row 8)
  boardSize: number;
}

interface MoveDiff {
  from: { row: number; col: number } | null;
  to: { row: number; col: number } | null;
  isCapture: boolean;
}

function parseObservation(obsString: string): BreakthroughState | null {
  if (!obsString) return null;
  const lines = obsString.trim().split('\n');
  const board: string[][] = [];
  let boardSize = 8;

  for (const line of lines) {
    const trimmed = line.trim();
    const match = trimmed.match(/^(\d+)(.+)$/);
    if (match) {
      const chars = match[2].split('');
      board.push(chars);
      boardSize = chars.length;
    }
  }

  if (board.length === 0) return null;
  return { board, boardSize };
}

function getObservationString(step: any): string {
  if (!step || !Array.isArray(step)) return '';
  for (const player of step) {
    const obs = player?.observation?.observationString;
    if (obs) return obs;
  }
  return '';
}

function isTerminal(step: any): boolean {
  if (!step || !Array.isArray(step)) return false;
  return step.some((p: any) => p?.status === 'DONE' || p?.observation?.isTerminal);
}

function getCurrentPlayer(step: any): number {
  if (!step || !Array.isArray(step)) return 0;
  for (const player of step) {
    const cp = player?.observation?.currentPlayer;
    if (cp !== undefined && cp >= 0) return cp;
  }
  return 0;
}

function getRewards(step: any): [number, number] {
  if (!step || !Array.isArray(step)) return [0, 0];
  return [step[0]?.reward ?? 0, step[1]?.reward ?? 0];
}

function countPieces(board: string[][]): { black: number; white: number } {
  let black = 0;
  let white = 0;
  for (const row of board) {
    for (const cell of row) {
      if (cell === 'b') black++;
      else if (cell === 'w') white++;
    }
  }
  return { black, white };
}

function computeMoveDiff(prevBoard: string[][] | null, currBoard: string[][]): MoveDiff {
  const diff: MoveDiff = { from: null, to: null, isCapture: false };
  if (!prevBoard) return diff;

  const appeared: { row: number; col: number; piece: string }[] = [];
  const disappeared: { row: number; col: number; piece: string }[] = [];

  const rows = Math.min(prevBoard.length, currBoard.length);
  for (let row = 0; row < rows; row++) {
    const cols = Math.min(prevBoard[row].length, currBoard[row].length);
    for (let col = 0; col < cols; col++) {
      const prev = prevBoard[row][col];
      const curr = currBoard[row][col];
      if (prev !== curr) {
        if (prev !== '.' && (curr === '.' || curr !== prev)) {
          disappeared.push({ row, col, piece: prev });
        }
        if (curr !== '.' && (prev === '.' || prev !== curr)) {
          appeared.push({ row, col, piece: curr });
        }
      }
    }
  }

  if (appeared.length >= 1 && disappeared.length >= 1) {
    const movingPiece = appeared[0].piece;
    const fromSquare = disappeared.find((d) => d.piece === movingPiece);
    const toSquare = appeared[0];

    if (fromSquare) {
      diff.from = { row: fromSquare.row, col: fromSquare.col };
    }
    diff.to = { row: toSquare.row, col: toSquare.col };

    const capturedPiece = disappeared.find(
      (d) => d.piece !== movingPiece && d.row === toSquare.row && d.col === toSquare.col
    );
    if (capturedPiece) {
      diff.isCapture = true;
    }
  }

  return diff;
}

const COLORS = {
  lightSquare: '#f0d9b5',
  darkSquare: '#b58863',
  blackPiece: '#2d3748',
  blackPieceStroke: '#1a202c',
  whitePiece: '#f7fafc',
  whitePieceStroke: '#a0aec0',
  text: '#050001',
  label: '#444343',
  moveFrom: 'rgba(192, 57, 43, 0.20)',
  moveTo: '#c0392b',
  captureMarker: '#c0392b',
  boardBorder: '#3c3b37',
};

const COL_LABELS = 'abcdefgh';

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header">
        <span class="player-card sketched-border" id="p1-card">Black (P1)</span>
        <span class="vs-label">vs</span>
        <span class="player-card sketched-border" id="p2-card">White (P2)</span>
      </div>
      <div class="info-row">
        <span class="pieces-info"></span>
        <span class="move-info"></span>
      </div>
      <canvas></canvas>
      <div class="status-container sketched-border"></div>
    </div>
  `;

  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const p1Card = parent.querySelector('#p1-card') as HTMLSpanElement;
  const p2Card = parent.querySelector('#p2-card') as HTMLSpanElement;
  const piecesInfo = parent.querySelector('.pieces-info') as HTMLSpanElement;
  const moveInfoEl = parent.querySelector('.move-info') as HTMLSpanElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;
  if (!canvas || !replay) return;

  canvas.width = 0;
  canvas.height = 0;
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = width;
  canvas.height = height;

  const c = canvas.getContext('2d');
  if (!c) return;

  const currentStep = steps[step];
  const obsString = getObservationString(currentStep);
  const state = parseObservation(obsString);

  let prevState: BreakthroughState | null = null;
  if (step > 0 && steps[step - 1]) {
    const prevObs = getObservationString(steps[step - 1]);
    prevState = parseObservation(prevObs);
  }

  // Transparent canvas
  c.clearRect(0, 0, width, height);

  if (!state) {
    statusContainer.textContent = 'Waiting for game data...';
    return;
  }

  const bs = state.boardSize;
  const terminal = isTerminal(currentStep);
  const cp = getCurrentPlayer(currentStep);
  const pieces = countPieces(state.board);
  const prevPieces = prevState ? countPieces(prevState.board) : null;
  const moveDiff = computeMoveDiff(prevState?.board ?? null, state.board);

  // =========================================================================
  //  DOM HEADER
  // =========================================================================
  p1Card.textContent = `Black (P1): ${pieces.black}`;
  p2Card.textContent = `White (P2): ${pieces.white}`;

  if (!terminal && cp === 0) {
    p1Card.classList.add('active');
  } else {
    p1Card.classList.remove('active');
  }
  if (!terminal && cp === 1) {
    p2Card.classList.add('active');
  } else {
    p2Card.classList.remove('active');
  }

  // =========================================================================
  //  DOM INFO ROW
  // =========================================================================
  const blackChanged = prevPieces !== null && pieces.black < prevPieces.black;
  const whiteChanged = prevPieces !== null && pieces.white < prevPieces.white;

  if (blackChanged) {
    piecesInfo.textContent = `Black lost a piece`;
  } else if (whiteChanged) {
    piecesInfo.textContent = `White lost a piece`;
  } else {
    piecesInfo.textContent = '';
  }

  if (moveDiff.to) {
    const fromStr = moveDiff.from ? `${COL_LABELS[moveDiff.from.col]}${bs - moveDiff.from.row}` : '?';
    const toStr = `${COL_LABELS[moveDiff.to.col]}${bs - moveDiff.to.row}`;
    const captureStr = moveDiff.isCapture ? ' (capture)' : '';
    moveInfoEl.textContent = `${fromStr} \u2192 ${toStr}${captureStr}`;
  } else {
    moveInfoEl.textContent = '';
  }

  // =========================================================================
  //  BOARD RENDERING (canvas)
  // =========================================================================
  const margin = 30;
  const maxBoardPx = Math.min(width - margin * 2, height - margin * 2 - 10, 500);
  const cellSize = maxBoardPx / bs;
  const boardPx = cellSize * bs;
  const ox = (width - boardPx) / 2;
  const oy = (height - boardPx) / 2;

  // Draw board squares
  for (let row = 0; row < bs; row++) {
    for (let col = 0; col < bs; col++) {
      const light = (row + col) % 2 === 0;
      c.fillStyle = light ? COLORS.lightSquare : COLORS.darkSquare;
      c.fillRect(ox + col * cellSize, oy + row * cellSize, cellSize, cellSize);
    }
  }

  // Last move highlighting (from-square overlay)
  if (moveDiff.from) {
    c.fillStyle = COLORS.moveFrom;
    c.fillRect(ox + moveDiff.from.col * cellSize, oy + moveDiff.from.row * cellSize, cellSize, cellSize);
  }

  // Last move highlighting (to-square overlay)
  if (moveDiff.to) {
    c.fillStyle = 'rgba(192, 57, 43, 0.12)';
    c.fillRect(ox + moveDiff.to.col * cellSize, oy + moveDiff.to.row * cellSize, cellSize, cellSize);
  }

  // Board border (dashed)
  c.strokeStyle = COLORS.boardBorder;
  c.lineWidth = 2;
  c.setLineDash([5, 4]);
  c.strokeRect(ox, oy, boardPx, boardPx);
  c.setLineDash([]);

  // Draw pieces
  const pieceR = cellSize * 0.38;
  for (let row = 0; row < state.board.length; row++) {
    for (let col = 0; col < state.board[row].length; col++) {
      const cell = state.board[row][col];
      if (cell === 'b' || cell === 'w') {
        const cx = ox + col * cellSize + cellSize / 2;
        const cy = oy + row * cellSize + cellSize / 2;

        // Shadow
        c.fillStyle = 'rgba(0,0,0,0.2)';
        c.beginPath();
        c.arc(cx + 1.5, cy + 1.5, pieceR, 0, Math.PI * 2);
        c.fill();

        // Piece
        c.fillStyle = cell === 'b' ? COLORS.blackPiece : COLORS.whitePiece;
        c.strokeStyle = cell === 'b' ? COLORS.blackPieceStroke : COLORS.whitePieceStroke;
        c.lineWidth = 1.5;
        c.beginPath();
        c.arc(cx, cy, pieceR, 0, Math.PI * 2);
        c.fill();
        c.stroke();

        // Shine
        c.fillStyle = cell === 'b' ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.4)';
        c.beginPath();
        c.arc(cx - pieceR * 0.2, cy - pieceR * 0.2, pieceR * 0.45, 0, Math.PI * 2);
        c.fill();

        // Dashed highlight ring on to-square piece
        if (moveDiff.to && moveDiff.to.row === row && moveDiff.to.col === col) {
          c.strokeStyle = COLORS.moveTo;
          c.lineWidth = 2;
          c.setLineDash([4, 3]);
          c.beginPath();
          c.arc(cx, cy, pieceR + 3, 0, Math.PI * 2);
          c.stroke();
          c.setLineDash([]);
        }
      }
    }
  }

  // Capture marker on to-square
  if (moveDiff.isCapture && moveDiff.to) {
    const capCx = ox + moveDiff.to.col * cellSize + cellSize - 8;
    const capCy = oy + moveDiff.to.row * cellSize + 8;
    const markerR = Math.max(6, cellSize * 0.12);

    c.fillStyle = COLORS.captureMarker;
    c.beginPath();
    c.arc(capCx, capCy, markerR, 0, Math.PI * 2);
    c.fill();

    c.fillStyle = '#fff';
    c.font = `bold ${Math.max(8, markerR * 1.4)}px 'Inter', sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText('x', capCx, capCy + 0.5);
  }

  // Labels
  c.fillStyle = COLORS.label;
  c.font = `${Math.max(10, cellSize * 0.25)}px 'Inter', sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'top';
  for (let col = 0; col < bs; col++) {
    c.fillText(COL_LABELS[col], ox + col * cellSize + cellSize / 2, oy + boardPx + 4);
  }
  c.textAlign = 'right';
  c.textBaseline = 'middle';
  for (let row = 0; row < bs; row++) {
    c.fillText(String(bs - row), ox - 6, oy + row * cellSize + cellSize / 2);
  }

  // =========================================================================
  //  DOM STATUS CONTAINER
  // =========================================================================
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over \u2014 Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over \u2014 Black wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over \u2014 White wins!';
    statusContainer.textContent = msg;
    statusContainer.style.fontWeight = '700';
  } else {
    statusContainer.textContent = `${cp === 0 ? 'Black' : 'White'}'s turn (Player ${cp + 1})`;
  }
}
