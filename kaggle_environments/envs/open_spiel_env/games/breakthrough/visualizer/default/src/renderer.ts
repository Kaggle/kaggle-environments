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
    // Match lines like "8bbbbbbbb" or "1wwwwwwww"
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

  // Find squares that changed
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

  // The moved piece: it disappeared from one square and appeared on another
  // In a standard move, one piece disappears and the same type appears elsewhere
  if (appeared.length >= 1 && disappeared.length >= 1) {
    // Find the from-square: the square where the moving piece disappeared
    // Find the to-square: the square where the moving piece appeared
    const movingPiece = appeared[0].piece;
    const fromSquare = disappeared.find((d) => d.piece === movingPiece);
    const toSquare = appeared[0];

    if (fromSquare) {
      diff.from = { row: fromSquare.row, col: fromSquare.col };
    }
    diff.to = { row: toSquare.row, col: toSquare.col };

    // Check for capture: an opponent piece disappeared from the to-square
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
  text: '#e0e0e0',
  label: '#718096',
  accentBlack: '#4fc3f7', // Blue accent for P1/Black
  accentWhite: '#ff8a65', // Orange accent for P2/White
  moveFrom: 'rgba(255, 215, 0, 0.30)', // Gold at 30% opacity
  moveTo: '#ffd700', // Gold border for to-square
  captureMarker: '#ef4444', // Red for capture marker
  panelBg: '#16213e',
  panelBorder: '#0f3460',
};

const COL_LABELS = 'abcdefgh';

const INFO_PANEL_HEIGHT = 52;

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  parent.innerHTML = `
    <div class="renderer-container">
      <canvas></canvas>
      <div class="status-bar"></div>
    </div>
  `;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const statusBar = parent.querySelector('.status-bar') as HTMLDivElement;
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

  // Parse previous step for move diff and capture detection
  let prevState: BreakthroughState | null = null;
  if (step > 0 && steps[step - 1]) {
    const prevObs = getObservationString(steps[step - 1]);
    prevState = parseObservation(prevObs);
  }

  c.fillStyle = '#1a1a2e';
  c.fillRect(0, 0, width, height);

  if (!state) {
    c.fillStyle = '#fff';
    c.font = '16px sans-serif';
    c.textAlign = 'center';
    c.fillText('Waiting for game data...', width / 2, height / 2);
    return;
  }

  const bs = state.boardSize;
  const terminal = isTerminal(currentStep);
  const cp = getCurrentPlayer(currentStep);
  const pieces = countPieces(state.board);
  const prevPieces = prevState ? countPieces(prevState.board) : null;
  const moveDiff = computeMoveDiff(prevState?.board ?? null, state.board);

  // --- Draw info panel at the top ---
  const panelY = 4;
  const panelH = INFO_PANEL_HEIGHT;
  const panelX = 10;
  const panelW = width - 20;

  // Panel background
  c.fillStyle = COLORS.panelBg;
  c.beginPath();
  roundRect(c, panelX, panelY, panelW, panelH, 8);
  c.fill();
  c.strokeStyle = COLORS.panelBorder;
  c.lineWidth = 1.5;
  c.beginPath();
  roundRect(c, panelX, panelY, panelW, panelH, 8);
  c.stroke();

  // Turn indicator accent stripe at top of panel
  const accentColor = terminal ? '#888' : cp === 0 ? COLORS.accentBlack : COLORS.accentWhite;
  c.fillStyle = accentColor;
  c.beginPath();
  // Draw a thin accent bar at the top of the panel, clipped to rounded corners
  roundRectTop(c, panelX, panelY, panelW, 5, 8);
  c.fill();

  // Turn / game-over text
  const panelCenterY = panelY + panelH / 2 + 2;
  c.textBaseline = 'middle';
  c.textAlign = 'center';

  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over -- Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over -- Black wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over -- White wins!';
    c.fillStyle = '#fff';
    c.font = 'bold 14px sans-serif';
    c.fillText(msg, width / 2, panelCenterY - 8);
  } else {
    const turnLabel = cp === 0 ? 'Black' : 'White';
    c.fillStyle = accentColor;
    c.font = 'bold 13px sans-serif';
    c.fillText(`${turnLabel}'s Turn (P${cp + 1})`, width / 2, panelCenterY - 8);
  }

  // Piece counts
  const blackCountChanged = prevPieces !== null && pieces.black < prevPieces.black;
  const whiteCountChanged = prevPieces !== null && pieces.white < prevPieces.white;

  const countY = panelCenterY + 10;
  c.font = '12px sans-serif';

  // Black count (left of center)
  const blackLabel = `Black (P1): ${pieces.black} pieces`;
  const whiteLabel = `White (P2): ${pieces.white} pieces`;
  const separator = '   |   ';
  const fullText = blackLabel + separator + whiteLabel;
  const fullWidth = c.measureText(fullText).width;
  const startX = width / 2 - fullWidth / 2;

  // Draw black count - highlight if capture occurred (black lost a piece)
  if (blackCountChanged) {
    c.fillStyle = COLORS.captureMarker;
    c.font = 'bold 12px sans-serif';
  } else {
    c.fillStyle = COLORS.accentBlack;
    c.font = '12px sans-serif';
  }
  c.textAlign = 'left';
  c.fillText(blackLabel, startX, countY);

  // Separator
  const sepX = startX + c.measureText(blackLabel).width;
  c.fillStyle = '#555';
  c.font = '12px sans-serif';
  c.fillText(separator, sepX, countY);

  // Draw white count - highlight if capture occurred (white lost a piece)
  const whiteX = sepX + c.measureText(separator).width;
  if (whiteCountChanged) {
    c.fillStyle = COLORS.captureMarker;
    c.font = 'bold 12px sans-serif';
  } else {
    c.fillStyle = COLORS.accentWhite;
    c.font = '12px sans-serif';
  }
  c.fillText(whiteLabel, whiteX, countY);

  // --- Board layout (shifted down for info panel) ---
  const margin = 30;
  const availableHeight = height - INFO_PANEL_HEIGHT - 10;
  const maxBoardPx = Math.min(width - margin * 2, availableHeight - margin * 2 - 30, 600);
  const cellSize = maxBoardPx / bs;
  const boardPx = cellSize * bs;
  const ox = (width - boardPx) / 2;
  const oy = INFO_PANEL_HEIGHT + 10 + (availableHeight - boardPx) / 2 - 10;

  // Draw board squares
  for (let row = 0; row < bs; row++) {
    for (let col = 0; col < bs; col++) {
      const light = (row + col) % 2 === 0;
      c.fillStyle = light ? COLORS.lightSquare : COLORS.darkSquare;
      c.fillRect(ox + col * cellSize, oy + row * cellSize, cellSize, cellSize);
    }
  }

  // --- Last move highlighting (from-square overlay) ---
  if (moveDiff.from) {
    c.fillStyle = COLORS.moveFrom;
    c.fillRect(ox + moveDiff.from.col * cellSize, oy + moveDiff.from.row * cellSize, cellSize, cellSize);
  }

  // --- Last move highlighting (to-square overlay, lighter) ---
  if (moveDiff.to) {
    c.fillStyle = 'rgba(255, 215, 0, 0.18)';
    c.fillRect(ox + moveDiff.to.col * cellSize, oy + moveDiff.to.row * cellSize, cellSize, cellSize);
  }

  // Board border
  c.strokeStyle = '#4a3728';
  c.lineWidth = 2;
  c.strokeRect(ox, oy, boardPx, boardPx);

  // Draw pieces
  const pieceR = cellSize * 0.38;
  for (let row = 0; row < state.board.length; row++) {
    for (let col = 0; col < state.board[row].length; col++) {
      const cell = state.board[row][col];
      if (cell === 'b' || cell === 'w') {
        const cx = ox + col * cellSize + cellSize / 2;
        const cy = oy + row * cellSize + cellSize / 2;

        // Shadow
        c.fillStyle = 'rgba(0,0,0,0.3)';
        c.beginPath();
        c.arc(cx + 2, cy + 2, pieceR, 0, Math.PI * 2);
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

        // Highlight ring on to-square piece
        if (moveDiff.to && moveDiff.to.row === row && moveDiff.to.col === col) {
          c.strokeStyle = COLORS.moveTo;
          c.lineWidth = 2.5;
          c.beginPath();
          c.arc(cx, cy, pieceR + 3, 0, Math.PI * 2);
          c.stroke();
        }
      }
    }
  }

  // --- Capture marker on to-square ---
  if (moveDiff.isCapture && moveDiff.to) {
    const capCx = ox + moveDiff.to.col * cellSize + cellSize - 8;
    const capCy = oy + moveDiff.to.row * cellSize + 8;
    const markerR = Math.max(6, cellSize * 0.12);

    // Red circle background
    c.fillStyle = COLORS.captureMarker;
    c.beginPath();
    c.arc(capCx, capCy, markerR, 0, Math.PI * 2);
    c.fill();

    // "x" text
    c.fillStyle = '#fff';
    c.font = `bold ${Math.max(8, markerR * 1.4)}px sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText('x', capCx, capCy + 0.5);
  }

  // Labels
  c.fillStyle = COLORS.label;
  c.font = `${Math.max(10, cellSize * 0.25)}px sans-serif`;
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

  // Status bar (kept for compatibility)
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over - Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over - Black wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over - White wins!';
    statusBar.textContent = msg;
  } else {
    statusBar.textContent = `${cp === 0 ? 'Black' : 'White'}'s turn (Player ${cp + 1})`;
  }
}

// --- Canvas helper: rounded rectangle path ---
function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// --- Canvas helper: rounded rectangle top portion (for accent bar) ---
function roundRectTop(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
