import type { RendererOptions } from '@kaggle-environments/core';

// Pentago observation string format:
//     > t     u <
//     a b c d e f
// v 1 . . O @ . . v
// s 2 . . . . . . v
//   3 . . . . . .
//   4 . . O . . .
// z 5 . . . . . . w
// ^ 6 . . . O . . ^
//     > y     x <
//
// 'O' or 'o' = player 1 (sometimes colored), '@' or 'x' = player 2
// '.' = empty

interface PentagoState {
  board: number[][]; // 6x6, 0=empty, 1=player1, 2=player2
}

function parseObservation(obsString: string): PentagoState | null {
  if (!obsString) return null;
  const lines = obsString.split('\n');
  const board: number[][] = [];

  for (const line of lines) {
    // Match board rows: lines containing row numbers 1-6 followed by pieces
    const match = line.match(/\d\s+([.OoXx@]\s+[.OoXx@]\s+[.OoXx@]\s+[.OoXx@]\s+[.OoXx@]\s+[.OoXx@])/);
    if (match) {
      const cells = match[1].trim().split(/\s+/);
      const row: number[] = [];
      for (const cell of cells) {
        if (cell === '.' || cell === '+') row.push(0);
        else if (cell === 'O' || cell === 'o') row.push(1);
        else if (cell === '@' || cell === 'x' || cell === 'X') row.push(2);
        else row.push(0);
      }
      if (row.length === 6) board.push(row);
    }
  }

  if (board.length !== 6) return null;
  return { board };
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

const COLORS = {
  boardBg: '#dcb871',
  boardStroke: '#8b6914',
  gridLine: '#a08050',
  quadrantLine: '#654321',
  p1: '#2d3748',
  p1Stroke: '#1a202c',
  p2: '#f7fafc',
  p2Stroke: '#a0aec0',
  p1Accent: '#4fc3f7',
  p2Accent: '#ff8a65',
  text: '#e0e0e0',
  infoBg: '#16213e',
  infoBorder: '#0f3460',
};

// ---- Quadrant extraction & rotation helpers ----

type Quadrant = number[][]; // 3x3

/** Quadrant definitions: [rowStart, colStart] */
const QUADRANT_OFFSETS: [number, number][] = [
  [0, 0], // top-left
  [0, 3], // top-right
  [3, 0], // bottom-left
  [3, 3], // bottom-right
];
// Quadrant labels for reference: ['TL', 'TR', 'BL', 'BR']

function extractQuadrant(board: number[][], qi: number): Quadrant {
  const [rs, cs] = QUADRANT_OFFSETS[qi];
  return [
    [board[rs][cs], board[rs][cs + 1], board[rs][cs + 2]],
    [board[rs + 1][cs], board[rs + 1][cs + 1], board[rs + 1][cs + 2]],
    [board[rs + 2][cs], board[rs + 2][cs + 1], board[rs + 2][cs + 2]],
  ];
}

function rotateCW(q: Quadrant): Quadrant {
  // (r,c) -> (c, 2-r)
  const out: Quadrant = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[c][2 - r] = q[r][c];
  return out;
}

function rotateCCW(q: Quadrant): Quadrant {
  // (r,c) -> (2-c, r)
  const out: Quadrant = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[2 - c][r] = q[r][c];
  return out;
}

function quadrantsEqual(a: Quadrant, b: Quadrant): boolean {
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) if (a[r][c] !== b[r][c]) return false;
  return true;
}

interface MoveInfo {
  placedRow: number;
  placedCol: number;
  placedPlayer: number;
  rotatedQuadrant: number; // 0-3, or -1 if unknown
  rotationDirection: 'cw' | 'ccw' | 'none';
}

/**
 * Detect what changed between previous and current board.
 * A Pentago move = place a stone + rotate one quadrant.
 * We find the placed stone and detect which quadrant was rotated and in which direction.
 */
function detectMove(prev: PentagoState, cur: PentagoState): MoveInfo | null {
  // Step 1: Find placed stone.
  // After placement + rotation the board changed. We need to account for the
  // rotation affecting the position of the placed stone. Strategy:
  // - For each quadrant, check if rotating prev quadrant CW or CCW makes it
  //   match the current quadrant (with exactly one new stone added).

  // First try: find cells that differ
  const diffs: { row: number; col: number; prevVal: number; curVal: number }[] = [];
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      if (prev.board[r][c] !== cur.board[r][c]) {
        diffs.push({ row: r, col: c, prevVal: prev.board[r][c], curVal: cur.board[r][c] });
      }
    }
  }

  if (diffs.length === 0) return null;

  // Count pieces to know who placed
  let prevP1 = 0,
    prevP2 = 0,
    curP1 = 0,
    curP2 = 0;
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      if (prev.board[r][c] === 1) prevP1++;
      if (prev.board[r][c] === 2) prevP2++;
      if (cur.board[r][c] === 1) curP1++;
      if (cur.board[r][c] === 2) curP2++;
    }
  }
  const placedPlayer = curP1 > prevP1 ? 1 : curP2 > prevP2 ? 2 : 0;
  if (placedPlayer === 0) return null;

  // Strategy: For each quadrant, try to determine if it was rotated.
  // Build a "hypothetical previous board after placing the stone" and then
  // check if rotating a quadrant of that board yields the current board.

  // Simple approach: find the newly placed stone (a cell that is empty in prev
  // but occupied in cur and NOT just moved by rotation). We try each quadrant rotation.

  // For each quadrant qi, try CW and CCW:
  //   Rotate cur quadrant qi in the OPPOSITE direction -> get "pre-rotation" state
  //   Compare with prev board: exactly one new stone should appear (the placement)
  for (let qi = 0; qi < 4; qi++) {
    const [rs, cs] = QUADRANT_OFFSETS[qi];
    const curQ = extractQuadrant(cur.board, qi);

    for (const dir of ['cw', 'ccw'] as const) {
      // Undo the rotation on the current quadrant to get pre-rotation state
      const undone = dir === 'cw' ? rotateCCW(curQ) : rotateCW(curQ);

      // Now compare the undone quadrant with prevQ for the rotated quadrant,
      // and compare remaining quadrants directly.
      // Count total differences across the whole board.
      let totalNewStones = 0;
      let placedR = -1,
        placedC = -1;
      let mismatch = false;

      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          let preRotVal: number;
          if (r >= rs && r < rs + 3 && c >= cs && c < cs + 3) {
            // Inside the rotated quadrant: use undone value
            preRotVal = undone[r - rs][c - cs];
          } else {
            // Outside the rotated quadrant: current value should match prev
            preRotVal = cur.board[r][c];
          }

          if (preRotVal !== prev.board[r][c]) {
            if (prev.board[r][c] === 0 && preRotVal === placedPlayer) {
              totalNewStones++;
              placedR = r;
              placedC = c;
            } else {
              mismatch = true;
              break;
            }
          }
        }
        if (mismatch) break;
      }

      if (!mismatch && totalNewStones === 1 && placedR >= 0) {
        return {
          placedRow: placedR,
          placedCol: placedC,
          placedPlayer,
          rotatedQuadrant: qi,
          rotationDirection: dir,
        };
      }
    }
  }

  // Fallback: maybe the rotation was a no-op (quadrant is symmetric or no rotation).
  // In that case, look for a single new stone with no rotation.
  {
    let newStones = 0;
    let placedR = -1,
      placedC = -1;
    let onlyNewStone = true;
    for (const d of diffs) {
      if (d.prevVal === 0 && d.curVal === placedPlayer) {
        newStones++;
        placedR = d.row;
        placedC = d.col;
      } else {
        onlyNewStone = false;
      }
    }
    if (onlyNewStone && newStones === 1) {
      return {
        placedRow: placedR,
        placedCol: placedC,
        placedPlayer,
        rotatedQuadrant: -1,
        rotationDirection: 'none',
      };
    }
  }

  // Last resort: just pick the first new stone we can find
  for (const d of diffs) {
    if (d.prevVal === 0 && d.curVal === placedPlayer) {
      // Try to find rotation by checking quadrant-level changes
      let rotQ = -1;
      let rotDir: 'cw' | 'ccw' | 'none' = 'none';
      for (let qi = 0; qi < 4; qi++) {
        const prevQ = extractQuadrant(prev.board, qi);
        const curQ = extractQuadrant(cur.board, qi);
        if (!quadrantsEqual(prevQ, curQ)) {
          // Check if this is the rotated quadrant
          if (rotQ === -1 || qi !== Math.floor(d.row / 3) * 2 + Math.floor(d.col / 3)) {
            // Check rotation direction
            if (quadrantsEqual(rotateCW(prevQ), curQ)) {
              rotQ = qi;
              rotDir = 'cw';
            } else if (quadrantsEqual(rotateCCW(prevQ), curQ)) {
              rotQ = qi;
              rotDir = 'ccw';
            }
          }
        }
      }
      return {
        placedRow: d.row,
        placedCol: d.col,
        placedPlayer,
        rotatedQuadrant: rotQ,
        rotationDirection: rotDir,
      };
    }
  }

  return null;
}

function countPieces(board: number[][]): { p1: number; p2: number; total: number } {
  let p1 = 0,
    p2 = 0;
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      if (board[r][c] === 1) p1++;
      else if (board[r][c] === 2) p2++;
    }
  }
  return { p1, p2, total: p1 + p2 };
}

// ---- Drawing helpers ----

function drawRotationArrow(c: CanvasRenderingContext2D, cx: number, cy: number, size: number) {
  c.save();
  c.strokeStyle = '#a08050';
  c.lineWidth = 1.5;
  c.beginPath();
  c.arc(cx, cy, size, Math.PI * 0.2, Math.PI * 1.8);
  c.stroke();

  // Arrowhead
  const angle = Math.PI * 1.8;
  const ax = cx + Math.cos(angle) * size;
  const ay = cy + Math.sin(angle) * size;
  c.beginPath();
  c.moveTo(ax, ay);
  c.lineTo(ax + 4, ay - 3);
  c.lineTo(ax + 1, ay + 4);
  c.closePath();
  c.fillStyle = '#a08050';
  c.fill();
  c.restore();
}

/**
 * Draw a curved rotation arrow on a quadrant border.
 * The arrow indicates CW or CCW rotation for the given quadrant.
 */
function drawQuadrantRotationIndicator(
  c: CanvasRenderingContext2D,
  ox: number,
  oy: number,
  cellSize: number,
  qi: number,
  direction: 'cw' | 'ccw',
  accentColor: string
) {
  const [rs, cs] = QUADRANT_OFFSETS[qi];
  const qx = ox + cs * cellSize;
  const qy = oy + rs * cellSize;
  const qSize = cellSize * 3;
  const centerX = qx + qSize / 2;
  const centerY = qy + qSize / 2;
  const radius = qSize * 0.38;

  c.save();
  c.strokeStyle = accentColor;
  c.lineWidth = 3;
  c.globalAlpha = 0.9;

  const startAngle = direction === 'cw' ? -Math.PI * 0.6 : -Math.PI * 0.4;
  const endAngle = direction === 'cw' ? Math.PI * 0.6 : Math.PI * 1.6;
  const counterclockwise = direction === 'ccw';

  c.beginPath();
  c.arc(centerX, centerY, radius, startAngle, endAngle, counterclockwise);
  c.stroke();

  // Arrowhead at the end of the arc
  const arrowAngle = counterclockwise ? endAngle : endAngle;
  const ax = centerX + Math.cos(arrowAngle) * radius;
  const ay = centerY + Math.sin(arrowAngle) * radius;

  // Tangent direction for arrow
  const tangent = counterclockwise ? arrowAngle + Math.PI / 2 : arrowAngle - Math.PI / 2;
  const arrowLen = 8;

  c.beginPath();
  c.moveTo(ax, ay);
  c.lineTo(ax + Math.cos(tangent + 0.5) * arrowLen, ay + Math.sin(tangent + 0.5) * arrowLen);
  c.lineTo(ax + Math.cos(tangent - 0.5) * arrowLen, ay + Math.sin(tangent - 0.5) * arrowLen);
  c.closePath();
  c.fillStyle = accentColor;
  c.fill();

  c.globalAlpha = 1;
  c.restore();
}

// ---- Main renderer ----

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

  c.fillStyle = '#1a1a2e';
  c.fillRect(0, 0, width, height);

  if (!state) {
    c.fillStyle = '#fff';
    c.font = '16px sans-serif';
    c.textAlign = 'center';
    c.fillText('Waiting for game data...', width / 2, height / 2);
    return;
  }

  // ---- Info panel at top ----
  const infoPanelH = 52;
  const terminal = isTerminal(currentStep);
  const cp = getCurrentPlayer(currentStep);
  const pieces = countPieces(state.board);
  const accentP1 = COLORS.p1Accent;
  const accentP2 = COLORS.p2Accent;
  const turnAccent = cp === 0 ? accentP1 : accentP2;

  // Panel background
  c.fillStyle = COLORS.infoBg;
  c.beginPath();
  c.roundRect(8, 6, width - 16, infoPanelH, 8);
  c.fill();

  // Accent left strip
  c.fillStyle = terminal ? '#888' : turnAccent;
  c.beginPath();
  c.roundRect(8, 6, 5, infoPanelH, [8, 0, 0, 8]);
  c.fill();

  // Turn / game-over text
  c.textBaseline = 'middle';
  c.textAlign = 'left';
  const panelCenterY = 6 + infoPanelH / 2;

  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over -- Draw';
    let msgColor = '#ccc';
    if (rewards[0] > rewards[1]) {
      msg = 'Game Over -- Player 1 (dark) wins!';
      msgColor = accentP1;
    } else if (rewards[1] > rewards[0]) {
      msg = 'Game Over -- Player 2 (light) wins!';
      msgColor = accentP2;
    }
    c.fillStyle = msgColor;
    c.font = 'bold 15px sans-serif';
    c.fillText(msg, 24, panelCenterY - 9);
  } else {
    const turnLabel = cp === 0 ? 'Player 1 (dark)' : 'Player 2 (light)';
    c.fillStyle = turnAccent;
    c.font = 'bold 15px sans-serif';
    c.fillText(`${turnLabel}'s turn`, 24, panelCenterY - 9);
  }

  // Piece counts line
  c.fillStyle = '#aab';
  c.font = '12px sans-serif';
  const countsText = `Dark: ${pieces.p1}  |  Light: ${pieces.p2}  |  Total: ${pieces.total} / 36`;
  c.fillText(countsText, 24, panelCenterY + 11);

  // Small colored circles as legend on the right side of the panel
  const legendX = width - 24;
  // P2 circle (light)
  c.fillStyle = COLORS.p2;
  c.strokeStyle = COLORS.p2Stroke;
  c.lineWidth = 1;
  c.beginPath();
  c.arc(legendX, panelCenterY - 9, 7, 0, Math.PI * 2);
  c.fill();
  c.stroke();
  c.fillStyle = accentP2;
  c.font = '11px sans-serif';
  c.textAlign = 'right';
  c.fillText('P2', legendX - 12, panelCenterY - 9);

  // P1 circle (dark)
  c.fillStyle = COLORS.p1;
  c.strokeStyle = COLORS.p1Stroke;
  c.lineWidth = 1;
  c.beginPath();
  c.arc(legendX, panelCenterY + 11, 7, 0, Math.PI * 2);
  c.fill();
  c.stroke();
  c.fillStyle = accentP1;
  c.font = '11px sans-serif';
  c.textAlign = 'right';
  c.fillText('P1', legendX - 12, panelCenterY + 11);

  // ---- Detect last move ----
  let moveInfo: MoveInfo | null = null;
  if (step > 0) {
    const prevStep = steps[step - 1];
    const prevObs = getObservationString(prevStep);
    const prevState = parseObservation(prevObs);
    if (prevState) {
      moveInfo = detectMove(prevState, state);
    }
  }

  // ---- Board layout (shifted down to accommodate info panel) ----
  const margin = 40;
  const availableH = height - infoPanelH - 16;
  const maxBoardPx = Math.min(width - margin * 2, availableH - margin * 2 - 30, 500);
  const cellSize = maxBoardPx / 6;
  const boardPx = cellSize * 6;
  const ox = (width - boardPx) / 2;
  const oy = infoPanelH + 16 + (availableH - boardPx) / 2 - 10;

  // ---- Rotated quadrant highlight tint ----
  if (moveInfo && moveInfo.rotatedQuadrant >= 0) {
    const qi = moveInfo.rotatedQuadrant;
    const [rs, cs] = QUADRANT_OFFSETS[qi];
    const qx = ox + cs * cellSize - 4;
    const qy = oy + rs * cellSize - 4;
    const qSize = cellSize * 3 + 8;
    const tintColor = moveInfo.placedPlayer === 1 ? accentP1 : accentP2;
    c.save();
    c.fillStyle = tintColor;
    c.globalAlpha = 0.08;
    c.beginPath();
    c.roundRect(qx, qy, qSize, qSize, 4);
    c.fill();
    c.globalAlpha = 1;
    c.restore();
  }

  // Board background
  c.fillStyle = COLORS.boardBg;
  c.beginPath();
  c.roundRect(ox - 8, oy - 8, boardPx + 16, boardPx + 16, 8);
  c.fill();

  // Rotated quadrant tint ON the board (drawn after board bg so it's visible)
  if (moveInfo && moveInfo.rotatedQuadrant >= 0) {
    const qi = moveInfo.rotatedQuadrant;
    const [rs, cs] = QUADRANT_OFFSETS[qi];
    const qx = ox + cs * cellSize;
    const qy = oy + rs * cellSize;
    const qSize = cellSize * 3;
    const tintColor = moveInfo.placedPlayer === 1 ? accentP1 : accentP2;
    c.save();
    c.fillStyle = tintColor;
    c.globalAlpha = 0.12;
    c.fillRect(qx, qy, qSize, qSize);
    c.globalAlpha = 1;
    c.restore();
  }

  // Grid lines
  c.strokeStyle = COLORS.gridLine;
  c.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {
    // Vertical
    c.beginPath();
    c.moveTo(ox + i * cellSize, oy);
    c.lineTo(ox + i * cellSize, oy + boardPx);
    c.stroke();
    // Horizontal
    c.beginPath();
    c.moveTo(ox, oy + i * cellSize);
    c.lineTo(ox + boardPx, oy + i * cellSize);
    c.stroke();
  }

  // Quadrant dividers (thicker lines at the middle)
  c.strokeStyle = COLORS.quadrantLine;
  c.lineWidth = 3;
  // Vertical middle
  c.beginPath();
  c.moveTo(ox + 3 * cellSize, oy);
  c.lineTo(ox + 3 * cellSize, oy + boardPx);
  c.stroke();
  // Horizontal middle
  c.beginPath();
  c.moveTo(ox, oy + 3 * cellSize);
  c.lineTo(ox + boardPx, oy + 3 * cellSize);
  c.stroke();

  // Board border
  c.strokeStyle = COLORS.quadrantLine;
  c.lineWidth = 2.5;
  c.strokeRect(ox, oy, boardPx, boardPx);

  // Quadrant rotation indicators (decorative arrows above/below board)
  const arrowSize = cellSize * 0.3;
  c.fillStyle = '#a08050';
  c.font = `${Math.max(10, cellSize * 0.22)}px sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'middle';

  const quadrants = [
    { label: 'TL', cx: ox + cellSize * 1.5, cy: oy - 16 },
    { label: 'TR', cx: ox + cellSize * 4.5, cy: oy - 16 },
    { label: 'BL', cx: ox + cellSize * 1.5, cy: oy + boardPx + 16 },
    { label: 'BR', cx: ox + cellSize * 4.5, cy: oy + boardPx + 16 },
  ];
  for (const q of quadrants) {
    drawRotationArrow(c, q.cx, q.cy, arrowSize);
  }

  // Draw stones
  const stoneR = cellSize * 0.38;
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 6; col++) {
      const cell = state.board[row][col];
      if (cell === 0) continue;

      const cx = ox + col * cellSize + cellSize / 2;
      const cy = oy + row * cellSize + cellSize / 2;

      // Shadow
      c.fillStyle = 'rgba(0,0,0,0.25)';
      c.beginPath();
      c.arc(cx + 1.5, cy + 1.5, stoneR, 0, Math.PI * 2);
      c.fill();

      // Stone
      c.fillStyle = cell === 1 ? COLORS.p1 : COLORS.p2;
      c.strokeStyle = cell === 1 ? COLORS.p1Stroke : COLORS.p2Stroke;
      c.lineWidth = 1.5;
      c.beginPath();
      c.arc(cx, cy, stoneR, 0, Math.PI * 2);
      c.fill();
      c.stroke();

      // Shine
      c.fillStyle = cell === 1 ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.35)';
      c.beginPath();
      c.arc(cx - stoneR * 0.2, cy - stoneR * 0.2, stoneR * 0.4, 0, Math.PI * 2);
      c.fill();
    }
  }

  // ---- Last move highlight: placed stone ring/glow ----
  if (moveInfo) {
    const cx = ox + moveInfo.placedCol * cellSize + cellSize / 2;
    const cy = oy + moveInfo.placedRow * cellSize + cellSize / 2;
    const accent = moveInfo.placedPlayer === 1 ? accentP1 : accentP2;

    // Outer glow
    c.save();
    c.shadowColor = accent;
    c.shadowBlur = 12;
    c.strokeStyle = accent;
    c.lineWidth = 3.5;
    c.beginPath();
    c.arc(cx, cy, stoneR + 3, 0, Math.PI * 2);
    c.stroke();
    c.shadowBlur = 0;
    c.restore();

    // Inner bright ring
    c.strokeStyle = accent;
    c.lineWidth = 2.5;
    c.beginPath();
    c.arc(cx, cy, stoneR + 1, 0, Math.PI * 2);
    c.stroke();
  }

  // ---- Last move highlight: rotation arrow on quadrant ----
  if (moveInfo && moveInfo.rotatedQuadrant >= 0 && moveInfo.rotationDirection !== 'none') {
    const accent = moveInfo.placedPlayer === 1 ? accentP1 : accentP2;
    drawQuadrantRotationIndicator(c, ox, oy, cellSize, moveInfo.rotatedQuadrant, moveInfo.rotationDirection, accent);
  }

  // Column labels
  c.fillStyle = '#a08050';
  c.font = `${Math.max(10, cellSize * 0.25)}px sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'top';
  const colLabels = 'abcdef';
  for (let col = 0; col < 6; col++) {
    c.fillText(colLabels[col], ox + col * cellSize + cellSize / 2, oy + boardPx + 6);
  }
  c.textAlign = 'right';
  c.textBaseline = 'middle';
  for (let row = 0; row < 6; row++) {
    c.fillText(String(row + 1), ox - 8, oy + row * cellSize + cellSize / 2);
  }

  // Status bar (kept for compatibility, mirrors info panel)
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over - Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over - Player 1 (Dark) wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over - Player 2 (Light) wins!';
    statusBar.textContent = msg;
  } else {
    statusBar.textContent = `${cp === 0 ? 'Dark' : 'Light'}'s turn (Player ${cp + 1})`;
  }
}
