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
// 'O' or 'o' = player 1, '@' or 'x' = player 2
// '.' = empty

interface PentagoState {
  board: number[][]; // 6x6, 0=empty, 1=player1, 2=player2
}

function parseObservation(obsString: string): PentagoState | null {
  if (!obsString) return null;
  const lines = obsString.split('\n');
  const board: number[][] = [];

  for (const line of lines) {
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
  boardStroke: '#3c3b37',
  gridLine: '#a08050',
  quadrantLine: '#654321',
  p1: '#2d3748',
  p1Stroke: '#1a202c',
  p2: '#f7fafc',
  p2Stroke: '#a0aec0',
  text: '#050001',
  textSecondary: '#444343',
  label: '#654321',
  moveHighlight: '#c0392b',
};

// ---- Quadrant extraction & rotation helpers ----

type Quadrant = number[][]; // 3x3

const QUADRANT_OFFSETS: [number, number][] = [
  [0, 0], // top-left
  [0, 3], // top-right
  [3, 0], // bottom-left
  [3, 3], // bottom-right
];

function extractQuadrant(board: number[][], qi: number): Quadrant {
  const [rs, cs] = QUADRANT_OFFSETS[qi];
  return [
    [board[rs][cs], board[rs][cs + 1], board[rs][cs + 2]],
    [board[rs + 1][cs], board[rs + 1][cs + 1], board[rs + 1][cs + 2]],
    [board[rs + 2][cs], board[rs + 2][cs + 1], board[rs + 2][cs + 2]],
  ];
}

function rotateCW(q: Quadrant): Quadrant {
  const out: Quadrant = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[c][2 - r] = q[r][c];
  return out;
}

function rotateCCW(q: Quadrant): Quadrant {
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
  rotatedQuadrant: number;
  rotationDirection: 'cw' | 'ccw' | 'none';
}

function detectMove(prev: PentagoState, cur: PentagoState): MoveInfo | null {
  const diffs: { row: number; col: number; prevVal: number; curVal: number }[] = [];
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      if (prev.board[r][c] !== cur.board[r][c]) {
        diffs.push({ row: r, col: c, prevVal: prev.board[r][c], curVal: cur.board[r][c] });
      }
    }
  }

  if (diffs.length === 0) return null;

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

  for (let qi = 0; qi < 4; qi++) {
    const [rs, cs] = QUADRANT_OFFSETS[qi];
    const curQ = extractQuadrant(cur.board, qi);

    for (const dir of ['cw', 'ccw'] as const) {
      const undone = dir === 'cw' ? rotateCCW(curQ) : rotateCW(curQ);
      let totalNewStones = 0;
      let placedR = -1,
        placedC = -1;
      let mismatch = false;

      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          let preRotVal: number;
          if (r >= rs && r < rs + 3 && c >= cs && c < cs + 3) {
            preRotVal = undone[r - rs][c - cs];
          } else {
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

  // Fallback: no rotation
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

  // Last resort
  for (const d of diffs) {
    if (d.prevVal === 0 && d.curVal === placedPlayer) {
      let rotQ = -1;
      let rotDir: 'cw' | 'ccw' | 'none' = 'none';
      for (let qi = 0; qi < 4; qi++) {
        const prevQ = extractQuadrant(prev.board, qi);
        const curQ = extractQuadrant(cur.board, qi);
        if (!quadrantsEqual(prevQ, curQ)) {
          if (rotQ === -1 || qi !== Math.floor(d.row / 3) * 2 + Math.floor(d.col / 3)) {
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
  c.lineWidth = 2.5;
  c.setLineDash([5, 4]);

  const startAngle = direction === 'cw' ? -Math.PI * 0.6 : -Math.PI * 0.4;
  const endAngle = direction === 'cw' ? Math.PI * 0.6 : Math.PI * 1.6;
  const counterclockwise = direction === 'ccw';

  c.beginPath();
  c.arc(centerX, centerY, radius, startAngle, endAngle, counterclockwise);
  c.stroke();
  c.setLineDash([]);

  // Arrowhead
  const arrowAngle = endAngle;
  const ax = centerX + Math.cos(arrowAngle) * radius;
  const ay = centerY + Math.sin(arrowAngle) * radius;

  const tangent = counterclockwise ? arrowAngle + Math.PI / 2 : arrowAngle - Math.PI / 2;
  const arrowLen = 8;

  c.beginPath();
  c.moveTo(ax, ay);
  c.lineTo(ax + Math.cos(tangent + 0.5) * arrowLen, ay + Math.sin(tangent + 0.5) * arrowLen);
  c.lineTo(ax + Math.cos(tangent - 0.5) * arrowLen, ay + Math.sin(tangent - 0.5) * arrowLen);
  c.closePath();
  c.fillStyle = accentColor;
  c.fill();

  c.restore();
}

// ---- Main renderer ----

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header">
        <span class="player-card sketched-border" id="p1-card">Dark (P1)</span>
        <span class="vs-label">vs</span>
        <span class="player-card sketched-border" id="p2-card">Light (P2)</span>
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

  // Transparent canvas
  c.clearRect(0, 0, width, height);

  if (!state) {
    statusContainer.textContent = 'Waiting for game data...';
    return;
  }

  const terminal = isTerminal(currentStep);
  const cp = getCurrentPlayer(currentStep);
  const pieces = countPieces(state.board);

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

  // =========================================================================
  //  DOM HEADER
  // =========================================================================
  p1Card.textContent = `Dark (P1): ${pieces.p1}`;
  p2Card.textContent = `Light (P2): ${pieces.p2}`;

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
  piecesInfo.textContent = `Total: ${pieces.total} / 36`;

  if (moveInfo) {
    const colLabels = 'abcdef';
    const coord = `${colLabels[moveInfo.placedCol]}${moveInfo.placedRow + 1}`;
    const who = moveInfo.placedPlayer === 1 ? 'Dark' : 'Light';
    const quadLabels = ['TL', 'TR', 'BL', 'BR'];
    let rotStr = '';
    if (moveInfo.rotatedQuadrant >= 0 && moveInfo.rotationDirection !== 'none') {
      const qLabel = quadLabels[moveInfo.rotatedQuadrant];
      const dirLabel = moveInfo.rotationDirection === 'cw' ? '\u21BB' : '\u21BA';
      rotStr = `, ${qLabel} ${dirLabel}`;
    }
    moveInfoEl.textContent = `${who} placed ${coord}${rotStr}`;
  } else {
    moveInfoEl.textContent = '';
  }

  // =========================================================================
  //  BOARD RENDERING (canvas)
  // =========================================================================
  const margin = 40;
  const maxBoardPx = Math.min(width - margin * 2, height - margin * 2 - 10, 500);
  const cellSize = maxBoardPx / 6;
  const boardPx = cellSize * 6;
  const ox = (width - boardPx) / 2;
  const oy = (height - boardPx) / 2;

  // Rotated quadrant highlight tint
  if (moveInfo && moveInfo.rotatedQuadrant >= 0) {
    const qi = moveInfo.rotatedQuadrant;
    const [rs, cs] = QUADRANT_OFFSETS[qi];
    const qx = ox + cs * cellSize;
    const qy = oy + rs * cellSize;
    const qSize = cellSize * 3;
    c.save();
    c.fillStyle = COLORS.moveHighlight;
    c.globalAlpha = 0.06;
    c.fillRect(qx, qy, qSize, qSize);
    c.globalAlpha = 1;
    c.restore();
  }

  // Board background
  c.fillStyle = COLORS.boardBg;
  c.beginPath();
  c.roundRect(ox - 8, oy - 8, boardPx + 16, boardPx + 16, 8);
  c.fill();

  // Board border (dashed)
  c.strokeStyle = COLORS.boardStroke;
  c.lineWidth = 1.5;
  c.setLineDash([5, 4]);
  c.beginPath();
  c.roundRect(ox - 8, oy - 8, boardPx + 16, boardPx + 16, 8);
  c.stroke();
  c.setLineDash([]);

  // Rotated quadrant tint ON the board
  if (moveInfo && moveInfo.rotatedQuadrant >= 0) {
    const qi = moveInfo.rotatedQuadrant;
    const [rs, cs] = QUADRANT_OFFSETS[qi];
    const qx = ox + cs * cellSize;
    const qy = oy + rs * cellSize;
    const qSize = cellSize * 3;
    c.save();
    c.fillStyle = COLORS.moveHighlight;
    c.globalAlpha = 0.08;
    c.fillRect(qx, qy, qSize, qSize);
    c.globalAlpha = 1;
    c.restore();
  }

  // Grid lines
  c.strokeStyle = COLORS.gridLine;
  c.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {
    c.beginPath();
    c.moveTo(ox + i * cellSize, oy);
    c.lineTo(ox + i * cellSize, oy + boardPx);
    c.stroke();
    c.beginPath();
    c.moveTo(ox, oy + i * cellSize);
    c.lineTo(ox + boardPx, oy + i * cellSize);
    c.stroke();
  }

  // Quadrant dividers (thicker lines)
  c.strokeStyle = COLORS.quadrantLine;
  c.lineWidth = 3;
  c.beginPath();
  c.moveTo(ox + 3 * cellSize, oy);
  c.lineTo(ox + 3 * cellSize, oy + boardPx);
  c.stroke();
  c.beginPath();
  c.moveTo(ox, oy + 3 * cellSize);
  c.lineTo(ox + boardPx, oy + 3 * cellSize);
  c.stroke();

  // Board border (solid, over the grid)
  c.strokeStyle = COLORS.quadrantLine;
  c.lineWidth = 2.5;
  c.strokeRect(ox, oy, boardPx, boardPx);

  // Quadrant rotation arrows (decorative)
  const arrowSize = cellSize * 0.3;
  for (const q of [
    { cx: ox + cellSize * 1.5, cy: oy - 16 },
    { cx: ox + cellSize * 4.5, cy: oy - 16 },
    { cx: ox + cellSize * 1.5, cy: oy + boardPx + 16 },
    { cx: ox + cellSize * 4.5, cy: oy + boardPx + 16 },
  ]) {
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
      c.fillStyle = 'rgba(0,0,0,0.2)';
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

  // ---- Last move highlight: placed stone dashed ring ----
  if (moveInfo) {
    const cx = ox + moveInfo.placedCol * cellSize + cellSize / 2;
    const cy = oy + moveInfo.placedRow * cellSize + cellSize / 2;

    c.strokeStyle = COLORS.moveHighlight;
    c.lineWidth = 2.5;
    c.setLineDash([4, 3]);
    c.beginPath();
    c.arc(cx, cy, stoneR + 3, 0, Math.PI * 2);
    c.stroke();
    c.setLineDash([]);
  }

  // ---- Last move highlight: rotation arrow on quadrant ----
  if (moveInfo && moveInfo.rotatedQuadrant >= 0 && moveInfo.rotationDirection !== 'none') {
    drawQuadrantRotationIndicator(
      c,
      ox,
      oy,
      cellSize,
      moveInfo.rotatedQuadrant,
      moveInfo.rotationDirection,
      COLORS.moveHighlight
    );
  }

  // Column labels
  c.fillStyle = COLORS.label;
  c.font = `${Math.max(10, cellSize * 0.25)}px 'Inter', sans-serif`;
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

  // =========================================================================
  //  DOM STATUS CONTAINER
  // =========================================================================
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over \u2014 Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over \u2014 Player 1 (Dark) wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over \u2014 Player 2 (Light) wins!';
    statusContainer.textContent = msg;
    statusContainer.style.fontWeight = '700';
  } else {
    statusContainer.textContent = `${cp === 0 ? 'Dark' : 'Light'}'s turn (Player ${cp + 1})`;
  }
}
