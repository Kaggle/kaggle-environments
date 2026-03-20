import type { RendererOptions } from '@kaggle-environments/core';

// Dots and Boxes observation string uses Unicode box-drawing characters.
// Example (2x2):
// ┌╴ ╶┬───┐
//         │
// ├╴ ╶┼╴ ╶┤
//     │   │
// └╴ ╶┴╴ ╶┘
//
// We parse the grid by looking at horizontal edges (───), vertical edges (│),
// and cell contents (player number 1 or 2, or space for unclaimed).

interface DotsAndBoxesState {
  numRows: number;
  numCols: number;
  hEdges: boolean[][]; // hEdges[row][col] — (numRows+1) x numCols
  vEdges: boolean[][]; // vEdges[row][col] — numRows x (numCols+1)
  boxes: number[][]; // boxes[row][col] — 0=unclaimed, 1=player1, 2=player2
}

// Tracks which player (0 or 1) placed each line
interface LineOwnership {
  hOwner: (number | null)[][]; // hOwner[row][col] — null=unplaced, 0=P1, 1=P2
  vOwner: (number | null)[][]; // vOwner[row][col] — null=unplaced, 0=P1, 1=P2
}

// Identifies a single line on the board
interface LineId {
  type: 'h' | 'v';
  row: number;
  col: number;
}

function parseObservation(obsString: string): DotsAndBoxesState | null {
  if (!obsString) return null;
  const lines = obsString.split('\n');
  if (lines.length < 3) return null;

  const displayLines = lines.filter((l) => l.length > 0);
  const numRows = Math.floor(displayLines.length / 2);
  if (numRows < 1) return null;

  const firstLine = displayLines[0];
  let numCols = 0;
  const cornerPattern = /[┌┬┐├┼┤└┴┘]/g;
  const corners = firstLine.match(cornerPattern);
  if (corners) {
    numCols = corners.length - 1;
  }
  if (numCols < 1) numCols = 2;

  const hEdges: boolean[][] = [];
  const vEdges: boolean[][] = [];
  const boxes: number[][] = [];

  for (let i = 0; i < displayLines.length; i += 2) {
    const line = displayLines[i];
    const row: boolean[] = [];
    for (let col = 0; col < numCols; col++) {
      const segment = extractSegment(line, col, numCols);
      row.push(segment.includes('───') || segment.includes('━'));
    }
    hEdges.push(row);
  }

  for (let i = 1; i < displayLines.length; i += 2) {
    const line = displayLines[i];
    const vRow: boolean[] = [];
    const boxRow: number[] = [];

    for (let col = 0; col <= numCols; col++) {
      const pos = getVerticalEdgePos(line, col, numCols);
      vRow.push(pos);
    }
    vEdges.push(vRow);

    for (let col = 0; col < numCols; col++) {
      const content = getBoxContent(line, col, numCols);
      boxRow.push(content);
    }
    boxes.push(boxRow);
  }

  return { numRows, numCols, hEdges, vEdges, boxes };
}

function extractSegment(line: string, col: number, numCols: number): string {
  const charWidth = Math.max(4, Math.floor((line.length - 1) / numCols));
  const start = col * charWidth;
  const end = Math.min(start + charWidth + 1, line.length);
  return line.substring(start, end);
}

function getVerticalEdgePos(line: string, col: number, numCols: number): boolean {
  const charWidth = Math.max(4, Math.floor(line.length / (numCols + 0.5)));
  const pos = col * charWidth;
  if (pos >= line.length) return false;
  const ch = line[pos];
  return ch === '│' || ch === '┃' || ch === '|';
}

function getBoxContent(line: string, col: number, numCols: number): number {
  const charWidth = Math.max(4, Math.floor(line.length / (numCols + 0.5)));
  const center = col * charWidth + Math.floor(charWidth / 2);
  if (center >= line.length) return 0;
  const ch = line[center];
  if (ch === '1') return 1;
  if (ch === '2') return 2;
  return 0;
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

function findNewLine(prev: DotsAndBoxesState | null, curr: DotsAndBoxesState | null): LineId | null {
  if (!prev || !curr) return null;

  for (let row = 0; row < curr.hEdges.length; row++) {
    for (let col = 0; col < (curr.hEdges[row]?.length ?? 0); col++) {
      const wasFilled = prev.hEdges[row]?.[col] ?? false;
      const isFilled = curr.hEdges[row]?.[col] ?? false;
      if (isFilled && !wasFilled) {
        return { type: 'h', row, col };
      }
    }
  }

  for (let row = 0; row < curr.vEdges.length; row++) {
    for (let col = 0; col < (curr.vEdges[row]?.length ?? 0); col++) {
      const wasFilled = prev.vEdges[row]?.[col] ?? false;
      const isFilled = curr.vEdges[row]?.[col] ?? false;
      if (isFilled && !wasFilled) {
        return { type: 'v', row, col };
      }
    }
  }

  return null;
}

function buildLineOwnership(
  steps: any[],
  upToStep: number,
  numRows: number,
  numCols: number
): { ownership: LineOwnership; lastLine: LineId | null; lastLinePlayer: number | null } {
  const hOwner: (number | null)[][] = [];
  const vOwner: (number | null)[][] = [];

  for (let row = 0; row <= numRows; row++) {
    hOwner.push(new Array(numCols).fill(null));
  }
  for (let row = 0; row < numRows; row++) {
    vOwner.push(new Array(numCols + 1).fill(null));
  }

  let lastLine: LineId | null = null;
  let lastLinePlayer: number | null = null;

  let prevState: DotsAndBoxesState | null = null;
  for (let i = 0; i <= upToStep; i++) {
    const obs = getObservationString(steps[i]);
    const currState = parseObservation(obs);

    if (i > 0 && prevState && currState) {
      const newLine = findNewLine(prevState, currState);
      if (newLine) {
        const player = getCurrentPlayer(steps[i - 1]);
        if (newLine.type === 'h') {
          if (hOwner[newLine.row]) hOwner[newLine.row][newLine.col] = player;
        } else {
          if (vOwner[newLine.row]) vOwner[newLine.row][newLine.col] = player;
        }
        lastLine = newLine;
        lastLinePlayer = player;
      }
    }

    prevState = currState;
  }

  return { ownership: { hOwner, vOwner }, lastLine, lastLinePlayer };
}

const COLORS = {
  dot: '#d1d5db',
  p1: '#4fc3f7',
  p2: '#ff8a65',
  p1Bright: '#80d8ff',
  p2Bright: '#ffab91',
  lineEmpty: '#4b5563',
  boxP1: 'rgba(79, 195, 247, 0.25)',
  boxP2: 'rgba(255, 138, 101, 0.25)',
  boxP1Text: 'rgba(79, 195, 247, 0.85)',
  boxP2Text: 'rgba(255, 138, 101, 0.85)',
  bg: '#28303F',
};

function computeScores(boxes: number[][]): [number, number] {
  let p1 = 0;
  let p2 = 0;
  for (const row of boxes) {
    for (const cell of row) {
      if (cell === 1) p1++;
      else if (cell === 2) p2++;
    }
  }
  return [p1, p2];
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <canvas></canvas>
      <div class="status-pill"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const statusPill = parent.querySelector('.status-pill') as HTMLDivElement;
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

  c.fillStyle = COLORS.bg;
  c.fillRect(0, 0, width, height);

  if (!state) {
    statusPill.textContent = 'Waiting for game data...';
    return;
  }

  const { numRows, numCols, hEdges, vEdges, boxes } = state;
  const terminal = isTerminal(currentStep);
  const cp = getCurrentPlayer(currentStep);
  const [p1Score, p2Score] = computeScores(boxes);

  // Compute previous scores for delta display
  let prevP1Score = 0;
  let prevP2Score = 0;
  if (step > 0) {
    const prevObs = getObservationString(steps[step - 1]);
    const prevState = parseObservation(prevObs);
    if (prevState) {
      [prevP1Score, prevP2Score] = computeScores(prevState.boxes);
    }
  }
  const p1Delta = p1Score - prevP1Score;
  const p2Delta = p2Score - prevP2Score;

  // Build line ownership by walking through all steps up to current
  const { ownership, lastLine } = buildLineOwnership(steps, step, numRows, numCols);

  // --- Header (DOM) ---
  const p1Active = !terminal && cp === 0;
  const p2Active = !terminal && cp === 1;
  const p1DeltaHtml = p1Delta > 0 ? `<span class="delta" style="color: ${COLORS.p1Bright};">+${p1Delta}</span>` : '';
  const p2DeltaHtml = p2Delta > 0 ? `<span class="delta" style="color: ${COLORS.p2Bright};">+${p2Delta}</span>` : '';

  header.innerHTML = `
    <div class="player-score" style="color: ${COLORS.p1};">
      ${p1Active ? `<span class="turn-dot" style="background: ${COLORS.p1};"></span>` : ''}
      <span>P1: ${p1Score}</span>${p1DeltaHtml}
    </div>
    <span class="vs">vs</span>
    <div class="player-score" style="color: ${COLORS.p2};">
      ${p2Active ? `<span class="turn-dot" style="background: ${COLORS.p2};"></span>` : ''}
      <span>P2: ${p2Score}</span>${p2DeltaHtml}
    </div>
  `;

  // --- Layout ---
  const maxGridPx = Math.min(width * 0.9, height * 0.9, 500);
  const cellSize = maxGridPx / Math.max(numRows, numCols);
  const gridW = cellSize * numCols;
  const gridH = cellSize * numRows;
  const ox = (width - gridW) / 2;
  const oy = (height - gridH) / 2;
  const dotR = Math.max(4, cellSize * 0.08);
  const lineW = Math.max(3, cellSize * 0.06);

  // --- Draw boxes ---
  for (let row = 0; row < numRows; row++) {
    for (let col = 0; col < numCols; col++) {
      if (boxes[row] && boxes[row][col]) {
        const owner = boxes[row][col];
        c.fillStyle = owner === 1 ? COLORS.boxP1 : COLORS.boxP2;
        c.fillRect(ox + col * cellSize, oy + row * cellSize, cellSize, cellSize);

        c.fillStyle = owner === 1 ? COLORS.boxP1Text : COLORS.boxP2Text;
        c.font = `bold ${Math.max(12, cellSize * 0.3)}px sans-serif`;
        c.textAlign = 'center';
        c.textBaseline = 'middle';
        c.fillText(owner === 1 ? 'P1' : 'P2', ox + col * cellSize + cellSize / 2, oy + row * cellSize + cellSize / 2);
      }
    }
  }

  // --- Draw horizontal edges ---
  for (let row = 0; row <= numRows; row++) {
    for (let col = 0; col < numCols; col++) {
      const filled = hEdges[row] && hEdges[row][col];
      const isLast = lastLine && lastLine.type === 'h' && lastLine.row === row && lastLine.col === col;

      if (filled) {
        const owner = ownership.hOwner[row]?.[col];
        const color = owner === 1 ? COLORS.p2 : COLORS.p1;
        const brightColor = owner === 1 ? COLORS.p2Bright : COLORS.p1Bright;

        if (isLast) {
          c.save();
          c.shadowColor = brightColor;
          c.shadowBlur = 12;
          c.strokeStyle = brightColor;
          c.lineWidth = lineW * 1.6;
          c.beginPath();
          c.moveTo(ox + col * cellSize + dotR, oy + row * cellSize);
          c.lineTo(ox + (col + 1) * cellSize - dotR, oy + row * cellSize);
          c.stroke();
          c.restore();
        }

        c.strokeStyle = isLast ? brightColor : color;
        c.lineWidth = isLast ? lineW * 1.4 : lineW;
        c.beginPath();
        c.moveTo(ox + col * cellSize + dotR, oy + row * cellSize);
        c.lineTo(ox + (col + 1) * cellSize - dotR, oy + row * cellSize);
        c.stroke();
      } else {
        c.strokeStyle = COLORS.lineEmpty;
        c.lineWidth = lineW * 0.5;
        c.setLineDash([4, 4]);
        c.beginPath();
        c.moveTo(ox + col * cellSize + dotR, oy + row * cellSize);
        c.lineTo(ox + (col + 1) * cellSize - dotR, oy + row * cellSize);
        c.stroke();
        c.setLineDash([]);
      }
    }
  }

  // --- Draw vertical edges ---
  for (let row = 0; row < numRows; row++) {
    for (let col = 0; col <= numCols; col++) {
      const filled = vEdges[row] && vEdges[row][col];
      const isLast = lastLine && lastLine.type === 'v' && lastLine.row === row && lastLine.col === col;

      if (filled) {
        const owner = ownership.vOwner[row]?.[col];
        const color = owner === 1 ? COLORS.p2 : COLORS.p1;
        const brightColor = owner === 1 ? COLORS.p2Bright : COLORS.p1Bright;

        if (isLast) {
          c.save();
          c.shadowColor = brightColor;
          c.shadowBlur = 12;
          c.strokeStyle = brightColor;
          c.lineWidth = lineW * 1.6;
          c.beginPath();
          c.moveTo(ox + col * cellSize, oy + row * cellSize + dotR);
          c.lineTo(ox + col * cellSize, oy + (row + 1) * cellSize - dotR);
          c.stroke();
          c.restore();
        }

        c.strokeStyle = isLast ? brightColor : color;
        c.lineWidth = isLast ? lineW * 1.4 : lineW;
        c.beginPath();
        c.moveTo(ox + col * cellSize, oy + row * cellSize + dotR);
        c.lineTo(ox + col * cellSize, oy + (row + 1) * cellSize - dotR);
        c.stroke();
      } else {
        c.strokeStyle = COLORS.lineEmpty;
        c.lineWidth = lineW * 0.5;
        c.setLineDash([4, 4]);
        c.beginPath();
        c.moveTo(ox + col * cellSize, oy + row * cellSize + dotR);
        c.lineTo(ox + col * cellSize, oy + (row + 1) * cellSize - dotR);
        c.stroke();
        c.setLineDash([]);
      }
    }
  }

  // --- Draw dots ---
  for (let row = 0; row <= numRows; row++) {
    for (let col = 0; col <= numCols; col++) {
      c.fillStyle = COLORS.dot;
      c.beginPath();
      c.arc(ox + col * cellSize, oy + row * cellSize, dotR, 0, Math.PI * 2);
      c.fill();
    }
  }

  // --- Status pill (DOM) ---
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = `Game Over \u2014 Draw (${p1Score} \u2013 ${p2Score})`;
    if (rewards[0] > rewards[1]) msg = `Player 1 Wins! (${p1Score} \u2013 ${p2Score})`;
    else if (rewards[1] > rewards[0]) msg = `Player 2 Wins! (${p1Score} \u2013 ${p2Score})`;
    statusPill.textContent = msg;
    statusPill.style.fontWeight = '700';
  } else {
    statusPill.textContent = `Player ${cp + 1}'s turn`;
  }
}
