import type { RendererOptions } from '@kaggle-environments/core';
import type { AmazonsBoardState, AmazonsCell, AmazonsStep } from './transformers/amazonsTransformer';

type Board = AmazonsCell[][];

const INK = '#050001';
const SOFT_INK = '#3c3b37';
const X_COLOR = '#050001';
const O_COLOR = '#ffffff';
const O_STROKE = '#050001';
const ARROW_COLOR = '#9a3324';
const HIGHLIGHT_FROM = 'rgba(189, 238, 255, 0.55)';
const HIGHLIGHT_TO = 'rgba(255, 215, 96, 0.65)';
const HIGHLIGHT_SHOOT = 'rgba(154, 51, 36, 0.25)';

function asObservation(step: AmazonsStep | undefined): AmazonsBoardState | null {
  return step?.boardState ?? null;
}

interface BoardDiff {
  fromCell: [number, number] | null;
  toCell: [number, number] | null;
  shootCell: [number, number] | null;
}

function diffBoards(prev: Board | null, curr: Board): BoardDiff {
  const diff: BoardDiff = { fromCell: null, toCell: null, shootCell: null };
  if (!prev) return diff;
  for (let r = 0; r < curr.length; r++) {
    for (let c = 0; c < curr[r].length; c++) {
      const before = prev[r]?.[c];
      const after = curr[r][c];
      if (before === after) continue;
      // amazon left this square
      if ((before === 'X' || before === 'O') && after === '.') {
        diff.fromCell = [r, c];
      }
      // amazon arrived here
      if ((after === 'X' || after === 'O') && before === '.') {
        diff.toCell = [r, c];
      }
      // arrow fired here
      if (after === '#' && before !== '#') {
        diff.shootCell = [r, c];
      }
    }
  }
  return diff;
}

function makePlayerCard(label: string, glyphClass: 'x' | 'o', active: boolean): string {
  return `
    <span class="amazons-player-card sketched-border ${glyphClass} ${active ? 'active' : ''}">
      <span class="amazons-glyph"></span>${label}
    </span>
  `;
}

function findPrevObservation(steps: AmazonsStep[], step: number): AmazonsBoardState | null {
  for (let i = step - 1; i >= 0; i--) {
    const obs = asObservation(steps[i]);
    if (obs) return obs;
  }
  return null;
}

function drawBoard(
  c: CanvasRenderingContext2D,
  width: number,
  height: number,
  obs: AmazonsBoardState,
  diff: BoardDiff
) {
  c.clearRect(0, 0, width, height);
  const size = obs.board_size;
  const margin = Math.max(18, Math.min(width, height) * 0.06);
  const boardPx = Math.min(width, height) - margin * 2;
  const cellPx = boardPx / size;
  const originX = (width - boardPx) / 2;
  const originY = (height - boardPx) / 2;

  // Light cell tint to evoke a checkerboard without breaking the parchment look.
  for (let r = 0; r < size; r++) {
    for (let col = 0; col < size; col++) {
      const x = originX + col * cellPx;
      const y = originY + r * cellPx;
      const dark = (r + col) % 2 === 1;
      c.fillStyle = dark ? 'rgba(60, 59, 55, 0.07)' : 'rgba(255, 255, 255, 0.35)';
      c.fillRect(x, y, cellPx, cellPx);
    }
  }

  // Move highlights (drawn beneath glyphs).
  const paintHighlight = (cell: [number, number] | null, color: string) => {
    if (!cell) return;
    const [r, col] = cell;
    c.fillStyle = color;
    c.fillRect(originX + col * cellPx, originY + r * cellPx, cellPx, cellPx);
  };
  paintHighlight(diff.fromCell, HIGHLIGHT_FROM);
  paintHighlight(diff.toCell, HIGHLIGHT_TO);
  paintHighlight(diff.shootCell, HIGHLIGHT_SHOOT);

  // Sketched grid lines.
  c.strokeStyle = SOFT_INK;
  c.lineWidth = 1;
  c.setLineDash([3, 3]);
  for (let i = 0; i <= size; i++) {
    const off = i * cellPx;
    c.beginPath();
    c.moveTo(originX + off, originY);
    c.lineTo(originX + off, originY + boardPx);
    c.stroke();
    c.beginPath();
    c.moveTo(originX, originY + off);
    c.lineTo(originX + boardPx, originY + off);
    c.stroke();
  }
  c.setLineDash([]);

  // Coordinate labels (1-indexed rows from top, cols a-j).
  c.fillStyle = INK;
  c.font = `${Math.max(10, cellPx * 0.28)}px 'Inter', sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  const colLabels = 'abcdefghij'.slice(0, size).split('');
  for (let i = 0; i < size; i++) {
    c.fillText(colLabels[i], originX + (i + 0.5) * cellPx, originY - margin * 0.45);
    c.fillText(String(i + 1), originX - margin * 0.45, originY + (i + 0.5) * cellPx);
  }

  // Pieces.
  const pieceR = cellPx * 0.36;
  for (let r = 0; r < size; r++) {
    for (let col = 0; col < size; col++) {
      const cell = obs.board[r]?.[col];
      const cx = originX + (col + 0.5) * cellPx;
      const cy = originY + (r + 0.5) * cellPx;
      if (cell === 'X' || cell === 'O') {
        c.beginPath();
        c.arc(cx, cy, pieceR, 0, Math.PI * 2);
        c.fillStyle = cell === 'X' ? X_COLOR : O_COLOR;
        c.fill();
        c.lineWidth = 1.5;
        c.strokeStyle = cell === 'X' ? X_COLOR : O_STROKE;
        c.stroke();
      } else if (cell === '#') {
        // Arrow / burned square: dashed cross-hatch.
        const inset = cellPx * 0.18;
        c.strokeStyle = ARROW_COLOR;
        c.lineWidth = 2;
        c.setLineDash([4, 3]);
        c.beginPath();
        c.moveTo(originX + col * cellPx + inset, originY + r * cellPx + inset);
        c.lineTo(originX + (col + 1) * cellPx - inset, originY + (r + 1) * cellPx - inset);
        c.moveTo(originX + (col + 1) * cellPx - inset, originY + r * cellPx + inset);
        c.lineTo(originX + col * cellPx + inset, originY + (r + 1) * cellPx - inset);
        c.stroke();
        c.setLineDash([]);
      }
    }
  }
}

function describeStatus(obs: AmazonsBoardState, diff: BoardDiff): { primary: string; annotation: string } {
  if (obs.is_terminal) {
    if (obs.winner === 'draw') {
      return { primary: 'Draw', annotation: 'no legal moves remain' };
    }
    if (obs.winner === 'x' || obs.winner === 'o') {
      const name = obs.winner === 'x' ? 'Black (X)' : 'White (O)';
      return { primary: `${name} wins`, annotation: 'opponent has no legal move' };
    }
    return { primary: 'Game over', annotation: '' };
  }
  const next = obs.current_player === 'x' ? 'Black (X)' : obs.current_player === 'o' ? 'White (O)' : obs.current_player;
  const phaseLabel: Record<string, string> = {
    from: 'pick an amazon',
    to: 'move it',
    shoot: 'shoot an arrow',
  };
  const annotationParts: string[] = [];
  if (diff.fromCell) annotationParts.push(`from ${cellName(diff.fromCell)}`);
  if (diff.toCell) annotationParts.push(`to ${cellName(diff.toCell)}`);
  if (diff.shootCell) annotationParts.push(`arrow ${cellName(diff.shootCell)}`);
  return {
    primary: `${next} to ${(obs.phase && phaseLabel[obs.phase]) ?? 'play'}`,
    annotation: annotationParts.join(' · '),
  };
}

function cellName([row, col]: [number, number]): string {
  return `${'abcdefghij'[col] ?? '?'}${row + 1}`;
}

export function renderer(options: RendererOptions<AmazonsStep[]>) {
  const { step, replay, parent } = options;
  const steps = (replay?.steps ?? []) as AmazonsStep[];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="amazons-header"></div>
      <canvas></canvas>
      <div class="amazons-status sketched-border">
        <div class="amazons-status-primary"></div>
        <div class="amazons-annotation"></div>
      </div>
    </div>
  `;
  const header = parent.querySelector('.amazons-header') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const primary = parent.querySelector('.amazons-status-primary') as HTMLDivElement;
  const annotation = parent.querySelector('.amazons-annotation') as HTMLDivElement;
  if (!canvas) return;

  const obs = asObservation(steps[step]);

  if (!obs) {
    header.innerHTML = `${makePlayerCard('Black (X)', 'x', false)}<span style="color:${SOFT_INK}">vs</span>${makePlayerCard('White (O)', 'o', false)}`;
    primary.textContent = 'Waiting for game data…';
    annotation.textContent = '';
    return;
  }

  const xActive = !obs.is_terminal && obs.current_player === 'x';
  const oActive = !obs.is_terminal && obs.current_player === 'o';
  header.innerHTML = `
    ${makePlayerCard('Black (X)', 'x', xActive)}
    <span style="color:${SOFT_INK}">vs</span>
    ${makePlayerCard('White (O)', 'o', oActive)}
  `;

  // Size canvas to its CSS box.
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const prevObs = findPrevObservation(steps, step);
  const diff = diffBoards(prevObs?.board ?? null, obs.board);

  drawBoard(ctx, rect.width, rect.height, obs, diff);

  const { primary: primaryText, annotation: annotationText } = describeStatus(obs, diff);
  primary.textContent = primaryText;
  annotation.textContent = annotationText;
}
