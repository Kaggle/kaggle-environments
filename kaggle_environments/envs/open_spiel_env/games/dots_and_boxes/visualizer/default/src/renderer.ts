import type { RendererOptions } from '@kaggle-environments/core';
import type { DotsAndBoxesBoardState, DotsAndBoxesStep } from './transformers/dotsAndBoxesTransformer';

const PLAYER_COLORS: Record<number, string> = {
  1: '#5cb8ff',
  2: '#ff7e6b',
};

const DOT_COLOR = '#dfe7f5';
const EMPTY_LINE_COLOR = 'rgba(255,255,255,0.08)';
const BG_COLOR = '#0a1428';

const colorWithAlpha = (hex: string, alpha: number): string => {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

let prevStep = -1;

export function renderer({ parent, step, replay }: RendererOptions<DotsAndBoxesStep[]>) {
  parent.innerHTML = `
    <div class="renderer-container">
      <div class="player-legend"></div>
      <canvas></canvas>
      <div class="status-bar"></div>
    </div>
  `;
  const legend = parent.querySelector('.player-legend') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const statusBar = parent.querySelector('.status-bar') as HTMLDivElement;
  const ctx = canvas.getContext('2d');
  if (!ctx || !replay) return;

  const steps = (replay.steps ?? []) as DotsAndBoxesStep[];
  const currentStep = steps[step];
  const obs: DotsAndBoxesBoardState | null = currentStep?.boardState ?? null;
  if (!obs) return;

  const isBackStep = step < prevStep;
  prevStep = step;

  const name1 = currentStep?.players?.[0]?.name ?? 'Player 1';
  const name2 = currentStep?.players?.[1]?.name ?? 'Player 2';
  legend.innerHTML = `
    <div class="legend-item">
      <span class="legend-swatch" style="background:${PLAYER_COLORS[1]}"></span>
      <span>${name1}</span>
      <span class="legend-score">(${obs.scores[0]})</span>
    </div>
    <div class="legend-item">
      <span class="legend-swatch" style="background:${PLAYER_COLORS[2]}"></span>
      <span>${name2}</span>
      <span class="legend-score">(${obs.scores[1]})</span>
    </div>
  `;

  const draw = (frame: number) => {
    canvas.width = 0;
    canvas.height = 0;
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width;
    canvas.height = height;

    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, width, height);

    const { num_rows: rows, num_cols: cols } = obs;
    const padding = 32;
    const availW = width - padding * 2;
    const availH = height - padding * 2;
    const cellSize = Math.max(20, Math.min(availW / cols, availH / rows));
    const boardW = cellSize * cols;
    const boardH = cellSize * rows;
    const xOffset = (width - boardW) / 2;
    const yOffset = (height - boardH) / 2;

    const dotRadius = Math.max(3, Math.min(7, cellSize * 0.08));
    const lineThickness = Math.max(3, Math.min(8, cellSize * 0.1));

    const dotX = (c: number) => xOffset + c * cellSize;
    const dotY = (r: number) => yOffset + r * cellSize;

    // Owned boxes (fill).
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const owner = obs.boxes[r][c];
        if (!owner) continue;
        ctx.fillStyle = colorWithAlpha(PLAYER_COLORS[owner], 0.35);
        ctx.fillRect(dotX(c), dotY(r), cellSize, cellSize);
        ctx.fillStyle = colorWithAlpha(PLAYER_COLORS[owner], 0.95);
        ctx.font = `${Math.floor(cellSize * 0.32)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(owner), dotX(c) + cellSize / 2, dotY(r) + cellSize / 2);
      }
    }

    const drawLine = (x1: number, y1: number, x2: number, y2: number, color: string, width: number) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    };

    const isLastAction = (orientation: 'h' | 'v', r: number, c: number): boolean =>
      !!obs.last_action &&
      obs.last_action.orientation === orientation &&
      obs.last_action.row === r &&
      obs.last_action.col === c;

    // Horizontal lines.
    for (let r = 0; r <= rows; r++) {
      for (let c = 0; c < cols; c++) {
        const owner = obs.h_lines[r][c];
        const x1 = dotX(c);
        const x2 = dotX(c + 1);
        const y = dotY(r);
        if (owner) {
          const color = PLAYER_COLORS[owner];
          const isNew = !isBackStep && isLastAction('h', r, c);
          const reveal = isNew ? frame : 1;
          drawLine(x1, y, x1 + (x2 - x1) * reveal, y, color, lineThickness);
        } else {
          drawLine(x1, y, x2, y, EMPTY_LINE_COLOR, lineThickness * 0.5);
        }
      }
    }

    // Vertical lines.
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c <= cols; c++) {
        const owner = obs.v_lines[r][c];
        const x = dotX(c);
        const y1 = dotY(r);
        const y2 = dotY(r + 1);
        if (owner) {
          const color = PLAYER_COLORS[owner];
          const isNew = !isBackStep && isLastAction('v', r, c);
          const reveal = isNew ? frame : 1;
          drawLine(x, y1, x, y1 + (y2 - y1) * reveal, color, lineThickness);
        } else {
          drawLine(x, y1, x, y2, EMPTY_LINE_COLOR, lineThickness * 0.5);
        }
      }
    }

    // Dots.
    for (let r = 0; r <= rows; r++) {
      for (let c = 0; c <= cols; c++) {
        ctx.beginPath();
        ctx.fillStyle = DOT_COLOR;
        ctx.arc(dotX(c), dotY(r), dotRadius, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  };

  const start = Date.now();
  const animate = () => {
    const frame = Math.min((Date.now() - start) / 300, 1);
    draw(frame);
    if (frame < 1) requestAnimationFrame(animate);
  };
  animate();

  // Status bar.
  const isLastStep = step === steps.length - 1;
  if (!isLastStep || !obs.is_terminal) {
    statusBar.innerHTML = '';
  } else {
    let message = currentStep?.winner ?? 'Game Over';
    if (obs.winner === 'draw') {
      message = `Draw ${obs.scores[0]}–${obs.scores[1]}`;
    } else if (obs.winner === '1') {
      message = `${name1} wins ${obs.scores[0]}–${obs.scores[1]}`;
    } else if (obs.winner === '2') {
      message = `${name2} wins ${obs.scores[1]}–${obs.scores[0]}`;
    }
    statusBar.innerHTML = `<div>${escapeHtml(message)}</div>`;
  }
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
