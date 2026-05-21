import type { RendererOptions } from '@kaggle-environments/core';
import type { DotsAndBoxesBoardState, DotsAndBoxesStep } from './transformers/dotsAndBoxesTransformer';

const P1_COLOR = '#1f4f8b';
const P2_COLOR = '#9a3324';
const EMPTY_LINE_COLOR = 'rgba(60, 59, 55, 0.18)';
const DOT_COLOR = '#3c3b37';
const AXIS_LABEL_COLOR = '#888378';

function formatMove(action: { orientation: 'h' | 'v'; row: number; col: number }): string {
  const kind = action.orientation === 'h' ? 'horizontal line' : 'vertical line';
  return `${kind} at row ${action.row}, col ${action.col}`;
}

const PLAYER_COLOR: Record<number, string> = {
  1: P1_COLOR,
  2: P2_COLOR,
};

function colorWithAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player 1' : 'Player 2';
}

function drawBoard(ctx: CanvasRenderingContext2D, width: number, height: number, obs: DotsAndBoxesBoardState) {
  ctx.clearRect(0, 0, width, height);

  const { num_rows: rows, num_cols: cols } = obs;
  const padding = 32;
  const innerW = Math.max(0, width - padding * 2);
  const innerH = Math.max(0, height - padding * 2);
  const cellSize = Math.max(12, Math.min(innerW / cols, innerH / rows));
  const boardW = cellSize * cols;
  const boardH = cellSize * rows;
  const originX = (width - boardW) / 2;
  const originY = (height - boardH) / 2;

  const dotX = (c: number) => originX + c * cellSize;
  const dotY = (r: number) => originY + r * cellSize;

  const dotRadius = Math.max(2.5, Math.min(5, cellSize * 0.08));
  const lineThickness = Math.max(2, Math.min(6, cellSize * 0.09));
  const emptyThickness = Math.max(1, lineThickness * 0.4);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const owner = obs.boxes[r][c];
      if (!owner) continue;
      ctx.fillStyle = colorWithAlpha(PLAYER_COLOR[owner], 0.18);
      ctx.fillRect(dotX(c), dotY(r), cellSize, cellSize);
    }
  }

  const last = obs.last_action;
  if (last) {
    const ownerCode = last.player === '1' ? 1 : last.player === '2' ? 2 : 0;
    if (ownerCode) {
      ctx.fillStyle = colorWithAlpha(PLAYER_COLOR[ownerCode], 0.18);
      const markCompletedAround = (boxR: number, boxC: number) => {
        if (boxR < 0 || boxR >= rows || boxC < 0 || boxC >= cols) return;
        if (obs.boxes[boxR][boxC] !== ownerCode) return;
        ctx.save();
        ctx.strokeStyle = PLAYER_COLOR[ownerCode];
        ctx.lineWidth = Math.max(1.5, lineThickness * 0.5);
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(dotX(boxC) + 3, dotY(boxR) + 3, cellSize - 6, cellSize - 6);
        ctx.restore();
      };
      if (last.orientation === 'h') {
        markCompletedAround(last.row - 1, last.col);
        markCompletedAround(last.row, last.col);
      } else {
        markCompletedAround(last.row, last.col - 1);
        markCompletedAround(last.row, last.col);
      }
    }
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const owner = obs.boxes[r][c];
      if (!owner) continue;
      ctx.fillStyle = colorWithAlpha(PLAYER_COLOR[owner], 0.85);
      ctx.font = `700 ${Math.round(cellSize * 0.36)}px 'Inter', sans-serif`;
      ctx.fillText(String(owner), dotX(c) + cellSize / 2, dotY(r) + cellSize / 2);
    }
  }

  const drawLine = (x1: number, y1: number, x2: number, y2: number, color: string, w: number, dashed = false) => {
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = w;
    ctx.lineCap = 'round';
    if (dashed) ctx.setLineDash([4, 4]);
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();
  };

  for (let r = 0; r <= rows; r++) {
    for (let c = 0; c < cols; c++) {
      const owner = obs.h_lines[r][c];
      const x1 = dotX(c);
      const x2 = dotX(c + 1);
      const y = dotY(r);
      if (owner) {
        drawLine(x1, y, x2, y, PLAYER_COLOR[owner], lineThickness);
      } else {
        drawLine(x1, y, x2, y, EMPTY_LINE_COLOR, emptyThickness, true);
      }
    }
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c <= cols; c++) {
      const owner = obs.v_lines[r][c];
      const x = dotX(c);
      const y1 = dotY(r);
      const y2 = dotY(r + 1);
      if (owner) {
        drawLine(x, y1, x, y2, PLAYER_COLOR[owner], lineThickness);
      } else {
        drawLine(x, y1, x, y2, EMPTY_LINE_COLOR, emptyThickness, true);
      }
    }
  }

  if (last) {
    const ownerCode = last.player === '1' ? 1 : last.player === '2' ? 2 : 0;
    if (ownerCode) {
      const color = PLAYER_COLOR[ownerCode];
      const emphasis = lineThickness * 1.5;
      if (last.orientation === 'h') {
        const x1 = dotX(last.col);
        const x2 = dotX(last.col + 1);
        const y = dotY(last.row);
        drawLine(x1, y, x2, y, color, emphasis);
      } else {
        const x = dotX(last.col);
        const y1 = dotY(last.row);
        const y2 = dotY(last.row + 1);
        drawLine(x, y1, x, y2, color, emphasis);
      }
    }
  }

  ctx.fillStyle = DOT_COLOR;
  for (let r = 0; r <= rows; r++) {
    for (let c = 0; c <= cols; c++) {
      ctx.beginPath();
      ctx.arc(dotX(c), dotY(r), dotRadius, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  const labelSize = Math.max(9, Math.min(14, Math.round(cellSize * 0.32)));
  ctx.font = `500 ${labelSize}px 'Inter', sans-serif`;
  ctx.fillStyle = AXIS_LABEL_COLOR;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let c = 0; c < cols; c++) {
    ctx.fillText(String(c), dotX(c) + cellSize / 2, originY - 12);
  }
  ctx.textAlign = 'right';
  for (let r = 0; r < rows; r++) {
    ctx.fillText(String(r), originX - 10, dotY(r) + cellSize / 2);
  }
}

export function renderer(options: RendererOptions<DotsAndBoxesStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as DotsAndBoxesStep[];
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
  const obs: DotsAndBoxesBoardState | null = currentStep?.boardState ?? null;
  if (!obs) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const isTerminal = obs.is_terminal;
  const activeIdx = isTerminal ? -1 : obs.current_player === '1' ? 0 : obs.current_player === '2' ? 1 : -1;

  const prevObs = step > 0 ? (steps[step - 1]?.boardState ?? null) : null;
  const delta: [number, number] = [
    prevObs ? obs.scores[0] - prevObs.scores[0] : 0,
    prevObs ? obs.scores[1] - prevObs.scores[1] : 0,
  ];

  const renderPlayerCard = (i: 0 | 1) => {
    const cls = i === 0 ? 'p0' : 'p1';
    const color = i === 0 ? P1_COLOR : P2_COLOR;
    const score = obs.scores[i];
    const d = delta[i];
    const deltaHtml = d > 0 ? `<span class="delta" style="color: ${color};">+${d}</span>` : '';
    return `
      <span class="player ${cls} sketched-border ${activeIdx === i ? 'active' : ''}" style="color: ${color};">
        <span class="glyph"></span>${playerNames[i]}
        <span class="score">${score}</span>
        ${deltaHtml}
      </span>
    `;
  };

  header.innerHTML = `
    ${renderPlayerCard(0)}
    <span class="vs">vs</span>
    ${renderPlayerCard(1)}
  `;

  const sizeAndDraw = () => {
    const rect = wrap.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const cssW = Math.max(1, Math.floor(rect.width));
    const cssH = Math.max(1, Math.floor(rect.height));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = cssW;
    canvas.height = cssH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawBoard(ctx, cssW, cssH, obs);
  };
  requestAnimationFrame(sizeAndDraw);

  let statusHTML = '';
  if (isTerminal) {
    if (obs.winner === 'draw') {
      statusHTML = `<p style="margin: 0;">Draw ${obs.scores[0]}&ndash;${obs.scores[1]}</p>`;
    } else if (obs.winner === '1') {
      statusHTML = `<p style="margin: 0;"><span style="color: ${P1_COLOR};">${playerNames[0]} Wins ${obs.scores[0]}&ndash;${obs.scores[1]}</span></p>`;
    } else if (obs.winner === '2') {
      statusHTML = `<p style="margin: 0;"><span style="color: ${P2_COLOR};">${playerNames[1]} Wins ${obs.scores[1]}&ndash;${obs.scores[0]}</span></p>`;
    } else {
      statusHTML = `<p style="margin: 0;">Game Over</p>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? P1_COLOR : P2_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;
  }

  if (obs.last_action) {
    const la = obs.last_action;
    const moverColor = la.player === '1' ? P1_COLOR : P2_COLOR;
    statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">${formatMove(la)}</span></span>`;
  }

  const totalBoxes = obs.num_rows * obs.num_cols;
  const claimed = obs.scores[0] + obs.scores[1];
  statusHTML += `<span class="annotation">${claimed}/${totalBoxes} boxes</span>`;
  statusContainer.innerHTML = statusHTML;
}
