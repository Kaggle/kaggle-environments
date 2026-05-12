import type { RendererOptions } from '@kaggle-environments/core';
import type { DarkHexBoardState, DarkHexStep } from './transformers/darkHexTransformer';

type DarkHexObservation = DarkHexBoardState;

const PLAYER_X_COLOR = '#1f77b4';
const PLAYER_O_COLOR = '#d62728';
const CELL_FILL = '#fbf7e8';
const SECONDARY_TEXT = '#444343';
const SKETCH_STROKE = '#3c3b37';

const SQRT3 = Math.sqrt(3);

function actionToCoords(action: number, numCols: number): { row: number; col: number } {
  return { row: Math.floor(action / numCols), col: action % numCols };
}

function colLabel(col: number): string {
  return String.fromCharCode('a'.charCodeAt(0) + col);
}

function actionString(action: number, numCols: number): string {
  const { row, col } = actionToCoords(action, numCols);
  return `${colLabel(col)}${row + 1}`;
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player X' : 'Player O';
}

function drawHexBoard(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  board: string[][],
  numRows: number,
  numCols: number,
  highlightCell: { row: number; col: number } | null,
  highlightColor: string
) {
  ctx.clearRect(0, 0, width, height);

  const padding = 28;
  const innerW = Math.max(0, width - padding * 2);
  const innerH = Math.max(0, height - padding * 2);
  const totalUnitsW = numCols + (numRows - 1) / 2;
  const totalUnitsH = (numRows - 1) * 1.5 + 2;
  const sizeFromW = innerW / (totalUnitsW * SQRT3);
  const sizeFromH = innerH / totalUnitsH;
  const size = Math.max(8, Math.min(sizeFromW, sizeFromH));
  const hexW = SQRT3 * size;
  const hexH = 2 * size;
  const boardW = totalUnitsW * hexW;
  const boardH = (numRows - 1) * 1.5 * size + hexH;
  const xOffset = (width - boardW) / 2;
  const yOffset = (height - boardH) / 2;

  const centerOf = (row: number, col: number) => ({
    x: xOffset + (col + row * 0.5 + 0.5) * hexW,
    y: yOffset + size + row * 1.5 * size,
  });

  const hexPath = (cx: number, cy: number, s: number) => {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i + Math.PI / 6;
      const x = cx + s * Math.cos(angle);
      const y = cy + s * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
  };

  // Draw colored edge guides for the connection goals.
  // Player X (0) connects top↔bottom (rows). Player O (1) connects left↔right (cols).
  const edgeWidth = Math.max(3, size * 0.18);
  ctx.lineWidth = edgeWidth;
  ctx.lineCap = 'round';

  // Top edge (X)
  ctx.strokeStyle = PLAYER_X_COLOR;
  let topL = centerOf(0, 0);
  let topR = centerOf(0, numCols - 1);
  ctx.beginPath();
  ctx.moveTo(topL.x - hexW / 2, topL.y - size * 0.6);
  ctx.lineTo(topR.x + hexW / 2, topR.y - size * 0.6);
  ctx.stroke();

  // Bottom edge (X)
  let botL = centerOf(numRows - 1, 0);
  let botR = centerOf(numRows - 1, numCols - 1);
  ctx.beginPath();
  ctx.moveTo(botL.x - hexW / 2, botL.y + size * 0.6);
  ctx.lineTo(botR.x + hexW / 2, botR.y + size * 0.6);
  ctx.stroke();

  // Left edge (O) -- zigzag along left side
  ctx.strokeStyle = PLAYER_O_COLOR;
  ctx.beginPath();
  for (let r = 0; r < numRows; r++) {
    const c = centerOf(r, 0);
    const x = c.x - hexW / 2 - size * 0.15;
    if (r === 0) ctx.moveTo(x, c.y - size * 0.5);
    else ctx.lineTo(x, c.y - size * 0.5);
    ctx.lineTo(x, c.y + size * 0.5);
  }
  ctx.stroke();

  // Right edge (O)
  ctx.beginPath();
  for (let r = 0; r < numRows; r++) {
    const c = centerOf(r, numCols - 1);
    const x = c.x + hexW / 2 + size * 0.15;
    if (r === 0) ctx.moveTo(x, c.y - size * 0.5);
    else ctx.lineTo(x, c.y - size * 0.5);
    ctx.lineTo(x, c.y + size * 0.5);
  }
  ctx.stroke();

  // Draw hex cells
  ctx.font = `${Math.round(size * 0.7)}px 'Inter', sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      const { x, y } = centerOf(r, c);
      const cell = board?.[r]?.[c] ?? '.';

      hexPath(x, y, size * 0.95);
      ctx.fillStyle = CELL_FILL;
      ctx.fill();
      ctx.lineWidth = 1.25;
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = SKETCH_STROKE;
      ctx.stroke();
      ctx.setLineDash([]);

      if (cell === 'x') {
        ctx.fillStyle = PLAYER_X_COLOR;
        ctx.fillText('X', x, y + size * 0.04);
      } else if (cell === 'o') {
        ctx.fillStyle = PLAYER_O_COLOR;
        ctx.fillText('O', x, y + size * 0.04);
      }
    }
  }

  // Highlight the most recent move on this board
  if (highlightCell) {
    const { x, y } = centerOf(highlightCell.row, highlightCell.col);
    hexPath(x, y, size * 0.95);
    ctx.lineWidth = Math.max(2.5, size * 0.12);
    ctx.setLineDash([]);
    ctx.strokeStyle = highlightColor;
    ctx.stroke();
  }

  // Coordinate labels
  ctx.font = `${Math.round(size * 0.32)}px 'Inter', sans-serif`;
  ctx.fillStyle = SECONDARY_TEXT;
  for (let c = 0; c < numCols; c++) {
    const top = centerOf(0, c);
    ctx.fillText(colLabel(c), top.x, top.y - size - size * 0.25);
  }
  for (let r = 0; r < numRows; r++) {
    const left = centerOf(r, 0);
    ctx.fillText(`${r + 1}`, left.x - hexW * 0.55 - size * 0.25, left.y);
  }
}

function buildBoardCard(label: string, color: string): HTMLDivElement {
  const card = document.createElement('div');
  card.className = 'board-card';
  const labelEl = document.createElement('div');
  labelEl.className = 'board-label';
  labelEl.style.color = color;
  labelEl.textContent = label;
  const canvas = document.createElement('canvas');
  card.appendChild(labelEl);
  card.appendChild(canvas);
  return card;
}

export function renderer(options: RendererOptions<DarkHexStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as DarkHexStep[];
  if (!steps.length) return;

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="boards"></div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const boards = parent.querySelector('.boards') as HTMLDivElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;

  const currentStep = steps[step];

  const obsX = currentStep?.boardX ?? null;
  const obsO = currentStep?.boardO ?? null;
  const observation = obsX ?? obsO;
  if (!observation) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const numCols = observation.num_cols;
  const isTerminal = observation.is_terminal;
  const currentPlayer = observation.current_player;

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const activeIdx = isTerminal ? -1 : currentPlayer === 'x' ? 0 : currentPlayer === 'o' ? 1 : -1;

  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${PLAYER_X_COLOR};">
      ${playerNames[0]} <span style="opacity: 0.7;">(X)</span>
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${PLAYER_O_COLOR};">
      ${playerNames[1]} <span style="opacity: 0.7;">(O)</span>
    </span>
  `;

  const lastAction = currentStep?.lastAction ?? null;
  const lastActor = currentStep?.lastActor ?? null;
  const lastCell = lastAction !== null ? actionToCoords(lastAction, numCols) : null;

  // Build two board cards.
  boards.innerHTML = '';
  const xCard = buildBoardCard("X's view", PLAYER_X_COLOR);
  const oCard = buildBoardCard("O's view", PLAYER_O_COLOR);
  boards.appendChild(xCard);
  boards.appendChild(oCard);

  // Natural aspect of the hex parallelogram board. Used to fit the canvas
  // into the card without leaving large empty bands.
  const numRows = observation.num_rows;
  const aspect = ((numCols + (numRows - 1) / 2) * Math.sqrt(3)) / ((numRows - 1) * 1.5 + 2);

  const drawCard = (card: HTMLDivElement, obs: DarkHexObservation | null, ownerIdx: number) => {
    const canvas = card.querySelector('canvas') as HTMLCanvasElement;
    canvas.style.width = '0';
    canvas.style.height = '0';
    const cardRect = card.getBoundingClientRect();
    const label = card.querySelector('.board-label') as HTMLElement | null;
    const labelH = label ? label.getBoundingClientRect().height : 0;
    const gap = 8; // matches .board-card { gap } in style.css
    const availW = cardRect.width;
    const availH = Math.max(0, cardRect.height - labelH - gap);
    let cssW = Math.min(availW, availH * aspect);
    let cssH = cssW / aspect;
    cssW = Math.max(1, Math.floor(cssW));
    cssH = Math.max(1, Math.floor(cssH));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.width = cssW;
    canvas.height = cssH;
    const ctx = canvas.getContext('2d');
    if (!ctx || !obs) return;

    const board = obs.board;
    // The actor always sees the result of their own action (a placed piece on success, or the
    // opponent's revealed piece on a collision). The opponent only learns about the action when
    // their own piece was the collision target -- which is the same cell, already showing their
    // own piece in their view.
    let highlight: { row: number; col: number } | null = null;
    const highlightColor = lastActor === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    if (lastCell && ownerIdx === lastActor) {
      const cellInThisView = board?.[lastCell.row]?.[lastCell.col];
      if (cellInThisView === 'x' || cellInThisView === 'o') {
        highlight = lastCell;
      }
    }
    drawHexBoard(ctx, canvas.width, canvas.height, board, obs.num_rows, obs.num_cols, highlight, highlightColor);
  };

  // Defer to next frame so layout sizes are settled.
  requestAnimationFrame(() => {
    drawCard(xCard, obsX, 0);
    drawCard(oCard, obsO, 1);
  });

  // Status footer.
  let statusHTML = '';
  if (isTerminal) {
    if (observation.winner === 'x') {
      statusHTML = `<span style="color: ${PLAYER_X_COLOR};">${playerNames[0]} (X) wins!</span>`;
    } else if (observation.winner === 'o') {
      statusHTML = `<span style="color: ${PLAYER_O_COLOR};">${playerNames[1]} (O) wins!</span>`;
    } else {
      statusHTML = `<span>Game over: ${observation.winner ?? 'finished'}</span>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;
  }
  if (lastAction !== null && lastActor !== null) {
    const moveStr = actionString(lastAction, numCols);
    const moverColor = lastActor === 0 ? PLAYER_X_COLOR : PLAYER_O_COLOR;
    statusHTML += `<span class="annotation">last move: <span style="color: ${moverColor}; font-weight: 600;">${moveStr}</span></span>`;
  }
  statusContainer.innerHTML = statusHTML;
}
