import type { RendererOptions } from '@kaggle-environments/core';
import type { UltimateTicTacToeStep } from './transformers/ultimateTicTacToeTransformer';

const P1_COLOR = '#2563eb'; // Sapphire / Blue
const P2_COLOR = '#dc2626'; // Ruby / Red

interface VisualizerState {
  initialized: boolean;
  activeStep: number;
  hoveredCell: { subgrid: number; cell: number } | null;
  animationStartTime: number;
  lastStepWithWinners: number;
  prevWinners: string[];
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player 1' : 'Player 2';
}

export function renderer(options: RendererOptions<UltimateTicTacToeStep[]>) {
  const { step, replay, parent } = options;
  const steps = (replay.steps ?? []) as UltimateTicTacToeStep[];

  if (!steps || steps.length === 0) return;

  // 1. Initialize or retrieve state on parent element
  let stateObj: VisualizerState = (parent as any).__visualizer_state__;
  if (!stateObj) {
    stateObj = {
      initialized: false,
      activeStep: step,
      hoveredCell: null,
      animationStartTime: 0,
      lastStepWithWinners: -1,
      prevWinners: Array(9).fill(''),
    };
    (parent as any).__visualizer_state__ = stateObj;
  }

  // 2. Build UI structure lazily (run once)
  const existingContainer = parent.querySelector('.renderer-container') as HTMLDivElement;
  if (!existingContainer) {
    parent.innerHTML = `
      <div class="renderer-container">
        <div class="header">
          <span class="player p0 sketched-border">
            <span class="glyph"></span>
            <span class="name">${getPlayerName(replay, 0)}</span>
          </span>
          <span class="vs">vs</span>
          <span class="player p1 sketched-border">
            <span class="glyph"></span>
            <span class="name">${getPlayerName(replay, 1)}</span>
          </span>
        </div>
        
        <div class="board-wrap">
          <canvas></canvas>
        </div>
        
        <div class="status-container sketched-border"></div>
      </div>
    `;

    // Canvas mousemove listener for cell hover previews
    const container = parent.querySelector('.renderer-container') as HTMLDivElement;
    const wrapElement = container.querySelector('.board-wrap') as HTMLDivElement;
    wrapElement.addEventListener('mousemove', (e) => {
      const state = (parent as any).__visualizer_state__;
      const currentStep = steps[state.activeStep];
      const parsedObs = currentStep?.boardState;
      const cv = wrapElement.querySelector('canvas');
      if (!cv) return;

      if (!parsedObs || parsedObs.is_terminal) {
        state.hoveredCell = null;
        cv.style.cursor = 'default';
        return;
      }

      const rect = cv.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const cssW = cv.width;
      const cssH = cv.height;
      const size = Math.min(cssW, cssH) - 40;
      const boardLeft = (cssW - size) / 2;
      const boardTop = Math.min(20, (cssH - size) / 2);

      if (mouseX >= boardLeft && mouseX <= boardLeft + size && mouseY >= boardTop && mouseY <= boardTop + size) {
        const relX = mouseX - boardLeft;
        const relY = mouseY - boardTop;
        const subgridSize = size / 3;
        const majorCol = Math.floor(relX / subgridSize);
        const majorRow = Math.floor(relY / subgridSize);
        const s = majorRow * 3 + majorCol;

        const subgridX = relX % subgridSize;
        const subgridY = relY % subgridSize;
        const cellSize = subgridSize / 3;
        const minorCol = Math.floor(subgridX / cellSize);
        const minorRow = Math.floor(subgridY / cellSize);
        const cellIdx = minorRow * 3 + minorCol;

        const activeSub = parsedObs.active_subgrid;
        const isPlayableSub =
          activeSub === null || activeSub === undefined ? parsedObs.subgrid_winners[s] === '' : s === activeSub;

        const isEmptyCell = parsedObs.board?.[s]?.[cellIdx] === '';

        if (isPlayableSub && isEmptyCell && parsedObs.subgrid_winners[s] === '') {
          cv.style.cursor = 'pointer';
          state.hoveredCell = { subgrid: s, cell: cellIdx };
        } else {
          cv.style.cursor = 'default';
          state.hoveredCell = null;
        }
      } else {
        cv.style.cursor = 'default';
        state.hoveredCell = null;
      }
    });

    wrapElement.addEventListener('mouseleave', () => {
      const state = (parent as any).__visualizer_state__;
      state.hoveredCell = null;
      const cv = wrapElement.querySelector('canvas');
      if (cv) cv.style.cursor = 'default';
    });

    stateObj.initialized = true;
  }

  // 3. Update state variables
  stateObj.activeStep = step;

  // 4. Get active step objects
  const currentStep = steps[step];
  const parsedObs = currentStep?.boardState;
  if (!parsedObs) return;

  const container = parent.querySelector('.renderer-container') as HTMLDivElement;

  const isTerminal = parsedObs.is_terminal;
  const activeIdx = isTerminal ? -1 : parsedObs.current_player === 'x' ? 0 : parsedObs.current_player === 'o' ? 1 : -1;

  // Update Header player cards active states
  const player1Card = container.querySelector('.player.p0') as HTMLSpanElement;
  const player2Card = container.querySelector('.player.p1') as HTMLSpanElement;
  if (activeIdx === 0) {
    player1Card.classList.add('active');
    player2Card.classList.remove('active');
  } else if (activeIdx === 1) {
    player1Card.classList.remove('active');
    player2Card.classList.add('active');
  } else {
    player1Card.classList.remove('active');
    player2Card.classList.remove('active');
  }

  // 5. Update Status Container Text
  const statusContainer = container.querySelector('.status-container') as HTMLDivElement;
  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];

  let statusHTML = '';
  if (isTerminal) {
    if (parsedObs.winner === 'draw') {
      statusHTML = `<span style="font-weight: 700; color: #475569;">Draw</span>`;
    } else {
      const winnerName = parsedObs.winner === 'x' ? playerNames[0] : playerNames[1];
      const winnerColor = parsedObs.winner === 'x' ? P1_COLOR : P2_COLOR;
      statusHTML = `<span style="color: ${winnerColor}; font-weight: 700;">${winnerName} Wins!</span>`;
    }
  } else {
    const turnColor = activeIdx === 0 ? P1_COLOR : P2_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    statusHTML = `<span>Turn: <span style="color: ${turnColor}; font-weight: 700;">${turnName}</span></span>`;

    if (parsedObs.active_subgrid !== null && parsedObs.active_subgrid !== undefined) {
      const subR = Math.floor(parsedObs.active_subgrid / 3);
      const subC = parsedObs.active_subgrid % 3;
      statusHTML += `<span class="annotation" style="margin-left: 8px;">| Mandated Sub-grid: [Row ${subR}, Col ${subC}]</span>`;
    } else {
      statusHTML += `<span class="annotation" style="margin-left: 8px;">| Free move (any open sub-grid)</span>`;
    }
  }

  const lastMovedPlayer = currentStep.players.find((p: any) => p.isTurn);
  if (lastMovedPlayer && lastMovedPlayer.actionDisplayText) {
    statusHTML += `<span class="annotation" style="margin-left: 12px; opacity: 0.8;">(Last: ${lastMovedPlayer.actionDisplayText})</span>`;
  }
  statusContainer.innerHTML = statusHTML;

  // 6. Winner Overlay
  let winnerOverlay = container.querySelector('.utt-winner-overlay') as HTMLDivElement;
  if (winnerOverlay) {
    winnerOverlay.remove();
  }

  const p0Wins = parsedObs.subgrid_winners.filter((w: string) => w === 'x').length;
  const p1Wins = parsedObs.subgrid_winners.filter((w: string) => w === 'o').length;

  if (isTerminal) {
    const winner = parsedObs.winner;
    let winnerLabel = 'Match Ended in a Draw';
    let subtitle = `Neither player succeeded in conquering 3 sub-grids in a row. (X won ${p0Wins} subgrids, O won ${p1Wins} subgrids)`;
    let overlayClass = 'draw';

    if (winner === 'x') {
      winnerLabel = `${playerNames[0]} Wins!`;
      subtitle = 'Player 1 (X) successfully conquered the macro-grid!';
      overlayClass = 'x';
    } else if (winner === 'o') {
      winnerLabel = `${playerNames[1]} Wins!`;
      subtitle = 'Player 2 (O) successfully conquered the macro-grid!';
      overlayClass = 'o';
    }

    winnerOverlay = document.createElement('div');
    winnerOverlay.className = `utt-winner-overlay ${overlayClass}`;
    winnerOverlay.innerHTML = `
      <div class="utt-winner-card">
        <div class="utt-winner-title">${winnerLabel}</div>
        <div class="utt-winner-subtitle">${subtitle}</div>
      </div>
    `;
    container.querySelector('.board-wrap')!.appendChild(winnerOverlay);
  }

  // 7. Render Board Canvas
  const wrap = container.querySelector('.board-wrap') as HTMLDivElement;
  let canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
  if (!canvas) {
    canvas = document.createElement('canvas');
    wrap.appendChild(canvas);
  }

  // Track subgrid wins transitions for animation
  if (stateObj.lastStepWithWinners !== step) {
    stateObj.prevWinners =
      step > 0 ? steps[step - 1]?.boardState?.subgrid_winners || Array(9).fill('') : Array(9).fill('');
    stateObj.animationStartTime = Date.now();
    stateObj.lastStepWithWinners = step;
  }

  const sizeAndDraw = () => {
    if (stateObj.activeStep !== step) return; // Stale draw call
    const wrapRect = wrap.getBoundingClientRect();
    const cssW = Math.max(1, Math.floor(wrapRect.width));
    const cssH = Math.max(1, Math.floor(wrapRect.height));

    if (canvas.width !== cssW || canvas.height !== cssH) {
      canvas.width = cssW;
      canvas.height = cssH;
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
    }

    const c = canvas.getContext('2d');
    if (!c) return;

    c.clearRect(0, 0, cssW, cssH);

    const size = Math.min(cssW, cssH) - 40;
    const boardLeft = (cssW - size) / 2;
    const boardTop = Math.min(20, (cssH - size) / 2);
    const subgridSize = size / 3;
    const cellSize = subgridSize / 3;

    const active = parsedObs.active_subgrid;

    // 1. Draw glowing active subgrid backgrounds and frosted glass inactive overlays
    for (let s = 0; s < 9; s++) {
      const majorRow = Math.floor(s / 3);
      const majorCol = s % 3;
      const x = boardLeft + majorCol * subgridSize;
      const y = boardTop + majorRow * subgridSize;
      const winner = parsedObs.subgrid_winners?.[s] || '';

      const isPlayable = !isTerminal && (active === null || active === undefined ? winner === '' : s === active);

      if (isPlayable) {
        c.save();
        // Glowing active perimeter pulse
        const time = Date.now() / 300;
        const pulse = 2 + Math.sin(time) * 1.5;

        const playerColor = parsedObs.current_player === 'x' ? P1_COLOR : P2_COLOR;
        const interiorColor = parsedObs.current_player === 'x' ? 'rgba(37, 99, 235, 0.06)' : 'rgba(220, 38, 38, 0.06)';

        c.shadowColor = playerColor;
        c.shadowBlur = 10 + pulse;
        c.strokeStyle = playerColor;
        c.lineWidth = 3;
        c.strokeRect(x + 5, y + 5, subgridSize - 10, subgridSize - 10);

        // Faint interior
        c.fillStyle = interiorColor;
        c.fillRect(x + 5, y + 5, subgridSize - 10, subgridSize - 10);
        c.restore();
      } else if (!isTerminal && winner === '') {
        // Frosted glass overlay on non-active but playable subgrids
        c.save();
        c.fillStyle = 'rgba(241, 245, 249, 0.45)'; // Frosted white overlay
        c.fillRect(x + 4, y + 4, subgridSize - 8, subgridSize - 8);
        c.restore();
      }
    }

    // 2. Draw minor grid lines (etched line look)
    c.save();
    c.strokeStyle = 'rgba(100, 116, 139, 0.25)'; // Slate 500 faint
    c.lineWidth = 1;
    for (let majorRow = 0; majorRow < 3; majorRow++) {
      for (let majorCol = 0; majorCol < 3; majorCol++) {
        const subgridX = boardLeft + majorCol * subgridSize;
        const subgridY = boardTop + majorRow * subgridSize;

        for (let r = 1; r < 3; r++) {
          c.beginPath();
          c.moveTo(subgridX + 12, subgridY + r * cellSize);
          c.lineTo(subgridX + subgridSize - 12, subgridY + r * cellSize);
          c.stroke();
        }

        for (let col = 1; col < 3; col++) {
          c.beginPath();
          c.moveTo(subgridX + col * cellSize, subgridY + 12);
          c.lineTo(subgridX + col * cellSize, subgridY + subgridSize - 12);
          c.stroke();
        }
      }
    }
    c.restore();

    // 3. Draw major grid lines (charcoal brushed look)
    c.save();
    c.strokeStyle = '#334155'; // Slate 700
    c.lineWidth = 4;
    c.lineCap = 'round';
    c.shadowColor = 'rgba(15, 23, 42, 0.15)';
    c.shadowBlur = 4;
    c.shadowOffsetY = 2;
    for (let i = 1; i < 3; i++) {
      // Horizontal
      c.beginPath();
      c.moveTo(boardLeft, boardTop + i * subgridSize);
      c.lineTo(boardLeft + size, boardTop + i * subgridSize);
      c.stroke();

      // Vertical
      c.beginPath();
      c.moveTo(boardLeft + i * subgridSize, boardTop);
      c.lineTo(boardLeft + i * subgridSize, boardTop + size);
      c.stroke();
    }
    c.restore();

    // 4. Pre-calculate cell move numbers
    const moveOrders = calculateMoveOrders(step, steps);

    // 5. Draw cell marks & subgrid winners
    let needsRefigStep = false;
    for (let s = 0; s < 9; s++) {
      const majorRow = Math.floor(s / 3);
      const majorCol = s % 3;
      const subgridX = boardLeft + majorCol * subgridSize;
      const subgridY = boardTop + majorRow * subgridSize;

      const subWinner = parsedObs.subgrid_winners?.[s] || '';
      const prevWinner = stateObj.prevWinners?.[s] || '';

      if (subWinner !== '') {
        // Subgrid is won! Draw large ornate winning mark
        const centerX = subgridX + subgridSize / 2;
        const centerY = subgridY + subgridSize / 2;
        const radius = subgridSize * 0.35;

        c.save();

        // Animate scale if it was just won
        let scale = 1;
        if (subWinner !== prevWinner) {
          const elapsed = Date.now() - stateObj.animationStartTime;
          scale = Math.min(1, elapsed / 400); // 400ms transition
          if (scale < 1) {
            needsRefigStep = true;
          }
        }

        c.translate(centerX, centerY);
        c.scale(scale, scale);
        c.translate(-centerX, -centerY);

        if (subWinner === 'x') {
          c.fillStyle = 'rgba(37, 99, 235, 0.08)'; // Sapphire faint glow
          c.fillRect(subgridX + 4, subgridY + 4, subgridSize - 8, subgridSize - 8);
          drawMetallicX(c, centerX, centerY, radius, 8, true);
        } else if (subWinner === 'o') {
          c.fillStyle = 'rgba(220, 38, 38, 0.08)'; // Ruby faint glow
          c.fillRect(subgridX + 4, subgridY + 4, subgridSize - 8, subgridSize - 8);
          drawMetallicO(c, centerX, centerY, radius, 8, false);
        } else if (subWinner === 'draw') {
          c.fillStyle = 'rgba(71, 85, 105, 0.15)'; // Tie slate overlay
          c.fillRect(subgridX + 4, subgridY + 4, subgridSize - 8, subgridSize - 8);

          c.strokeStyle = '#64748b';
          c.lineWidth = 4;
          c.lineCap = 'round';
          c.beginPath();
          c.moveTo(centerX - radius, centerY);
          c.lineTo(centerX + radius, centerY);
          c.stroke();
        }

        c.restore();
      } else {
        // Subgrid is ongoing. Draw individual cell marks
        const subBoard = parsedObs.board[s];
        for (let cellIdx = 0; cellIdx < 9; cellIdx++) {
          const minorRow = Math.floor(cellIdx / 3);
          const minorCol = cellIdx % 3;
          const cellX = subgridX + minorCol * cellSize + cellSize / 2;
          const cellY = subgridY + minorRow * cellSize + cellSize / 2;

          const mark = subBoard[cellIdx];
          const radius = cellSize * 0.28;

          if (mark === 'x') {
            drawMetallicX(c, cellX, cellY, radius, 3.5, true);
          } else if (mark === 'o') {
            drawMetallicO(c, cellX, cellY, radius, 3.5, false);
          }

          // Draw move numbers
          const moveNo = moveOrders[s][cellIdx];
          if (moveNo > 0) {
            c.save();
            c.fillStyle = '#64748b'; // Slate 500
            c.font = 'bold 8.5px "Share Tech Mono", monospace';
            c.textAlign = 'right';
            c.textBaseline = 'bottom';
            c.fillText(
              `${mark.toUpperCase()}${moveNo}`,
              subgridX + minorCol * cellSize + cellSize - 3,
              subgridY + minorRow * cellSize + cellSize - 2
            );
            c.restore();
          }
        }
      }
    }

    // 6. Draw hovered cell preview translucent mark
    if (stateObj.hoveredCell && !isTerminal) {
      const { subgrid: s, cell: cellIdx } = stateObj.hoveredCell;
      const majorRow = Math.floor(s / 3);
      const majorCol = s % 3;
      const subgridX = boardLeft + majorCol * subgridSize;
      const subgridY = boardTop + majorRow * subgridSize;
      const minorRow = Math.floor(cellIdx / 3);
      const minorCol = cellIdx % 3;
      const cellX = subgridX + minorCol * cellSize + cellSize / 2;
      const cellY = subgridY + minorRow * cellSize + cellSize / 2;

      const radius = cellSize * 0.28;
      const turnPlayer = parsedObs.current_player;

      c.save();
      // Inset cell shadow background
      c.fillStyle = 'rgba(56, 189, 248, 0.12)';
      c.fillRect(subgridX + minorCol * cellSize + 2, subgridY + minorRow * cellSize + 2, cellSize - 4, cellSize - 4);
      c.strokeStyle = 'rgba(56, 189, 248, 0.4)';
      c.lineWidth = 1.5;
      c.strokeRect(subgridX + minorCol * cellSize + 2, subgridY + minorRow * cellSize + 2, cellSize - 4, cellSize - 4);

      // Translucent mark
      if (turnPlayer === 'x') {
        drawMetallicX(c, cellX, cellY, radius, 3, true);
        c.fillStyle = 'rgba(248, 250, 252, 0.5)';
        c.beginPath();
        c.arc(cellX, cellY, radius + 2, 0, 2 * Math.PI);
        c.fill();
      } else if (turnPlayer === 'o') {
        drawMetallicO(c, cellX, cellY, radius, 3, false);
        c.fillStyle = 'rgba(248, 250, 252, 0.5)';
        c.beginPath();
        c.arc(cellX, cellY, radius + 2, 0, 2 * Math.PI);
        c.fill();
      }
      c.restore();
    }

    // 7. Request another frame if we are in transition
    if (needsRefigStep) {
      requestAnimationFrame(sizeAndDraw);
    }
  };

  requestAnimationFrame(sizeAndDraw);
}

// Helpers
function calculateMoveOrders(step: number, steps: UltimateTicTacToeStep[]): number[][] {
  const moveOrders = Array.from({ length: 9 }, () => Array(9).fill(0));
  let cellMoveCount = 0;
  for (let t = 0; t <= step; t++) {
    const move = steps[t]?.move;
    if (move && move.cellIdx !== null) {
      cellMoveCount++;
      moveOrders[move.subgridIdx][move.cellIdx] = cellMoveCount;
    }
  }
  return moveOrders;
}

function drawMetallicX(
  c: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  lineWidth: number,
  isSapphire: boolean
) {
  c.save();
  c.lineCap = 'round';

  // Drop shadow
  c.strokeStyle = 'rgba(15, 23, 42, 0.45)';
  c.lineWidth = lineWidth;
  c.beginPath();
  c.moveTo(x - r + 1, y - r + 1.5);
  c.lineTo(x + r + 1, y + r + 1.5);
  c.moveTo(x + r + 1, y - r + 1.5);
  c.lineTo(x - r + 1, y + r + 1.5);
  c.stroke();

  // Metallic gradient
  const grad = c.createLinearGradient(x - r, y - r, x + r, y + r);
  if (isSapphire) {
    grad.addColorStop(0, '#60a5fa');
    grad.addColorStop(0.5, '#2563eb');
    grad.addColorStop(1, '#1e3a8a');
  } else {
    grad.addColorStop(0, '#f87171');
    grad.addColorStop(0.5, '#dc2626');
    grad.addColorStop(1, '#7f1d1d');
  }

  c.strokeStyle = grad;
  c.lineWidth = lineWidth;
  c.beginPath();
  c.moveTo(x - r, y - r);
  c.lineTo(x + r, y + r);
  c.moveTo(x + r, y - r);
  c.lineTo(x - r, y + r);
  c.stroke();

  // Highlight
  c.strokeStyle = isSapphire ? 'rgba(219, 234, 254, 0.5)' : 'rgba(fee2e2, 0.5)';
  c.lineWidth = lineWidth * 0.3;
  c.beginPath();
  c.moveTo(x - r - 0.5, y - r - 0.5);
  c.lineTo(x + r - 0.5, y + r - 0.5);
  c.moveTo(x + r - 0.5, y - r - 0.5);
  c.lineTo(x - r - 0.5, y + r - 0.5);
  c.stroke();

  c.restore();
}

function drawMetallicO(
  c: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  lineWidth: number,
  isSapphire: boolean
) {
  c.save();

  // Drop shadow
  c.strokeStyle = 'rgba(15, 23, 42, 0.45)';
  c.lineWidth = lineWidth;
  c.beginPath();
  c.arc(x + 1, y + 1.5, r, 0, 2 * Math.PI);
  c.stroke();

  // Metallic gradient
  const grad = c.createLinearGradient(x - r, y - r, x + r, y + r);
  if (isSapphire) {
    grad.addColorStop(0, '#60a5fa');
    grad.addColorStop(0.5, '#2563eb');
    grad.addColorStop(1, '#1e3a8a');
  } else {
    grad.addColorStop(0, '#f87171');
    grad.addColorStop(0.5, '#dc2626');
    grad.addColorStop(1, '#7f1d1d');
  }

  c.strokeStyle = grad;
  c.lineWidth = lineWidth;
  c.beginPath();
  c.arc(x, y, r, 0, 2 * Math.PI);
  c.stroke();

  // Highlight
  c.strokeStyle = isSapphire ? 'rgba(219, 234, 254, 0.5)' : 'rgba(fee2e2, 0.5)';
  c.lineWidth = lineWidth * 0.3;
  c.beginPath();
  c.arc(x - 0.5, y - 0.5, r, 0, 2 * Math.PI);
  c.stroke();

  c.restore();
}
