import type { RendererOptions } from '@kaggle-environments/core';

export function renderer(options: RendererOptions) {
  const { step, replay, parent, agents } = options;
  const steps = replay.steps as any[];

  // 1. Rebuild HTML framework
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

  if (!canvas || !replay) return;

  const player1Name = agents?.[0]?.name || 'Player 1';
  const player2Name = agents?.[1]?.name || 'Player 2';

  legend.innerHTML = `
    <div class="legend-item">
      <span style="color: #06b6d4; font-size: 20px; font-weight: bold; margin-right: 4px;">✕</span>
      <span>${player1Name} <span style="opacity: 0.7;">(X)</span></span>
    </div>
    <div class="legend-item">
      <span style="color: #eab308; font-size: 20px; font-weight: bold; margin-right: 4px;">◯</span>
      <span>${player2Name} <span style="opacity: 0.7;">(O)</span></span>
    </div>
  `;

  const c = canvas.getContext('2d');
  if (!c) return;

  // Size canvas responsively
  canvas.width = 0;
  canvas.height = 0;
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = width;
  canvas.height = height;

  c.fillStyle = '#111116';
  c.fillRect(0, 0, width, height);

  const currentStep = steps[step];
  let state: any = null;
  const rawObs = currentStep?.[0]?.observation?.observationString;
  if (rawObs) {
    try {
      state = JSON.parse(rawObs);
    } catch (e) {
      console.error('Error parsing observation JSON:', e);
    }
  }

  // Draw board if state is available
  if (state && state.board) {
    const size = Math.min(width, height) - 40;
    const boardLeft = (width - size) / 2;
    const boardTop = (height - size) / 2;
    const subgridSize = size / 3;
    const cellSize = subgridSize / 3;

    // A. Draw glowing background for active subgrid(s)
    const isTerminal = state.is_terminal;
    if (!isTerminal) {
      const active = state.active_subgrid;
      for (let s = 0; s < 9; s++) {
        const subgridWinner = state.subgrid_winners?.[s] || '';
        const isPlayable = active === null || active === undefined ? subgridWinner === '' : s === active;

        if (isPlayable) {
          const majorRow = Math.floor(s / 3);
          const majorCol = s % 3;
          const x = boardLeft + majorCol * subgridSize;
          const y = boardTop + majorRow * subgridSize;

          c.save();
          // Subtle neon green background glow
          c.fillStyle = 'rgba(34, 197, 94, 0.08)';
          c.fillRect(x + 4, y + 4, subgridSize - 8, subgridSize - 8);
          c.strokeStyle = 'rgba(34, 197, 94, 0.4)';
          c.lineWidth = 3;
          c.strokeRect(x + 4, y + 4, subgridSize - 8, subgridSize - 8);
          c.restore();
        }
      }
    }

    // B. Draw minor grid lines (inside each subgrid)
    c.save();
    c.strokeStyle = '#334155';
    c.lineWidth = 1;
    for (let majorRow = 0; majorRow < 3; majorRow++) {
      for (let majorCol = 0; majorCol < 3; majorCol++) {
        const subgridX = boardLeft + majorCol * subgridSize;
        const subgridY = boardTop + majorRow * subgridSize;

        // Draw minor horizontal lines
        for (let r = 1; r < 3; r++) {
          c.beginPath();
          c.moveTo(subgridX + 10, subgridY + r * cellSize);
          c.lineTo(subgridX + subgridSize - 10, subgridY + r * cellSize);
          c.stroke();
        }

        // Draw minor vertical lines
        for (let col = 1; col < 3; col++) {
          c.beginPath();
          c.moveTo(subgridX + col * cellSize, subgridY + 10);
          c.lineTo(subgridX + col * cellSize, subgridY + subgridSize - 10);
          c.stroke();
        }
      }
    }
    c.restore();

    // C. Draw major grid lines (the large Tic-Tac-Toe grid)
    c.save();
    c.strokeStyle = '#64748b';
    c.lineWidth = 4;
    c.lineCap = 'round';
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

    // D. Draw cell marks (X and O) and major subgrid overlays
    for (let s = 0; s < 9; s++) {
      const majorRow = Math.floor(s / 3);
      const majorCol = s % 3;
      const subgridX = boardLeft + majorCol * subgridSize;
      const subgridY = boardTop + majorRow * subgridSize;

      const subWinner = state.subgrid_winners?.[s] || '';

      if (subWinner === 'x') {
        // Draw large X overlay
        const centerX = subgridX + subgridSize / 2;
        const centerY = subgridY + subgridSize / 2;
        const radius = subgridSize * 0.35;
        c.save();
        c.shadowColor = '#06b6d4';
        c.shadowBlur = 10;
        drawX(c, centerX, centerY, radius, 12, 'rgba(6, 182, 212, 0.85)');
        c.restore();
      } else if (subWinner === 'o') {
        // Draw large O overlay
        const centerX = subgridX + subgridSize / 2;
        const centerY = subgridY + subgridSize / 2;
        const radius = subgridSize * 0.35;
        c.save();
        c.shadowColor = '#eab308';
        c.shadowBlur = 10;
        drawO(c, centerX, centerY, radius, 12, 'rgba(234, 179, 8, 0.85)');
        c.restore();
      } else if (subWinner === 'draw') {
        // Draw draw shade
        c.save();
        c.fillStyle = 'rgba(100, 116, 139, 0.35)';
        c.fillRect(subgridX + 6, subgridY + 6, subgridSize - 12, subgridSize - 12);
        c.restore();
      } else {
        // Draw individual cells
        const subBoard = state.board[s];
        for (let cellIdx = 0; cellIdx < 9; cellIdx++) {
          const minorRow = Math.floor(cellIdx / 3);
          const minorCol = cellIdx % 3;
          const cellX = subgridX + minorCol * cellSize + cellSize / 2;
          const cellY = subgridY + minorRow * cellSize + cellSize / 2;

          const mark = subBoard[cellIdx];
          const radius = cellSize * 0.28;

          if (mark === 'x') {
            drawX(c, cellX, cellY, radius, 3.5, '#06b6d4');
          } else if (mark === 'o') {
            drawO(c, cellX, cellY, radius, 3.5, '#eab308');
          }
        }
      }
    }

    // E. Draw general game-over overlay if terminal
    if (isTerminal) {
      c.save();
      c.fillStyle = 'rgba(17, 17, 22, 0.7)';
      c.fillRect(0, 0, width, height);

      c.fillStyle = 'white';
      c.font = 'bold 28px sans-serif';
      c.textAlign = 'center';
      c.textBaseline = 'middle';

      let resultText = 'Match Complete';
      if (state.winner === 'x') {
        resultText = `${player1Name} Wins!`;
      } else if (state.winner === 'o') {
        resultText = `${player2Name} Wins!`;
      } else if (state.winner === 'draw') {
        resultText = "It's a Draw!";
      }

      c.shadowColor = 'black';
      c.shadowBlur = 4;
      c.fillText(resultText, width / 2, height / 2);
      c.restore();
    }
  }

  // 4. Update status bar
  if (state) {
    if (state.is_terminal) {
      let resultText = 'Game Over';
      if (state.winner === 'x') {
        resultText = `${player1Name} (X) won the match.`;
      } else if (state.winner === 'o') {
        resultText = `${player2Name} (O) won the match.`;
      } else if (state.winner === 'draw') {
        resultText = 'The match ended in a draw.';
      }
      statusBar.innerHTML = `<div class="status-line">${resultText}</div>`;
    } else {
      const activeName = state.current_player === 'x' ? player1Name : player2Name;
      const activeColor = state.current_player === 'x' ? '#06b6d4' : '#eab308';
      const activeMarkSymbol = state.current_player === 'x' ? '✕' : '◯';

      let phaseText = '';
      if (state.phase === 'choose_subgrid') {
        phaseText = ' - Free to choose any open local board';
      } else if (state.phase === 'choose_cell') {
        phaseText = ` - Must play in local board ${state.active_subgrid}`;
      }

      statusBar.innerHTML = `
        <div class="status-line">
          <span style="color: ${activeColor}; font-weight: bold; margin-right: 6px;">${activeMarkSymbol}</span>
          <span>It is <strong>${activeName}</strong>'s turn${phaseText}</span>
        </div>
      `;
    }
  } else {
    statusBar.innerHTML = '<div class="status-line">Loading game state...</div>';
  }
}

// Helpers
function drawX(c: CanvasRenderingContext2D, x: number, y: number, r: number, lineWidth: number, color: string) {
  c.save();
  c.strokeStyle = color;
  c.lineWidth = lineWidth;
  c.lineCap = 'round';
  c.beginPath();
  c.moveTo(x - r, y - r);
  c.lineTo(x + r, y + r);
  c.moveTo(x + r, y - r);
  c.lineTo(x - r, y + r);
  c.stroke();
  c.restore();
}

function drawO(c: CanvasRenderingContext2D, x: number, y: number, r: number, lineWidth: number, color: string) {
  c.save();
  c.strokeStyle = color;
  c.lineWidth = lineWidth;
  c.beginPath();
  c.arc(x, y, r, 0, 2 * Math.PI);
  c.stroke();
  c.restore();
}
