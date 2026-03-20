import type { RendererOptions } from '@kaggle-environments/core';

// Nim observation string format: "(player): pile0 pile1 pile2 ..."

interface NimState {
  currentPlayer: number;
  piles: number[];
}

function parseObservation(obsString: string): NimState | null {
  if (!obsString) return null;
  // Format: "(0): 1 3 5 7"
  const match = obsString.match(/\((\d+)\):\s*([\d\s]+)/);
  if (!match) return null;
  return {
    currentPlayer: parseInt(match[1]),
    piles: match[2].trim().split(/\s+/).map(Number),
  };
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

function getRewards(step: any): [number, number] {
  if (!step || !Array.isArray(step)) return [0, 0];
  return [step[0]?.reward ?? 0, step[1]?.reward ?? 0];
}

const PLAYER_COLORS = {
  p1: '#4fc3f7',
  p2: '#ff8a65',
};

const COLORS = {
  stone: '#64b5f6',
  stoneStroke: '#1976d2',
  stoneGone: '#2a2a4a',
  stoneGoneStroke: '#3a3a5a',
  recentlyRemoved: '#ef5350',
  recentlyRemovedStroke: '#c62828',
  text: '#e0e0e0',
  highlight: '#ffd700',
  pileLabel: '#90caf9',
  panelBg: '#16213e',
  colBg: 'rgba(255, 255, 255, 0.04)',
  colBgHighlight: 'rgba(255, 215, 0, 0.08)',
  colBorder: 'rgba(255, 255, 255, 0.08)',
  colBorderHighlight: '#ffd700',
};

/** Compute the diff between previous step and current step. */
function computeMoveDiff(
  steps: any[],
  stepIndex: number,
  currentPiles: number[]
): { changedPile: number; removedCount: number } | null {
  if (stepIndex <= 0) return null;
  const prevObs = getObservationString(steps[stepIndex - 1]);
  const prevState = parseObservation(prevObs);
  if (!prevState) return null;

  for (let i = 0; i < currentPiles.length; i++) {
    const diff = prevState.piles[i] - currentPiles[i];
    if (diff > 0) {
      return { changedPile: i, removedCount: diff };
    }
  }
  return null;
}

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

  // Get initial piles to show "removed" stones
  let initialPiles: number[] | null = null;
  for (let i = 0; i < steps.length; i++) {
    const obs = getObservationString(steps[i]);
    const s = parseObservation(obs);
    if (s) {
      initialPiles = s.piles;
      break;
    }
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

  const terminal = isTerminal(currentStep);
  const moveDiff = computeMoveDiff(steps, step, state.piles);

  // --- Determine who made the last move (for coloring the move indicator) ---
  // The current observation says whose turn it is NEXT, so the last mover is the other player.
  // At terminal, we use rewards to determine the winner.
  const lastMoverIndex = moveDiff !== null ? 1 - state.currentPlayer : -1;

  // =========================================================================
  //  INFO PANEL (top area)
  // =========================================================================
  const panelHeight = 54;
  const panelPad = 10;
  const borderWidth = 5;

  // Panel background
  c.fillStyle = COLORS.panelBg;
  c.fillRect(0, 0, width, panelHeight);

  // Colored left border for current player
  const playerColor = terminal ? COLORS.text : state.currentPlayer === 0 ? PLAYER_COLORS.p1 : PLAYER_COLORS.p2;
  c.fillStyle = playerColor;
  c.fillRect(0, 0, borderWidth, panelHeight);

  // Background tint
  c.fillStyle = playerColor.replace(')', ', 0.07)').replace('rgb', 'rgba').replace('#', '');
  // Use a manual alpha overlay instead of string manipulation
  c.save();
  c.globalAlpha = 0.07;
  c.fillStyle = playerColor;
  c.fillRect(borderWidth, 0, width - borderWidth, panelHeight);
  c.restore();

  // Title: "Nim" on the left
  c.fillStyle = COLORS.text;
  c.font = `bold ${Math.max(15, Math.min(18, width * 0.035))}px sans-serif`;
  c.textAlign = 'left';
  c.textBaseline = 'middle';
  c.fillText('Nim', borderWidth + panelPad, panelHeight * 0.33);

  // Turn indicator or game-over message
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over -- Draw';
    let msgColor = COLORS.text;
    if (rewards[0] > rewards[1]) {
      msg = 'Game Over -- Player 1 wins!';
      msgColor = PLAYER_COLORS.p1;
    } else if (rewards[1] > rewards[0]) {
      msg = 'Game Over -- Player 2 wins!';
      msgColor = PLAYER_COLORS.p2;
    }
    c.fillStyle = msgColor;
    c.font = `bold ${Math.max(13, Math.min(16, width * 0.03))}px sans-serif`;
    c.textAlign = 'left';
    c.fillText(msg, borderWidth + panelPad, panelHeight * 0.72);
  } else {
    const turnLabel = `Player ${state.currentPlayer + 1}'s turn`;
    c.fillStyle = playerColor;
    c.font = `bold ${Math.max(13, Math.min(16, width * 0.03))}px sans-serif`;
    c.textAlign = 'left';
    c.fillText(turnLabel, borderWidth + panelPad, panelHeight * 0.72);
  }

  // Right side: stones remaining + move info
  const stonesRemaining = state.piles.reduce((a, b) => a + b, 0);
  const totalInitial = initialPiles ? initialPiles.reduce((a, b) => a + b, 0) : stonesRemaining;

  c.fillStyle = COLORS.text;
  c.font = `${Math.max(12, Math.min(14, width * 0.026))}px sans-serif`;
  c.textAlign = 'right';
  c.textBaseline = 'middle';
  c.fillText(`Stones remaining: ${stonesRemaining} / ${totalInitial}`, width - panelPad, panelHeight * 0.33);

  if (moveDiff) {
    const moveColor = lastMoverIndex === 0 ? PLAYER_COLORS.p1 : PLAYER_COLORS.p2;
    c.fillStyle = moveColor;
    c.font = `${Math.max(11, Math.min(13, width * 0.024))}px sans-serif`;
    c.fillText(
      `Removed this turn: ${moveDiff.removedCount} from Pile ${moveDiff.changedPile + 1}`,
      width - panelPad,
      panelHeight * 0.72
    );
  } else if (!terminal) {
    c.fillStyle = 'rgba(224, 224, 224, 0.5)';
    c.font = `${Math.max(11, Math.min(13, width * 0.024))}px sans-serif`;
    c.fillText('No move yet', width - panelPad, panelHeight * 0.72);
  }

  // Divider line below panel
  c.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  c.lineWidth = 1;
  c.beginPath();
  c.moveTo(0, panelHeight);
  c.lineTo(width, panelHeight);
  c.stroke();

  // =========================================================================
  //  PILE RENDERING
  // =========================================================================
  const numPiles = state.piles.length;
  const maxPile = initialPiles ? Math.max(...initialPiles) : Math.max(...state.piles);
  const drawAreaTop = panelHeight + 10;
  const drawAreaBottom = height * 0.92;
  const drawAreaHeight = drawAreaBottom - drawAreaTop;

  const stoneR = Math.min((width * 0.8) / (numPiles * 3), (drawAreaHeight * 0.75) / ((maxPile + 1) * 2.5), 28);
  const colSpacing = Math.min(stoneR * 3.5, (width * 0.85) / numPiles);
  const totalW = colSpacing * numPiles;
  const startX = (width - totalW) / 2 + colSpacing / 2;
  const baseY = drawAreaBottom - stoneR * 1.2;

  // Get previous step piles for recently-removed detection
  let prevPiles: number[] | null = null;
  if (step > 0) {
    const prevObs = getObservationString(steps[step - 1]);
    const prevState = parseObservation(prevObs);
    if (prevState) prevPiles = prevState.piles;
  }

  for (let pile = 0; pile < numPiles; pile++) {
    const cx = startX + pile * colSpacing;
    const maxInPile = initialPiles ? initialPiles[pile] : state.piles[pile];
    const currentCount = state.piles[pile];
    const prevCount = prevPiles ? prevPiles[pile] : currentCount;
    const isChangedPile = moveDiff !== null && moveDiff.changedPile === pile;

    // --- Column background ---
    const colLeft = cx - colSpacing / 2 + 3;
    const colRight = cx + colSpacing / 2 - 3;
    const colW = colRight - colLeft;
    const colTop = drawAreaTop;
    const colBot = baseY + stoneR + 6;
    const cornerR = 6;

    // Draw rounded-rect column background
    c.fillStyle = isChangedPile ? COLORS.colBgHighlight : COLORS.colBg;
    c.beginPath();
    c.moveTo(colLeft + cornerR, colTop);
    c.lineTo(colLeft + colW - cornerR, colTop);
    c.arcTo(colLeft + colW, colTop, colLeft + colW, colTop + cornerR, cornerR);
    c.lineTo(colLeft + colW, colBot - cornerR);
    c.arcTo(colLeft + colW, colBot, colLeft + colW - cornerR, colBot, cornerR);
    c.lineTo(colLeft + cornerR, colBot);
    c.arcTo(colLeft, colBot, colLeft, colBot - cornerR, cornerR);
    c.lineTo(colLeft, colTop + cornerR);
    c.arcTo(colLeft, colTop, colLeft + cornerR, colTop, cornerR);
    c.closePath();
    c.fill();

    // Highlight border for changed pile
    if (isChangedPile) {
      c.strokeStyle = COLORS.colBorderHighlight;
      c.lineWidth = 2;
      c.stroke();

      // Outer glow
      c.save();
      c.shadowColor = COLORS.colBorderHighlight;
      c.shadowBlur = 10;
      c.strokeStyle = 'rgba(255, 215, 0, 0.4)';
      c.lineWidth = 1;
      c.stroke();
      c.restore();
    } else {
      c.strokeStyle = COLORS.colBorder;
      c.lineWidth = 1;
      c.stroke();
    }

    // --- Draw stones ---
    for (let s = 0; s < maxInPile; s++) {
      const sy = baseY - s * stoneR * 2.3;
      const present = s < currentCount;
      // A stone is "recently removed" if it existed in the previous step but not now
      const recentlyRemoved = !present && s < prevCount && isChangedPile;

      if (present) {
        // Filled stone
        c.fillStyle = COLORS.stone;
        c.strokeStyle = COLORS.stoneStroke;
        c.lineWidth = 2;
        c.beginPath();
        c.arc(cx, sy, stoneR, 0, Math.PI * 2);
        c.fill();
        c.stroke();

        // Shine effect
        c.fillStyle = 'rgba(255, 255, 255, 0.2)';
        c.beginPath();
        c.arc(cx - stoneR * 0.25, sy - stoneR * 0.25, stoneR * 0.4, 0, Math.PI * 2);
        c.fill();
      } else if (recentlyRemoved) {
        // Recently removed stone: red/orange filled with transparency
        c.fillStyle = 'rgba(239, 83, 80, 0.25)';
        c.strokeStyle = COLORS.recentlyRemovedStroke;
        c.lineWidth = 2;
        c.beginPath();
        c.arc(cx, sy, stoneR, 0, Math.PI * 2);
        c.fill();
        c.stroke();

        // "X" mark over the stone to emphasize removal
        const xSize = stoneR * 0.45;
        c.strokeStyle = COLORS.recentlyRemoved;
        c.lineWidth = 2.5;
        c.beginPath();
        c.moveTo(cx - xSize, sy - xSize);
        c.lineTo(cx + xSize, sy + xSize);
        c.moveTo(cx + xSize, sy - xSize);
        c.lineTo(cx - xSize, sy + xSize);
        c.stroke();
      } else {
        // Ghost stone (removed in earlier turns)
        c.strokeStyle = COLORS.stoneGoneStroke;
        c.lineWidth = 1.5;
        c.setLineDash([3, 3]);
        c.beginPath();
        c.arc(cx, sy, stoneR, 0, Math.PI * 2);
        c.stroke();
        c.setLineDash([]);
      }
    }

    // --- Pile label ---
    const labelFontSize = Math.max(11, stoneR * 0.65);
    const countFontSize = Math.max(10, stoneR * 0.5);
    const labelY = baseY + stoneR + 12;
    c.fillStyle = isChangedPile ? COLORS.colBorderHighlight : COLORS.pileLabel;
    c.font = `bold ${labelFontSize}px sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'top';
    c.fillText(`Pile ${pile + 1}`, cx, labelY);

    c.fillStyle = COLORS.text;
    c.font = `${countFontSize}px sans-serif`;
    c.fillText(`(${currentCount})`, cx, labelY + labelFontSize + 2);

    // --- "-N" indicator for the changed pile ---
    if (isChangedPile && moveDiff) {
      const moveColor = lastMoverIndex === 0 ? PLAYER_COLORS.p1 : PLAYER_COLORS.p2;
      const badgeFontSize = Math.max(12, stoneR * 0.65);
      const badgeText = `-${moveDiff.removedCount}`;

      // Position badge to the right of the column
      const badgeX = cx + colSpacing / 2 - 2;
      // Vertically center it relative to where stones were removed
      const removedMidIdx = currentCount + (moveDiff.removedCount - 1) / 2;
      const badgeY = baseY - removedMidIdx * stoneR * 2.3;

      // Badge background
      c.fillStyle = moveColor;
      c.font = `bold ${badgeFontSize}px sans-serif`;
      c.textAlign = 'left';
      c.textBaseline = 'middle';
      const metrics = c.measureText(badgeText);
      const bPadX = 5;
      const bPadY = 3;
      const bW = metrics.width + bPadX * 2;
      const bH = badgeFontSize + bPadY * 2;

      // Rounded badge background
      const bLeft = badgeX;
      const bTop = badgeY - bH / 2;
      const bCorner = 4;
      c.fillStyle = 'rgba(0, 0, 0, 0.6)';
      c.beginPath();
      c.moveTo(bLeft + bCorner, bTop);
      c.lineTo(bLeft + bW - bCorner, bTop);
      c.arcTo(bLeft + bW, bTop, bLeft + bW, bTop + bCorner, bCorner);
      c.lineTo(bLeft + bW, bTop + bH - bCorner);
      c.arcTo(bLeft + bW, bTop + bH, bLeft + bW - bCorner, bTop + bH, bCorner);
      c.lineTo(bLeft + bCorner, bTop + bH);
      c.arcTo(bLeft, bTop + bH, bLeft, bTop + bH - bCorner, bCorner);
      c.lineTo(bLeft, bTop + bCorner);
      c.arcTo(bLeft, bTop, bLeft + bCorner, bTop, bCorner);
      c.closePath();
      c.fill();

      c.strokeStyle = moveColor;
      c.lineWidth = 1.5;
      c.stroke();

      // Badge text
      c.fillStyle = moveColor;
      c.fillText(badgeText, bLeft + bPadX, badgeY);
    }
  }

  // =========================================================================
  //  STATUS BAR (bottom, kept for compatibility)
  // =========================================================================
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over -- Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over -- Player 1 wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over -- Player 2 wins!';
    statusBar.textContent = msg;
  } else {
    statusBar.textContent = `Player ${state.currentPlayer + 1}'s turn`;
  }
}
