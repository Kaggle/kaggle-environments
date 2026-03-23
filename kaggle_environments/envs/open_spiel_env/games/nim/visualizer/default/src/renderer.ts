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

const COLORS = {
  stone: '#3c3b37',
  stoneStroke: '#050001',
  stoneGone: '#d4cfc4',
  recentlyRemoved: '#c0392b',
  recentlyRemovedFill: 'rgba(192, 57, 43, 0.15)',
  text: '#050001',
  colBg: 'rgba(0, 0, 0, 0.04)',
  colBgHighlight: 'rgba(189, 238, 255, 0.25)',
  colBorder: '#3c3b37',
  colBorderDash: [4, 4] as number[],
  colBorderHighlightDash: [] as number[],
  pileLabel: '#050001',
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
      <div class="header">
        <span class="player-card sketched-border" id="p1-card">Player 1</span>
        <span class="vs-label">vs</span>
        <span class="player-card sketched-border" id="p2-card">Player 2</span>
      </div>
      <div class="info-row">
        <span class="stones-info"></span>
        <span class="move-info"></span>
      </div>
      <canvas></canvas>
      <div class="status-container sketched-border"></div>
    </div>
  `;

  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const p1Card = parent.querySelector('#p1-card') as HTMLSpanElement;
  const p2Card = parent.querySelector('#p2-card') as HTMLSpanElement;
  const stonesInfo = parent.querySelector('.stones-info') as HTMLSpanElement;
  const moveInfo = parent.querySelector('.move-info') as HTMLSpanElement;
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

  // Canvas is transparent -- page background shows through
  c.clearRect(0, 0, width, height);

  if (!state) {
    statusContainer.textContent = 'Waiting for game data...';
    return;
  }

  const terminal = isTerminal(currentStep);
  const moveDiff = computeMoveDiff(steps, step, state.piles);
  const lastMoverIndex = moveDiff !== null ? 1 - state.currentPlayer : -1;

  // =========================================================================
  //  DOM HEADER -- active player highlighting
  // =========================================================================
  if (!terminal && state.currentPlayer === 0) {
    p1Card.classList.add('active');
  } else {
    p1Card.classList.remove('active');
  }
  if (!terminal && state.currentPlayer === 1) {
    p2Card.classList.add('active');
  } else {
    p2Card.classList.remove('active');
  }

  // =========================================================================
  //  DOM INFO ROW -- stones remaining + move info
  // =========================================================================
  const stonesRemaining = state.piles.reduce((a, b) => a + b, 0);
  const totalInitial = initialPiles ? initialPiles.reduce((a, b) => a + b, 0) : stonesRemaining;
  stonesInfo.textContent = `Stones: ${stonesRemaining} / ${totalInitial}`;

  if (moveDiff) {
    const who = lastMoverIndex === 0 ? 'P1' : 'P2';
    moveInfo.textContent = `${who} took ${moveDiff.removedCount} from Pile ${moveDiff.changedPile + 1}`;
  } else if (!terminal) {
    moveInfo.textContent = '';
  }

  // =========================================================================
  //  PILE RENDERING (canvas)
  // =========================================================================
  const numPiles = state.piles.length;
  const maxPile = initialPiles ? Math.max(...initialPiles) : Math.max(...state.piles);
  const drawAreaTop = 8;
  const drawAreaBottom = height * 0.88;
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

    // Dashed border on column
    c.strokeStyle = COLORS.colBorder;
    c.lineWidth = isChangedPile ? 1.5 : 1;
    c.setLineDash(isChangedPile ? COLORS.colBorderHighlightDash : COLORS.colBorderDash);
    c.stroke();
    c.setLineDash([]);

    // --- Draw stones ---
    for (let s = 0; s < maxInPile; s++) {
      const sy = baseY - s * stoneR * 2.3;
      const present = s < currentCount;
      const recentlyRemoved = !present && s < prevCount && isChangedPile;

      if (present) {
        // Filled stone -- dark ink on paper
        c.fillStyle = COLORS.stone;
        c.strokeStyle = COLORS.stoneStroke;
        c.lineWidth = 1.5;
        c.beginPath();
        c.arc(cx, sy, stoneR, 0, Math.PI * 2);
        c.fill();
        c.stroke();

        // Shine highlight
        c.fillStyle = 'rgba(255, 255, 255, 0.25)';
        c.beginPath();
        c.arc(cx - stoneR * 0.25, sy - stoneR * 0.25, stoneR * 0.35, 0, Math.PI * 2);
        c.fill();
      } else if (recentlyRemoved) {
        // Recently removed stone: red with transparency
        c.fillStyle = COLORS.recentlyRemovedFill;
        c.strokeStyle = COLORS.recentlyRemoved;
        c.lineWidth = 1.5;
        c.beginPath();
        c.arc(cx, sy, stoneR, 0, Math.PI * 2);
        c.fill();
        c.stroke();

        // "X" mark
        const xSize = stoneR * 0.45;
        c.strokeStyle = COLORS.recentlyRemoved;
        c.lineWidth = 2;
        c.beginPath();
        c.moveTo(cx - xSize, sy - xSize);
        c.lineTo(cx + xSize, sy + xSize);
        c.moveTo(cx + xSize, sy - xSize);
        c.lineTo(cx - xSize, sy + xSize);
        c.stroke();
      } else {
        // Ghost stone (removed in earlier turns) -- dashed outline
        c.strokeStyle = COLORS.stoneGone;
        c.lineWidth = 1;
        c.setLineDash([3, 3]);
        c.beginPath();
        c.arc(cx, sy, stoneR, 0, Math.PI * 2);
        c.stroke();
        c.setLineDash([]);
      }
    }

    // --- Pile label (canvas, since it's spatially tied to the board) ---
    const labelFontSize = Math.max(11, stoneR * 0.65);
    const countFontSize = Math.max(10, stoneR * 0.5);
    const labelY = baseY + stoneR + 12;
    c.fillStyle = COLORS.pileLabel;
    c.font = `600 ${labelFontSize}px 'Inter', sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'top';
    c.fillText(`Pile ${pile + 1}`, cx, labelY);

    c.fillStyle = '#444343';
    c.font = `${countFontSize}px 'Inter', sans-serif`;
    c.fillText(`(${currentCount})`, cx, labelY + labelFontSize + 2);

    // --- "-N" badge for the changed pile ---
    if (isChangedPile && moveDiff) {
      const badgeFontSize = Math.max(12, stoneR * 0.65);
      const badgeText = `-${moveDiff.removedCount}`;

      const badgeX = cx + colSpacing / 2 - 2;
      const removedMidIdx = currentCount + (moveDiff.removedCount - 1) / 2;
      const badgeY = baseY - removedMidIdx * stoneR * 2.3;

      c.font = `bold ${badgeFontSize}px 'Inter', sans-serif`;
      c.textAlign = 'left';
      c.textBaseline = 'middle';
      const metrics = c.measureText(badgeText);
      const bPadX = 5;
      const bPadY = 3;
      const bW = metrics.width + bPadX * 2;
      const bH = badgeFontSize + bPadY * 2;
      const bLeft = badgeX;
      const bTop = badgeY - bH / 2;
      const bCorner = 4;

      // Badge background -- white with dashed border
      c.fillStyle = 'white';
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

      c.strokeStyle = COLORS.recentlyRemoved;
      c.lineWidth = 1;
      c.setLineDash([3, 3]);
      c.stroke();
      c.setLineDash([]);

      // Badge text
      c.fillStyle = COLORS.recentlyRemoved;
      c.fillText(badgeText, bLeft + bPadX, badgeY);
    }
  }

  // =========================================================================
  //  DOM STATUS CONTAINER
  // =========================================================================
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over \u2014 Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over \u2014 Player 1 wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over \u2014 Player 2 wins!';
    statusContainer.textContent = msg;
    statusContainer.style.fontWeight = '700';
  } else {
    statusContainer.textContent = `Player ${state.currentPlayer + 1}'s turn`;
  }
}
