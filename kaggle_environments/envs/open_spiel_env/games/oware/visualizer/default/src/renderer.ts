import type { RendererOptions } from '@kaggle-environments/core';

// Oware observation string format: "player | score0 score1 | s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11"
// Seeds 0-5 are player 0's houses (left to right), 6-11 are player 1's houses (right to left visually)

interface OwareState {
  currentPlayer: number;
  scores: [number, number];
  seeds: number[];
}

function parseObservation(obsString: string): OwareState | null {
  if (!obsString) return null;
  const parts = obsString.split('|').map((s) => s.trim());
  if (parts.length < 3) return null;
  const currentPlayer = parseInt(parts[0]);
  const scoreParts = parts[1].split(/\s+/).map(Number);
  const seedParts = parts[2].split(/\s+/).map(Number);
  return {
    currentPlayer,
    scores: [scoreParts[0] || 0, scoreParts[1] || 0],
    seeds: seedParts,
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

/** Get the action taken in a step (the submission that is not -1). */
function getAction(stepData: any): number | null {
  if (!stepData || !Array.isArray(stepData)) return null;
  for (const p of stepData) {
    const sub = p?.action?.submission;
    if (sub !== undefined && sub !== null && sub !== -1) return sub;
  }
  return null;
}

const COLORS = {
  board: '#c9a96e',
  boardStroke: '#3c3b37',
  pit: '#a07840',
  pitStroke: '#3c3b37',
  seed: '#3c3b37',
  seedStroke: '#050001',
  store: '#8b6914',
  storeStroke: '#3c3b37',
  text: '#050001',
  textSecondary: '#444343',
  deltaPlus: '#2e7d32',
  deltaMinus: '#c0392b',
  captureRing: '#c0392b',
  playedMarker: '#050001',
  label: '#050001',
};

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header">
        <span class="player-card sketched-border" id="p1-card">P1: 0</span>
        <span class="vs-label">vs</span>
        <span class="player-card sketched-border" id="p2-card">P2: 0</span>
      </div>
      <div class="info-row">
        <span class="move-info"></span>
      </div>
      <canvas></canvas>
      <div class="status-container sketched-border"></div>
    </div>
  `;

  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const p1Card = parent.querySelector('#p1-card') as HTMLSpanElement;
  const p2Card = parent.querySelector('#p2-card') as HTMLSpanElement;
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

  // Parse previous step for diff visualization
  let prevState: OwareState | null = null;
  if (step > 0) {
    const prevStep = steps[step - 1];
    const prevObs = getObservationString(prevStep);
    prevState = parseObservation(prevObs);
  }

  // Transparent canvas -- page background shows through
  c.clearRect(0, 0, width, height);

  if (!state) {
    statusContainer.textContent = 'Waiting for game data...';
    return;
  }

  const terminal = isTerminal(currentStep);

  // Compute diffs
  const seedDeltas: number[] = new Array(12).fill(0);
  let scoreDeltas: [number, number] = [0, 0];
  let playedPitIndex: number | null = null;
  const capturedPits: Set<number> = new Set();

  if (prevState) {
    for (let i = 0; i < 12; i++) {
      seedDeltas[i] = (state.seeds[i] ?? 0) - (prevState.seeds[i] ?? 0);
    }
    scoreDeltas = [state.scores[0] - prevState.scores[0], state.scores[1] - prevState.scores[1]];

    const action = getAction(currentStep);
    if (action !== null && action >= 0) {
      const prevPlayer = prevState.currentPlayer;
      playedPitIndex = prevPlayer === 0 ? action : 6 + action;
    }

    if (prevState.currentPlayer === 0 && scoreDeltas[0] > 0) {
      for (let i = 6; i < 12; i++) {
        if ((prevState.seeds[i] ?? 0) > 0 && (state.seeds[i] ?? 0) === 0) {
          capturedPits.add(i);
        }
      }
    } else if (prevState.currentPlayer === 1 && scoreDeltas[1] > 0) {
      for (let i = 0; i < 6; i++) {
        if ((prevState.seeds[i] ?? 0) > 0 && (state.seeds[i] ?? 0) === 0) {
          capturedPits.add(i);
        }
      }
    }
  }

  // =========================================================================
  //  DOM HEADER -- active player + scores
  // =========================================================================
  const p1ScoreStr = `P1: ${state.scores[0]}`;
  const p2ScoreStr = `P2: ${state.scores[1]}`;
  const p1DeltaStr = scoreDeltas[0] > 0 ? ` (+${scoreDeltas[0]})` : '';
  const p2DeltaStr = scoreDeltas[1] > 0 ? ` (+${scoreDeltas[1]})` : '';
  p1Card.textContent = p1ScoreStr + p1DeltaStr;
  p2Card.textContent = p2ScoreStr + p2DeltaStr;

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
  //  DOM INFO ROW
  // =========================================================================
  if (playedPitIndex !== null) {
    const who = prevState ? (prevState.currentPlayer === 0 ? 'P1' : 'P2') : '';
    const pitLabel =
      playedPitIndex < 6 ? String.fromCharCode(65 + playedPitIndex) : String.fromCharCode(102 - (playedPitIndex - 6));
    moveInfo.textContent = `${who} sowed from pit ${pitLabel}`;
  } else {
    moveInfo.textContent = '';
  }

  // =========================================================================
  //  BOARD RENDERING (canvas)
  // =========================================================================
  const boardW = Math.min(width * 0.95, 500);
  const boardH = Math.min(height * 0.55, 220);
  const bx = (width - boardW) / 2;
  const by = (height - boardH) / 2;
  const storeW = boardW * 0.1;
  const pitAreaW = boardW - storeW * 2;
  const pitW = pitAreaW / 6;
  const pitH = boardH / 2;
  const pitRadius = Math.min(pitW, pitH) * 0.38;

  // Board background
  c.fillStyle = COLORS.board;
  c.beginPath();
  c.roundRect(bx, by, boardW, boardH, 12);
  c.fill();
  c.strokeStyle = COLORS.boardStroke;
  c.lineWidth = 1.5;
  c.setLineDash([4, 4]);
  c.beginPath();
  c.roundRect(bx, by, boardW, boardH, 12);
  c.stroke();
  c.setLineDash([]);

  // === Draw stores ===
  const drawStore = (x: number, y: number, score: number, label: string, scoreDelta: number) => {
    c.fillStyle = COLORS.store;
    c.beginPath();
    c.roundRect(x + 4, y + 8, storeW - 8, boardH - 16, 10);
    c.fill();
    c.strokeStyle = COLORS.boardStroke;
    c.lineWidth = 1;
    c.setLineDash([3, 3]);
    c.beginPath();
    c.roundRect(x + 4, y + 8, storeW - 8, boardH - 16, 10);
    c.stroke();
    c.setLineDash([]);

    // Score number
    c.fillStyle = '#f5f1e2';
    c.font = `bold ${Math.max(18, pitRadius * 1.1)}px 'Inter', sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(String(score), x + storeW / 2, y + boardH / 2 - 14);

    // Score delta
    if (scoreDelta > 0) {
      c.fillStyle = COLORS.deltaPlus;
      c.font = `bold ${Math.max(11, pitRadius * 0.5)}px 'Inter', sans-serif`;
      c.fillText(`+${scoreDelta}`, x + storeW / 2, y + boardH / 2 + 4);
    }

    // Player label
    c.fillStyle = '#f5f1e2';
    c.font = `bold ${Math.max(11, pitRadius * 0.45)}px 'Inter', sans-serif`;
    c.fillText(label, x + storeW / 2, y + boardH / 2 + (scoreDelta > 0 ? 22 : 16));
  };

  drawStore(bx, by, state.scores[1], 'P2', scoreDeltas[1]);
  drawStore(bx + boardW - storeW, by, state.scores[0], 'P1', scoreDeltas[0]);

  // === Draw pits ===
  const drawPit = (cx: number, cy: number, count: number, pitIndex: number, isActiveRow: boolean) => {
    const isPlayed = playedPitIndex === pitIndex;
    const isCaptured = capturedPits.has(pitIndex);
    const delta = seedDeltas[pitIndex];

    // Pit fill
    c.fillStyle = COLORS.pit;
    c.beginPath();
    c.arc(cx, cy, pitRadius, 0, Math.PI * 2);
    c.fill();

    // Border
    if (isActiveRow && !terminal) {
      c.strokeStyle = COLORS.pitStroke;
      c.lineWidth = 2.5;
      c.beginPath();
      c.arc(cx, cy, pitRadius, 0, Math.PI * 2);
      c.stroke();
    } else {
      c.strokeStyle = COLORS.pitStroke;
      c.lineWidth = 1;
      c.beginPath();
      c.arc(cx, cy, pitRadius, 0, Math.PI * 2);
      c.stroke();
    }

    // Played marker: dashed ring
    if (isPlayed) {
      c.strokeStyle = COLORS.playedMarker;
      c.lineWidth = 2;
      c.setLineDash([4, 3]);
      c.beginPath();
      c.arc(cx, cy, pitRadius + 5, 0, Math.PI * 2);
      c.stroke();
      c.setLineDash([]);

      // Small diamond icon above the pit
      const starY = cy - pitRadius - 10;
      c.fillStyle = COLORS.playedMarker;
      c.font = `bold ${Math.max(10, pitRadius * 0.35)}px 'Inter', sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText('\u25C6', cx, starY);
    }

    // Capture indicator: dashed ring
    if (isCaptured) {
      c.strokeStyle = COLORS.captureRing;
      c.lineWidth = 2.5;
      c.setLineDash([5, 3]);
      c.beginPath();
      c.arc(cx, cy, pitRadius + 4, 0, Math.PI * 2);
      c.stroke();
      c.setLineDash([]);

      c.fillStyle = COLORS.captureRing;
      c.font = `bold ${Math.max(8, pitRadius * 0.25)}px 'Inter', sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText('CAPTURED', cx, cy - pitRadius - 10);
    }

    // Draw seeds as small circles
    if (count > 0 && count <= 12) {
      const seedR = Math.max(3, pitRadius * 0.15);
      const positions = getSeedPositions(count, pitRadius * 0.65);
      for (const [sx, sy] of positions) {
        c.fillStyle = COLORS.seed;
        c.strokeStyle = COLORS.seedStroke;
        c.lineWidth = 0.5;
        c.beginPath();
        c.arc(cx + sx, cy + sy, seedR, 0, Math.PI * 2);
        c.fill();
        c.stroke();
      }
    }

    // Count below pit
    c.fillStyle = COLORS.text;
    c.font = `bold ${Math.max(11, pitRadius * 0.45)}px 'Inter', sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(String(count), cx, cy + pitRadius + Math.max(10, pitRadius * 0.45));

    // Delta indicator
    if (prevState && delta !== 0) {
      const deltaFontSize = Math.max(10, pitRadius * 0.38);
      c.font = `bold ${deltaFontSize}px 'Inter', sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      if (delta > 0) {
        c.fillStyle = COLORS.deltaPlus;
        c.fillText(`+${delta}`, cx + pitRadius * 0.7, cy - pitRadius * 0.7);
      } else {
        c.fillStyle = COLORS.deltaMinus;
        c.fillText(`${delta}`, cx + pitRadius * 0.7, cy - pitRadius * 0.7);
      }
    }
  };

  // Player 1 row (bottom, left to right: pits 0-5)
  for (let i = 0; i < 6; i++) {
    const cx = bx + storeW + pitW * i + pitW / 2;
    const cy = by + pitH + pitH / 2;
    const isActiveRow = state.currentPlayer === 0;
    drawPit(cx, cy, state.seeds[i] ?? 0, i, isActiveRow);
  }

  // Player 2 row (top, right to left: pits 6-11)
  for (let i = 0; i < 6; i++) {
    const cx = bx + storeW + pitW * (5 - i) + pitW / 2;
    const cy = by + pitH / 2;
    const isActiveRow = state.currentPlayer === 1;
    drawPit(cx, cy, state.seeds[6 + i] ?? 0, 6 + i, isActiveRow);
  }

  // Row labels
  c.font = `${Math.max(10, pitRadius * 0.35)}px 'Inter', sans-serif`;
  c.textAlign = 'center';
  const labelY1 = by + boardH + Math.max(16, pitRadius * 0.6);
  const labelY2 = by - Math.max(8, pitRadius * 0.3);
  for (let i = 0; i < 6; i++) {
    const cx = bx + storeW + pitW * i + pitW / 2;
    c.fillStyle = COLORS.label;
    c.fillText(String.fromCharCode(65 + i), cx, labelY1); // A-F for P1
    c.fillText(String.fromCharCode(102 - i), cx, labelY2); // f-a for P2
  }

  // =========================================================================
  //  DOM STATUS CONTAINER
  // =========================================================================
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = `Game Over \u2014 Draw (${state.scores[0]} \u2013 ${state.scores[1]})`;
    if (rewards[0] > rewards[1]) msg = `Game Over \u2014 Player 1 wins! (${state.scores[0]} \u2013 ${state.scores[1]})`;
    else if (rewards[1] > rewards[0])
      msg = `Game Over \u2014 Player 2 wins! (${state.scores[0]} \u2013 ${state.scores[1]})`;
    statusContainer.textContent = msg;
    statusContainer.style.fontWeight = '700';
  } else {
    statusContainer.textContent = `Player ${state.currentPlayer + 1}'s turn`;
  }
}

function getSeedPositions(count: number, radius: number): [number, number][] {
  const positions: [number, number][] = [];
  if (count <= 6) {
    const angleStep = (Math.PI * 2) / Math.max(count, 1);
    const r = count === 1 ? 0 : radius * 0.5;
    for (let i = 0; i < count; i++) {
      const angle = angleStep * i - Math.PI / 2;
      positions.push([Math.cos(angle) * r, Math.sin(angle) * r]);
    }
  } else {
    const outer = Math.min(count, 8);
    const inner = count - outer;
    for (let i = 0; i < outer; i++) {
      const angle = ((Math.PI * 2) / outer) * i - Math.PI / 2;
      positions.push([Math.cos(angle) * radius * 0.65, Math.sin(angle) * radius * 0.65]);
    }
    for (let i = 0; i < inner; i++) {
      const angle = ((Math.PI * 2) / Math.max(inner, 1)) * i - Math.PI / 2;
      positions.push([Math.cos(angle) * radius * 0.3, Math.sin(angle) * radius * 0.3]);
    }
  }
  return positions;
}
