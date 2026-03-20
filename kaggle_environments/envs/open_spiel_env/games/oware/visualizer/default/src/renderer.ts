import type { RendererOptions } from '@kaggle-environments/core';

// Oware observation string format: "player | score0 score1 | s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11"
// Seeds 0-5 are player 0's houses (left to right), 6-11 are player 1's houses (right to left visually)

interface OwareState {
  currentPlayer: number;
  scores: [number, number];
  seeds: number[];
}

// Player colors: P1 = blue, P2 = orange
const PLAYER_COLORS = ['#4fc3f7', '#ff8a65'] as const;
const PLAYER_COLORS_DIM = ['rgba(79,195,247,0.18)', 'rgba(255,138,101,0.18)'] as const;
const PLAYER_COLORS_GLOW = ['rgba(79,195,247,0.35)', 'rgba(255,138,101,0.35)'] as const;

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

const PIT_COLORS = {
  board: '#8B6914',
  pit: '#654321',
  pitStroke: '#3d2a13',
  seed: '#e8d5a3',
  seedStroke: '#b8a070',
  store: '#4a3010',
  text: '#f5e6c8',
  highlight: '#ffd700',
  deltaPlus: '#66bb6a',
  deltaMinus: '#ef5350',
  captureRing: '#ffeb3b',
  playedMarker: '#ffffff',
  panelBg: '#16213e',
};

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

  // Parse previous step for diff visualization
  let prevState: OwareState | null = null;
  if (step > 0) {
    const prevStep = steps[step - 1];
    const prevObs = getObservationString(prevStep);
    prevState = parseObservation(prevObs);
  }

  // Background
  c.fillStyle = '#1a1a2e';
  c.fillRect(0, 0, width, height);

  if (!state) {
    c.fillStyle = '#fff';
    c.font = '16px sans-serif';
    c.textAlign = 'center';
    c.fillText('Waiting for game data...', width / 2, height / 2);
    return;
  }

  // --- Layout: info panel at top, board below ---
  const panelH = 44;
  const panelMargin = 8;
  const boardW = Math.min(width * 0.9, 700);
  const boardH = Math.min((height - panelH - panelMargin * 2) * 0.6, 250);
  const bx = (width - boardW) / 2;
  const by = panelH + panelMargin * 2 + (height - panelH - panelMargin * 2 - boardH) / 2 - 10;
  const storeW = boardW * 0.1;
  const pitAreaW = boardW - storeW * 2;
  const pitW = pitAreaW / 6;
  const pitH = boardH / 2;
  const pitRadius = Math.min(pitW, pitH) * 0.38;

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

    // Determine the played pit: the action from the CURRENT step tells us what
    // was played to get here. But actually, the action in currentStep is the
    // action that LED to this observation. In the replay data, the action in
    // step N is the action that was applied to transition from step N-1's state
    // to step N's state. We look at the current step's action.
    const action = getAction(currentStep);
    if (action !== null && action >= 0) {
      // The previous player (prevState.currentPlayer) made this action.
      const prevPlayer = prevState.currentPlayer;
      // Player 0's pits are 0-5, player 1's pits are 6-11
      playedPitIndex = prevPlayer === 0 ? action : 6 + action;
    }

    // Detect captured pits: pits in the opponent's row that went to 0
    // while the acting player gained score.
    if (prevState.currentPlayer === 0 && scoreDeltas[0] > 0) {
      // P1 acted, captures from P2's row (pits 6-11)
      for (let i = 6; i < 12; i++) {
        if ((prevState.seeds[i] ?? 0) > 0 && (state.seeds[i] ?? 0) === 0) {
          capturedPits.add(i);
        }
      }
    } else if (prevState.currentPlayer === 1 && scoreDeltas[1] > 0) {
      // P2 acted, captures from P1's row (pits 0-5)
      for (let i = 0; i < 6; i++) {
        if ((prevState.seeds[i] ?? 0) > 0 && (state.seeds[i] ?? 0) === 0) {
          capturedPits.add(i);
        }
      }
    }
  }

  // === Draw Turn Indicator Panel at top ===
  const panelX = bx;
  const panelY = panelMargin;
  const panelW = boardW;
  const activeColor = PLAYER_COLORS[state.currentPlayer];

  // Panel background
  c.fillStyle = PIT_COLORS.panelBg;
  c.beginPath();
  c.roundRect(panelX, panelY, panelW, panelH, 10);
  c.fill();

  // Colored accent bar on left edge
  c.fillStyle = activeColor;
  c.beginPath();
  c.roundRect(panelX, panelY, 5, panelH, [10, 0, 0, 10]);
  c.fill();

  // Turn text or game over text
  c.textBaseline = 'middle';
  const panelCenterY = panelY + panelH / 2;
  const panelFontSize = Math.max(13, Math.min(panelH * 0.38, 18));

  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over -- Draw';
    let msgColor = PIT_COLORS.text;
    if (rewards[0] > rewards[1]) {
      msg = 'Game Over -- Player 1 wins!';
      msgColor = PLAYER_COLORS[0];
    } else if (rewards[1] > rewards[0]) {
      msg = 'Game Over -- Player 2 wins!';
      msgColor = PLAYER_COLORS[1];
    }
    c.fillStyle = msgColor;
    c.font = `bold ${panelFontSize}px sans-serif`;
    c.textAlign = 'left';
    c.fillText(msg, panelX + 16, panelCenterY);
  } else {
    // "Player N's turn"
    c.font = `bold ${panelFontSize}px sans-serif`;
    c.textAlign = 'left';
    c.fillStyle = activeColor;
    const turnLabel = `Player ${state.currentPlayer + 1}'s turn`;
    c.fillText(turnLabel, panelX + 16, panelCenterY);
  }

  // Score display on the right side of panel
  const scoreFontSize = Math.max(12, Math.min(panelH * 0.34, 16));
  c.font = `bold ${scoreFontSize}px sans-serif`;
  c.textAlign = 'right';

  // Measure and draw score components with player colors
  // Format: "Score: P1Score - P2Score  (+N)"
  const scoreRightX = panelX + panelW - 16;
  let cursorX = scoreRightX;

  // Draw score change indicators first (rightmost)
  if (prevState) {
    if (scoreDeltas[1] > 0) {
      c.fillStyle = PIT_COLORS.deltaPlus;
      c.font = `bold ${Math.max(10, scoreFontSize - 2)}px sans-serif`;
      const deltaText = ` +${scoreDeltas[1]}`;
      c.fillText(deltaText, cursorX, panelCenterY);
      cursorX -= c.measureText(deltaText).width;
    }
  }

  // P2 score
  c.font = `bold ${scoreFontSize}px sans-serif`;
  c.fillStyle = PLAYER_COLORS[1];
  const p2ScoreStr = String(state.scores[1]);
  c.fillText(p2ScoreStr, cursorX, panelCenterY);
  cursorX -= c.measureText(p2ScoreStr).width;

  // Separator
  c.fillStyle = PIT_COLORS.text;
  c.fillText(' - ', cursorX, panelCenterY);
  cursorX -= c.measureText(' - ').width;

  // P1 score (with delta if present)
  c.fillStyle = PLAYER_COLORS[0];
  const p1ScoreStr = String(state.scores[0]);
  // If P1 has a score delta, draw it between P1 score and separator
  if (prevState && scoreDeltas[0] > 0) {
    c.font = `bold ${Math.max(10, scoreFontSize - 2)}px sans-serif`;
    c.fillStyle = PIT_COLORS.deltaPlus;
    const d1Text = ` +${scoreDeltas[0]}`;
    c.fillText(d1Text, cursorX, panelCenterY);
    cursorX -= c.measureText(d1Text).width;
  }
  c.font = `bold ${scoreFontSize}px sans-serif`;
  c.fillStyle = PLAYER_COLORS[0];
  c.fillText(p1ScoreStr, cursorX, panelCenterY);
  cursorX -= c.measureText(p1ScoreStr).width;

  // "Score: " label
  c.fillStyle = PIT_COLORS.text;
  c.fillText('Score: ', cursorX, panelCenterY);

  // === Draw Board ===
  c.fillStyle = PIT_COLORS.board;
  c.beginPath();
  c.roundRect(bx, by, boardW, boardH, 16);
  c.fill();

  // === Draw active player row glow ===
  if (!terminal) {
    const glowColor = PLAYER_COLORS_GLOW[state.currentPlayer];
    c.save();
    c.shadowColor = PLAYER_COLORS[state.currentPlayer];
    c.shadowBlur = 12;
    c.strokeStyle = glowColor;
    c.lineWidth = 3;
    c.beginPath();
    if (state.currentPlayer === 0) {
      // Bottom row
      c.roundRect(bx + storeW - 2, by + pitH - 2, pitAreaW + 4, pitH + 4, [0, 0, 12, 12]);
    } else {
      // Top row
      c.roundRect(bx + storeW - 2, by - 2, pitAreaW + 4, pitH + 4, [12, 12, 0, 0]);
    }
    c.stroke();
    c.restore();
  }

  // === Draw stores with player-tinted backgrounds ===
  const drawStore = (x: number, y: number, score: number, label: string, playerIdx: number, scoreDelta: number) => {
    // Tinted background using player color
    const pColor = PLAYER_COLORS[playerIdx];
    const pColorDim = PLAYER_COLORS_DIM[playerIdx];

    c.fillStyle = PIT_COLORS.store;
    c.beginPath();
    c.roundRect(x + 4, y + 8, storeW - 8, boardH - 16, 12);
    c.fill();

    // Tinted overlay
    c.fillStyle = pColorDim;
    c.beginPath();
    c.roundRect(x + 4, y + 8, storeW - 8, boardH - 16, 12);
    c.fill();

    // Score number (larger and bolder)
    c.fillStyle = PIT_COLORS.text;
    c.font = `bold ${Math.max(18, pitRadius * 1.1)}px sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(String(score), x + storeW / 2, y + boardH / 2 - 14);

    // Score delta
    if (scoreDelta > 0) {
      c.fillStyle = PIT_COLORS.deltaPlus;
      c.font = `bold ${Math.max(11, pitRadius * 0.5)}px sans-serif`;
      c.fillText(`+${scoreDelta}`, x + storeW / 2, y + boardH / 2 + 4);
    }

    // Player label with player color
    c.fillStyle = pColor;
    c.font = `bold ${Math.max(11, pitRadius * 0.45)}px sans-serif`;
    c.fillText(label, x + storeW / 2, y + boardH / 2 + (scoreDelta > 0 ? 22 : 16));
  };

  drawStore(bx, by, state.scores[1], 'P2', 1, scoreDeltas[1]);
  drawStore(bx + boardW - storeW, by, state.scores[0], 'P1', 0, scoreDeltas[0]);

  // === Draw pits ===
  const drawPit = (
    cx: number,
    cy: number,
    count: number,
    pitIndex: number,
    isActiveRow: boolean,
    playerIdx: number
  ) => {
    const isPlayed = playedPitIndex === pitIndex;
    const isCaptured = capturedPits.has(pitIndex);
    const delta = seedDeltas[pitIndex];
    const playerColor = PLAYER_COLORS[playerIdx];

    // Pit fill
    c.fillStyle = PIT_COLORS.pit;
    c.beginPath();
    c.arc(cx, cy, pitRadius, 0, Math.PI * 2);
    c.fill();

    // Border: use player color ring for active row pits, default stroke otherwise
    if (isActiveRow && !terminal) {
      c.save();
      c.strokeStyle = playerColor;
      c.lineWidth = 2.5;
      c.shadowColor = playerColor;
      c.shadowBlur = 6;
      c.beginPath();
      c.arc(cx, cy, pitRadius, 0, Math.PI * 2);
      c.stroke();
      c.restore();
    } else {
      c.strokeStyle = PIT_COLORS.pitStroke;
      c.lineWidth = 1.5;
      c.beginPath();
      c.arc(cx, cy, pitRadius, 0, Math.PI * 2);
      c.stroke();
    }

    // Played marker: concentric dashed ring
    if (isPlayed) {
      c.save();
      c.strokeStyle = PIT_COLORS.playedMarker;
      c.lineWidth = 2;
      c.setLineDash([4, 3]);
      c.globalAlpha = 0.85;
      c.beginPath();
      c.arc(cx, cy, pitRadius + 5, 0, Math.PI * 2);
      c.stroke();
      c.setLineDash([]);
      c.globalAlpha = 1.0;

      // Small star/diamond icon above the pit
      const starY = cy - pitRadius - 10;
      c.fillStyle = PIT_COLORS.playedMarker;
      c.font = `bold ${Math.max(10, pitRadius * 0.35)}px sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText('\u25C6', cx, starY); // diamond character
      c.restore();
    }

    // Capture indicator: gold pulsing ring
    if (isCaptured) {
      c.save();
      c.strokeStyle = PIT_COLORS.captureRing;
      c.lineWidth = 3;
      c.globalAlpha = 0.9;
      c.setLineDash([6, 3]);
      c.beginPath();
      c.arc(cx, cy, pitRadius + 4, 0, Math.PI * 2);
      c.stroke();
      c.setLineDash([]);
      c.globalAlpha = 1.0;

      // "CAPTURED" mini label
      c.fillStyle = PIT_COLORS.captureRing;
      c.font = `bold ${Math.max(8, pitRadius * 0.25)}px sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText('CAPTURED', cx, cy - pitRadius - 10);
      c.restore();
    }

    // Draw seeds as small circles
    if (count > 0 && count <= 12) {
      const seedR = Math.max(3, pitRadius * 0.15);
      const positions = getSeedPositions(count, pitRadius * 0.65);
      for (const [sx, sy] of positions) {
        c.fillStyle = PIT_COLORS.seed;
        c.strokeStyle = PIT_COLORS.seedStroke;
        c.lineWidth = 0.5;
        c.beginPath();
        c.arc(cx + sx, cy + sy, seedR, 0, Math.PI * 2);
        c.fill();
        c.stroke();
      }
    }

    // Always show count
    c.fillStyle = PIT_COLORS.text;
    c.font = `bold ${Math.max(11, pitRadius * 0.45)}px sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(String(count), cx, cy + pitRadius + Math.max(10, pitRadius * 0.45));

    // Delta indicator
    if (prevState && delta !== 0) {
      const deltaFontSize = Math.max(10, pitRadius * 0.38);
      c.font = `bold ${deltaFontSize}px sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      if (delta > 0) {
        c.fillStyle = PIT_COLORS.deltaPlus;
        c.fillText(`+${delta}`, cx + pitRadius * 0.7, cy - pitRadius * 0.7);
      } else {
        c.fillStyle = PIT_COLORS.deltaMinus;
        c.fillText(`${delta}`, cx + pitRadius * 0.7, cy - pitRadius * 0.7);
      }
    }
  };

  // Player 1 row (bottom, left to right: pits 0-5)
  for (let i = 0; i < 6; i++) {
    const cx = bx + storeW + pitW * i + pitW / 2;
    const cy = by + pitH + pitH / 2;
    const isActiveRow = state.currentPlayer === 0;
    drawPit(cx, cy, state.seeds[i] ?? 0, i, isActiveRow, 0);
  }

  // Player 2 row (top, right to left: pits 6-11)
  for (let i = 0; i < 6; i++) {
    const cx = bx + storeW + pitW * (5 - i) + pitW / 2;
    const cy = by + pitH / 2;
    const isActiveRow = state.currentPlayer === 1;
    drawPit(cx, cy, state.seeds[6 + i] ?? 0, 6 + i, isActiveRow, 1);
  }

  // Row labels
  c.font = `${Math.max(10, pitRadius * 0.35)}px sans-serif`;
  c.textAlign = 'center';
  const labelY1 = by + boardH + Math.max(16, pitRadius * 0.6);
  const labelY2 = by - Math.max(8, pitRadius * 0.3);
  for (let i = 0; i < 6; i++) {
    const cx = bx + storeW + pitW * i + pitW / 2;
    c.fillStyle = PLAYER_COLORS[0];
    c.fillText(String.fromCharCode(65 + i), cx, labelY1); // A-F for P1
    c.fillStyle = PLAYER_COLORS[1];
    c.fillText(String.fromCharCode(102 - i), cx, labelY2); // f-a for P2
  }

  // Status bar (simplified, main info is in the panel now)
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over - Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over - Player 1 wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over - Player 2 wins!';
    statusBar.textContent = `${msg} (${state.scores[0]} - ${state.scores[1]})`;
  } else {
    statusBar.textContent = `Player ${state.currentPlayer + 1}'s turn | Score: ${state.scores[0]} - ${state.scores[1]}`;
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
    // Inner and outer rings
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
