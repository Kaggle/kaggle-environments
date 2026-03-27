import type { RendererOptions } from '@kaggle-environments/core';

// --- Observation from proxy (JSON) ---

interface OwareObs {
  board: [number[], number[]]; // [p0_pits(A-F), p1_pits(a-f)]
  scores: [number, number];
  current_player: number;
  is_terminal: boolean;
  winner: number | 'draw' | null;
  last_action: { player: number; pit: number; pit_name: string } | null;
}

function getObservation(step: any, playerIdx: number = 0): OwareObs | null {
  const raw = step?.[playerIdx]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function getRewards(step: any): [number, number] {
  if (!step || !Array.isArray(step)) return [0, 0];
  return [step[0]?.reward ?? 0, step[1]?.reward ?? 0];
}

// --- Board layout ---
// Traditional Oware layout:
//   P1's row (top):  pits 5, 4, 3, 2, 1, 0 of board[1] (reversed for display)
//   P0's row (bot):  pits 0, 1, 2, 3, 4, 5 of board[0]
// Sowing goes counter-clockwise

const PIT_LABELS_P0 = ['A', 'B', 'C', 'D', 'E', 'F'];
const PIT_LABELS_P1 = ['f', 'e', 'd', 'c', 'b', 'a'];

// --- Colors ---
const COLORS = {
  p0: '#2563eb', // blue
  p1: '#dc2626', // red
  pitFill: '#fefce8', // warm cream
  pitStroke: '#3c3b37',
  highlight: '#facc15', // gold for played pit
  capture: '#ef4444', // red for captured
  delta: '#16a34a', // green for positive delta
};

// Helper to get the seed count at a display position
function getSeedCount(obs: OwareObs, row: number, col: number): number {
  if (row === 0) {
    // P1's row, displayed reversed: col 0 = pit 5, col 1 = pit 4, ...
    return obs.board[1][5 - col];
  }
  // P0's row: col 0 = pit 0, col 1 = pit 1, ...
  return obs.board[0][col];
}

// --- Renderer ---

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <canvas></canvas>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;
  if (!canvas || !replay) return;

  // Size canvas
  canvas.width = 0;
  canvas.height = 0;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;

  const c = canvas.getContext('2d');
  if (!c) return;

  const currentStep = steps[step];
  const obs = getObservation(currentStep);
  const prevObs = step > 0 ? getObservation(steps[step - 1]) : null;

  if (!obs) {
    statusContainer.textContent = 'Waiting for game data...';
    return;
  }

  const terminal = obs.is_terminal;

  // --- Header: player names and scores ---
  const p0Active = !terminal && obs.current_player === 0;
  const p1Active = !terminal && obs.current_player === 1;
  header.innerHTML = `
    <span class="sketched-border" style="padding: 4px 14px; background-color: ${p1Active ? '#bdeeff' : 'white'}; font-weight: 700; transition: background-color 300ms;">
      <span style="color: ${COLORS.p1};">Player 2</span>
      <span style="color: #444343; margin-left: 6px;">${obs.scores[1]}</span>
    </span>
    <span style="color: #444343;">vs</span>
    <span class="sketched-border" style="padding: 4px 14px; background-color: ${p0Active ? '#bdeeff' : 'white'}; font-weight: 700; transition: background-color 300ms;">
      <span style="color: ${COLORS.p0};">Player 1</span>
      <span style="color: #444343; margin-left: 6px;">${obs.scores[0]}</span>
    </span>
  `;

  // --- Draw board on canvas ---
  const w = canvas.width;
  const h = canvas.height;
  c.clearRect(0, 0, w, h);

  const cols = 6;
  const rows = 2;
  const margin = 24;
  const labelSpace = 20;
  const boardW = w - margin * 2;
  const boardH = h - margin * 2 - labelSpace * 2;
  const cellW = boardW / cols;
  const cellH = boardH / rows;
  const pitRadius = Math.min(cellW, cellH) * 0.38;
  const boardX = margin;
  const boardY = margin + labelSpace;

  // Determine which display position was played (for highlighting)
  const lastAction = obs.last_action;
  let playedRow = -1;
  let playedCol = -1;
  if (lastAction) {
    if (lastAction.player === 0) {
      playedRow = 1;
      playedCol = lastAction.pit;
    } else {
      playedRow = 0;
      playedCol = 5 - lastAction.pit;
    }
  }

  // Score deltas
  const scoreDelta: [number, number] = [0, 0];
  if (prevObs) {
    scoreDelta[0] = obs.scores[0] - prevObs.scores[0];
    scoreDelta[1] = obs.scores[1] - prevObs.scores[1];
  }

  // Draw each pit
  const drawPit = (col: number, row: number, label: string) => {
    const cx = boardX + col * cellW + cellW / 2;
    const cy = boardY + row * cellH + cellH / 2;
    const seeds = getSeedCount(obs, row, col);
    const prevSeeds = prevObs ? getSeedCount(prevObs, row, col) : seeds;
    const isPlayed = row === playedRow && col === playedCol;
    const delta = seeds - prevSeeds;
    const wasCaptured = prevObs !== null && delta < 0 && !isPlayed;

    // Pit background
    c.beginPath();
    c.ellipse(cx, cy, pitRadius, pitRadius * 0.85, 0, 0, Math.PI * 2);
    c.fillStyle = isPlayed ? COLORS.highlight + '44' : '#fefce8';
    c.fill();
    c.strokeStyle = isPlayed ? COLORS.highlight : COLORS.pitStroke;
    c.lineWidth = isPlayed ? 2.5 : 1;
    c.setLineDash(isPlayed ? [] : [4, 3]);
    c.stroke();
    c.setLineDash([]);

    // Seed count
    c.fillStyle = '#050001';
    c.font = `bold ${Math.round(pitRadius * 0.7)}px Inter, sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(String(seeds), cx, cy);

    // Pit label below/above
    c.fillStyle = '#444343';
    c.font = `${Math.round(pitRadius * 0.35)}px Inter, sans-serif`;
    if (row === 1) {
      c.fillText(label, cx, cy + pitRadius + 14);
    } else {
      c.fillText(label, cx, cy - pitRadius - 10);
    }

    // Delta badge
    if (delta !== 0 && prevObs) {
      const badgeX = cx + pitRadius * 0.65;
      const badgeY = cy - pitRadius * 0.65;
      const text = delta > 0 ? `+${delta}` : String(delta);
      const color = wasCaptured ? COLORS.capture : delta > 0 ? COLORS.delta : COLORS.capture;

      c.fillStyle = color;
      c.font = `bold ${Math.round(pitRadius * 0.32)}px Inter, sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(text, badgeX, badgeY);
    }
  };

  // Top row: Player 1's pits (displayed reversed)
  for (let col = 0; col < cols; col++) {
    drawPit(col, 0, PIT_LABELS_P1[col]);
  }

  // Bottom row: Player 0's pits
  for (let col = 0; col < cols; col++) {
    drawPit(col, 1, PIT_LABELS_P0[col]);
  }

  // Player indicators on the sides
  c.font = `bold ${Math.round(pitRadius * 0.35)}px Inter, sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'middle';

  c.fillStyle = COLORS.p1;
  c.fillText('P2', boardX - 4, boardY + cellH / 2);

  c.fillStyle = COLORS.p0;
  c.fillText('P1', boardX - 4, boardY + cellH + cellH / 2);

  // Score deltas on right side
  if (scoreDelta[0] > 0 || scoreDelta[1] > 0) {
    c.font = `bold ${Math.round(pitRadius * 0.35)}px Inter, sans-serif`;
    c.textAlign = 'right';
    if (scoreDelta[1] > 0) {
      c.fillStyle = COLORS.delta;
      c.fillText(`+${scoreDelta[1]}`, boardX + boardW + margin - 4, boardY + cellH / 2);
    }
    if (scoreDelta[0] > 0) {
      c.fillStyle = COLORS.delta;
      c.fillText(`+${scoreDelta[0]}`, boardX + boardW + margin - 4, boardY + cellH + cellH / 2);
    }
  }

  // --- Status ---
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg: string;
    if (rewards[0] > rewards[1]) msg = `Game Over — Player 1 wins! (${obs.scores[0]}–${obs.scores[1]})`;
    else if (rewards[1] > rewards[0]) msg = `Game Over — Player 2 wins! (${obs.scores[1]}–${obs.scores[0]})`;
    else msg = `Game Over — Draw (${obs.scores[0]}–${obs.scores[1]})`;
    statusContainer.textContent = msg;
    statusContainer.style.fontWeight = '700';
  } else if (lastAction) {
    const playerLabel = lastAction.player === 0 ? 'Player 1' : 'Player 2';
    let moveText = `${playerLabel} sowed from pit ${lastAction.pit_name}`;
    const sd = scoreDelta[lastAction.player];
    if (sd > 0) moveText += ` — captured ${sd}!`;
    statusContainer.textContent = moveText;
  } else {
    statusContainer.textContent = `Player ${obs.current_player + 1}'s turn`;
  }
}
