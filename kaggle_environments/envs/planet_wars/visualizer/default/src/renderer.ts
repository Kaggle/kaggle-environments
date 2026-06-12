import type { RendererOptions } from '@kaggle-environments/core';
import { getStepData } from '@kaggle-environments/core';

// Owner ids in this env: 0 = neutral, 1 = player 1, 2 = player 2.
// We map agent index (0, 1) to owner id via owner = agentIndex + 1.
const PLAYER_COLORS = ['#0072B2', '#D55E00']; // Wong palette, colorblind-safe
const NEUTRAL_COLOR = '#7a7a72';
const NEUTRAL_FILL = '#e9e3cc';
const BOARD_BG = '#f5f1e2';

interface Planet {
  id: number;
  x: number;
  y: number;
  owner: number; // 0 neutral, 1, 2
  ships: number;
  growth: number;
}

interface Fleet {
  owner: number; // 1 or 2
  ships: number;
  source: number;
  dest: number;
  totalTrip: number;
  turnsRemaining: number;
}

function parsePlanet(p: number[]): Planet {
  return { id: p[0], x: p[1], y: p[2], owner: p[3], ships: p[4], growth: p[5] };
}

function parseFleet(f: number[]): Fleet {
  return {
    owner: f[0],
    ships: f[1],
    source: f[2],
    dest: f[3],
    totalTrip: f[4],
    turnsRemaining: f[5],
  };
}

function ownerColor(owner: number): string {
  if (owner === 1 || owner === 2) return PLAYER_COLORS[owner - 1];
  return NEUTRAL_COLOR;
}

function ownerFill(owner: number): string {
  if (owner === 1 || owner === 2) return PLAYER_COLORS[owner - 1];
  return NEUTRAL_FILL;
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent, agents } = options;

  const stepData = getStepData(replay, step);
  if (!stepData || !(stepData as any)[0]?.observation) return;

  const obs = (stepData as any)[0].observation;
  const planets: Planet[] = (obs.planets || []).map(parsePlanet);
  const fleets: Fleet[] = (obs.fleets || []).map(parseFleet);

  let prevObs: any = null;
  if (step > 0) {
    const prevStep = getStepData(replay, step - 1);
    if (prevStep) prevObs = (prevStep as any)[0]?.observation;
  }
  const prevPlanetById = new Map<number, Planet>();
  if (prevObs?.planets) {
    for (const p of prevObs.planets) {
      const pp = parsePlanet(p);
      prevPlanetById.set(pp.id, pp);
    }
  }

  // Per-player totals (ships on planets + in fleets).
  const totals: Record<number, number> = { 1: 0, 2: 0 };
  for (const p of planets) {
    if (p.owner in totals) totals[p.owner] += Math.floor(p.ships);
  }
  for (const f of fleets) {
    if (f.owner in totals) totals[f.owner] += Math.floor(f.ships);
  }
  const alive: Record<number, boolean> = { 1: false, 2: false };
  for (const p of planets) if (p.owner in alive) alive[p.owner] = true;
  for (const f of fleets) if (f.owner in alive) alive[f.owner] = true;

  // Game-over check: when both agents have status DONE.
  const statuses = Array.isArray(stepData) ? (stepData as any[]).map((s: any) => s?.status) : [];
  const isGameOver = statuses.every((s: string) => s === 'DONE' || s === 'INVALID');

  // ---- DOM ----
  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="canvas-wrapper">
        <canvas></canvas>
      </div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const canvasWrapper = canvas.parentElement as HTMLDivElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;
  if (!canvas || !replay) return;

  // Header: one card per player, highlighted if still alive.
  const playerNames: string[] = [];
  for (let i = 0; i < 2; i++) {
    const a = agents?.[i];
    playerNames.push(a?.name || `Player ${i + 1}`);
  }
  const headerParts: string[] = [];
  for (let i = 0; i < 2; i++) {
    const owner = i + 1;
    const isAlive = alive[owner];
    const cls = `player-card ${isAlive ? 'active' : 'eliminated'}`;
    headerParts.push(
      `<span class="${cls}">` +
        `<span class="color-dot" style="background-color: ${PLAYER_COLORS[i]}"></span>` +
        `${playerNames[i]}` +
        `<span class="ship-count">${totals[owner]} ships</span>` +
        `</span>`
    );
    if (i === 0) headerParts.push(`<span class="vs-label">vs</span>`);
  }
  header.innerHTML = headerParts.join('');

  // Status: turn N / max and per-player breakdown.
  const planetCounts: Record<number, number> = { 0: 0, 1: 0, 2: 0 };
  for (const p of planets) planetCounts[p.owner] = (planetCounts[p.owner] || 0) + 1;
  const maxTurns = (replay as any).configuration?.episodeSteps ?? '';
  const turnText = `Turn ${step}${maxTurns ? ' / ' + maxTurns : ''}`;
  const breakdown = `P1 planets: ${planetCounts[1] || 0} · neutral: ${planetCounts[0] || 0} · P2 planets: ${planetCounts[2] || 0}`;
  statusContainer.textContent = `${turnText}   —   ${breakdown}`;

  // ---- Canvas ----
  const dpr = window.devicePixelRatio || 1;
  const cssSize = Math.max(100, Math.floor(canvas.getBoundingClientRect().width));
  canvas.width = Math.round(cssSize * dpr);
  canvas.height = Math.round(cssSize * dpr);
  const c = canvas.getContext('2d');
  if (!c) return;
  c.scale(dpr, dpr);
  const w = cssSize;
  c.clearRect(0, 0, w, w);

  // Compute world bounds from planets (with a small margin).
  if (planets.length === 0) return;
  let minX = planets[0].x;
  let maxX = planets[0].x;
  let minY = planets[0].y;
  let maxY = planets[0].y;
  for (const p of planets) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const worldW = Math.max(1, maxX - minX);
  const worldH = Math.max(1, maxY - minY);
  const worldSize = Math.max(worldW, worldH);
  // Margin in world units so the largest planets don't get clipped.
  const margin = worldSize * 0.1;
  const worldExtent = worldSize + 2 * margin;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const scale = w / worldExtent;
  const toScreenX = (x: number) => (x - cx) * scale + w / 2;
  const toScreenY = (y: number) => (y - cy) * scale + w / 2;

  // Subtle dotted backdrop grid to suggest "space" without dominating.
  c.fillStyle = BOARD_BG;
  c.fillRect(0, 0, w, w);
  c.fillStyle = 'rgba(60, 59, 55, 0.18)';
  const gridStep = w / 16;
  for (let gx = gridStep / 2; gx < w; gx += gridStep) {
    for (let gy = gridStep / 2; gy < w; gy += gridStep) {
      c.beginPath();
      c.arc(gx, gy, 0.8, 0, Math.PI * 2);
      c.fill();
    }
  }

  // Fleets: dashed travel lines from source to destination, with a chevron
  // at the interpolated current position.
  for (const f of fleets) {
    if (f.source < 0 || f.source >= planets.length) continue;
    if (f.dest < 0 || f.dest >= planets.length) continue;
    const src = planets[f.source];
    const dst = planets[f.dest];
    const sx = toScreenX(src.x);
    const sy = toScreenY(src.y);
    const dx = toScreenX(dst.x);
    const dy = toScreenY(dst.y);
    const color = ownerColor(f.owner);

    // Travel line.
    c.save();
    c.strokeStyle = color;
    c.globalAlpha = 0.35;
    c.lineWidth = 1.2;
    c.setLineDash([5, 4]);
    c.beginPath();
    c.moveTo(sx, sy);
    c.lineTo(dx, dy);
    c.stroke();
    c.restore();

    // Fleet position (chevron).
    const t = f.totalTrip > 0 ? 1 - f.turnsRemaining / f.totalTrip : 1;
    const fx = sx + (dx - sx) * t;
    const fy = sy + (dy - sy) * t;
    const angle = Math.atan2(dy - sy, dx - sx);
    const size = Math.max(4, Math.min(12, 3 + Math.log(1 + f.ships) * 1.5));
    c.save();
    c.translate(fx, fy);
    c.rotate(angle);
    c.beginPath();
    c.moveTo(size, 0);
    c.lineTo(-size * 0.7, -size * 0.6);
    c.lineTo(-size * 0.3, 0);
    c.lineTo(-size * 0.7, size * 0.6);
    c.closePath();
    c.fillStyle = color;
    c.fill();
    c.strokeStyle = '#050001';
    c.lineWidth = 0.8;
    c.stroke();

    // Ship count above the chevron.
    c.rotate(-angle);
    c.font = `600 ${Math.max(9, size + 2)}px Inter, sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'bottom';
    c.fillStyle = '#050001';
    c.fillText(String(f.ships), 0, -size - 2);
    c.restore();
  }

  // Planets: radius scales with growth rate (0..5). Show ship count inside.
  for (const p of planets) {
    const px = toScreenX(p.x);
    const py = toScreenY(p.y);
    // Planet radius in screen px. Growth 0 stays visible, growth 5 is the
    // largest. Tuned so even ~30 planets fit a 512 board.
    const r = Math.max(8, 9 + p.growth * 3.2);

    // Detect ownership change since previous step.
    const prev = prevPlanetById.get(p.id);
    const ownerChanged = !!prev && prev.owner !== p.owner;

    // Body.
    c.beginPath();
    c.arc(px, py, r, 0, Math.PI * 2);
    c.fillStyle = ownerFill(p.owner);
    c.globalAlpha = p.owner === 0 ? 0.55 : 0.85;
    c.fill();
    c.globalAlpha = 1;

    // Sketched border.
    c.beginPath();
    c.arc(px, py, r, 0, Math.PI * 2);
    c.strokeStyle = ownerColor(p.owner);
    c.lineWidth = ownerChanged ? 2.5 : 1.2;
    if (p.owner === 0) c.setLineDash([3, 3]);
    else c.setLineDash([]);
    c.stroke();
    c.setLineDash([]);

    // Capture highlight ring.
    if (ownerChanged) {
      c.beginPath();
      c.arc(px, py, r + 4, 0, Math.PI * 2);
      c.strokeStyle = ownerColor(p.owner);
      c.lineWidth = 1.2;
      c.setLineDash([2, 3]);
      c.stroke();
      c.setLineDash([]);
    }

    // Growth-rate tick marks around the rim (one per +1 growth).
    if (p.growth > 0) {
      c.strokeStyle = '#3c3b37';
      c.globalAlpha = 0.7;
      c.lineWidth = 1;
      for (let g = 0; g < p.growth; g++) {
        const a = (g / 5) * Math.PI * 2 - Math.PI / 2;
        const rx1 = px + Math.cos(a) * (r + 2);
        const ry1 = py + Math.sin(a) * (r + 2);
        const rx2 = px + Math.cos(a) * (r + 5);
        const ry2 = py + Math.sin(a) * (r + 5);
        c.beginPath();
        c.moveTo(rx1, ry1);
        c.lineTo(rx2, ry2);
        c.stroke();
      }
      c.globalAlpha = 1;
    }

    // Ship count.
    const shipText = String(Math.floor(p.ships));
    c.font = `700 ${Math.max(9, r * 0.9)}px Inter, sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillStyle = p.owner === 0 ? '#050001' : '#ffffff';
    c.fillText(shipText, px, py);

    // Delta vs previous step (+N / -N) above the planet.
    if (prev) {
      const delta = Math.floor(p.ships) - Math.floor(prev.ships);
      if (delta !== 0) {
        c.font = `600 ${Math.max(8, r * 0.55)}px Inter, sans-serif`;
        c.fillStyle = delta > 0 ? '#1b7f4a' : '#b0431e';
        c.textBaseline = 'bottom';
        c.fillText(delta > 0 ? `+${delta}` : `${delta}`, px, py - r - 2);
      }
    }
  }

  // Game-over overlay.
  if (isGameOver) {
    const winner =
      totals[1] > totals[2] && alive[1]
        ? 1
        : totals[2] > totals[1] && alive[2]
          ? 2
          : alive[1] && !alive[2]
            ? 1
            : alive[2] && !alive[1]
              ? 2
              : 0;
    let resultText = 'Draw';
    if (winner === 1) resultText = `${playerNames[0]} wins`;
    else if (winner === 2) resultText = `${playerNames[1]} wins`;

    const overlay = document.createElement('div');
    overlay.className = 'game-over-overlay';
    overlay.innerHTML = `
      <div class="game-over-modal">
        <h2>Game Over</h2>
        <div class="result-text">${resultText}</div>
        <div class="score-line">
          ${playerNames[0]}: ${totals[1]} &mdash; ${playerNames[1]}: ${totals[2]}
        </div>
      </div>
    `;
    canvasWrapper.appendChild(overlay);
  }
}
