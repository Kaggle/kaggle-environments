import type { RendererOptions } from '@kaggle-environments/core';
import { getStepData } from '@kaggle-environments/core';

// Game constants
const BOARD_SIZE = 100;
const CENTER = 50;
const SUN_RADIUS = 10;

// Player colors (bright for dark background)
const PLAYER_COLORS = ['#FF4444', '#4a9eff', '#44FF44', '#FFFF44'];
const NEUTRAL_COLOR = '#666666';

// Text size presets: [planetFont, deltaFont, fleetFont, stepFont]
const TEXT_SIZES: Record<string, number> = {
  small: 0.7,
  medium: 1.0,
  large: 1.4,
};

function getPlayerColor(owner: number): string {
  if (owner < 0 || owner >= PLAYER_COLORS.length) return NEUTRAL_COLOR;
  return PLAYER_COLORS[owner];
}

interface Planet {
  id: number;
  owner: number;
  x: number;
  y: number;
  radius: number;
  ships: number;
  production: number;
}

interface Fleet {
  id: number;
  owner: number;
  x: number;
  y: number;
  angle: number;
  fromPlanetId: number;
  ships: number;
}

function parsePlanet(p: number[]): Planet {
  return { id: p[0], owner: p[1], x: p[2], y: p[3], radius: p[4], ships: p[5], production: p[6] };
}

function parseFleet(f: number[]): Fleet {
  return { id: f[0], owner: f[1], x: f[2], y: f[3], angle: f[4], fromPlanetId: f[5], ships: f[6] };
}

// --- Settings persistence via data attributes on parent ---
interface Settings {
  showFleetNumbers: boolean;
  showProductionDots: boolean;
  textSize: string; // 'small' | 'medium' | 'large'
}

function getSettings(parent: HTMLElement): Settings {
  return {
    showFleetNumbers: parent.dataset.showFleetNumbers !== 'false',
    showProductionDots: parent.dataset.showProductionDots !== 'false',
    textSize: parent.dataset.textSize || 'medium',
  };
}

function setSetting(parent: HTMLElement, key: string, value: string) {
  parent.dataset[key] = value;
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent, agents } = options;

  const stepData = getStepData(replay, step);
  if (!stepData || !(stepData as any)[0]?.observation) return;

  const settings = getSettings(parent);
  const textScale = TEXT_SIZES[settings.textSize] || 1.0;

  const obs = (stepData as any)[0].observation;
  const planets: Planet[] = (obs.planets || []).map(parsePlanet);
  const fleets: Fleet[] = (obs.fleets || []).map(parseFleet);
  const cometPlanetIds = new Set<number>(obs.comet_planet_ids || []);
  const numAgents = (replay as any).info?.TeamNames?.length || 2;

  // Previous step for diff detection
  let prevObs: any = null;
  if (step > 0) {
    const prevStep = getStepData(replay, step - 1);
    if (prevStep) prevObs = (prevStep as any)[0]?.observation;
  }

  // Build previous planet map for diff
  const prevPlanetMap = new Map<number, Planet>();
  if (prevObs?.planets) {
    for (const p of prevObs.planets) {
      const pp = parsePlanet(p);
      prevPlanetMap.set(pp.id, pp);
    }
  }

  // Detect game over
  const statuses = (stepData as any).map ? Array.from(stepData as any).map((s: any) => s?.status) : [];
  const isGameOver = statuses.some((s: string) => s === 'DONE');

  // Compute scores
  const playerScores: number[] = [];
  for (let i = 0; i < numAgents; i++) {
    let score = 0;
    for (const p of planets) {
      if (p.owner === i) score += Math.floor(p.ships);
    }
    for (const f of fleets) {
      if (f.owner === i) score += Math.floor(f.ships);
    }
    playerScores.push(score);
  }

  // Determine active players (those with planets or fleets)
  const activePlayers = new Set<number>();
  for (const p of planets) {
    if (p.owner >= 0) activePlayers.add(p.owner);
  }
  for (const f of fleets) {
    activePlayers.add(f.owner);
  }

  // Rebuild DOM
  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="controls-bar"></div>
      <div class="canvas-wrapper">
        <canvas></canvas>
      </div>
    </div>
  `;

  const header = parent.querySelector('.header') as HTMLDivElement;
  const controlsBar = parent.querySelector('.controls-bar') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const canvasWrapper = canvas.parentElement as HTMLDivElement;
  if (!canvas || !replay) return;

  // --- Controls bar ---
  const fleetBtnActive = settings.showFleetNumbers ? ' active' : '';
  const prodBtnActive = settings.showProductionDots ? ' active' : '';
  controlsBar.innerHTML =
    `<button class="ctrl-btn${fleetBtnActive}" data-action="toggle-fleet-numbers">` +
    `Fleet #</button>` +
    `<button class="ctrl-btn${prodBtnActive}" data-action="toggle-production-dots">` +
    `Production</button>` +
    `<span class="ctrl-group">` +
    `<span class="ctrl-label">Text:</span>` +
    ['small', 'medium', 'large']
      .map((sz) => {
        const active = settings.textSize === sz ? ' active' : '';
        return `<button class="ctrl-btn${active}" data-action="text-size" data-value="${sz}">${sz[0].toUpperCase() + sz.slice(1)}</button>`;
      })
      .join('') +
    `</span>`;

  // Wire up control event listeners (these mutate data attrs and re-render)
  controlsBar.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest('[data-action]') as HTMLElement | null;
    if (!btn) return;
    const action = btn.dataset.action;
    if (action === 'toggle-fleet-numbers') {
      setSetting(parent, 'showFleetNumbers', settings.showFleetNumbers ? 'false' : 'true');
      renderer(options);
    } else if (action === 'toggle-production-dots') {
      setSetting(parent, 'showProductionDots', settings.showProductionDots ? 'false' : 'true');
      renderer(options);
    } else if (action === 'text-size') {
      setSetting(parent, 'textSize', btn.dataset.value || 'medium');
      renderer(options);
    }
  });

  // Size canvas: always square, fill available space, handle DPR
  const dpr = window.devicePixelRatio || 1;
  const wrapperRect = canvasWrapper.getBoundingClientRect();
  const cssSize = Math.max(100, Math.floor(Math.min(wrapperRect.width, wrapperRect.height)));
  canvas.style.width = `${cssSize}px`;
  canvas.style.height = `${cssSize}px`;
  canvas.style.position = 'absolute';
  canvas.style.left = `${(wrapperRect.width - cssSize) / 2}px`;
  canvas.style.top = `${(wrapperRect.height - cssSize) / 2}px`;
  canvas.width = Math.round(cssSize * dpr);
  canvas.height = Math.round(cssSize * dpr);

  const c = canvas.getContext('2d');
  if (!c) return;
  c.scale(dpr, dpr);

  // All drawing uses CSS pixels; the DPR scaling handles sharpness
  const w = cssSize;
  const scale = w / BOARD_SIZE;

  // --- Header: player cards ---
  const playerNames: string[] = [];
  for (let i = 0; i < numAgents; i++) {
    const agent = agents?.[i];
    playerNames.push(agent?.name || `Player ${i + 1}`);
  }

  const headerParts: string[] = [];
  for (let i = 0; i < numAgents; i++) {
    const isActive = activePlayers.has(i);
    const activeClass = isActive ? ' active' : '';
    headerParts.push(
      `<span class="player-card${activeClass}">` +
        `<span class="color-dot" style="background-color: ${PLAYER_COLORS[i]}"></span>` +
        `${playerNames[i]}` +
        `<span class="ship-count">${playerScores[i]}</span>` +
        `</span>`
    );
    if (i < numAgents - 1) {
      headerParts.push(`<span style="color: #666;">vs</span>`);
    }
  }
  header.innerHTML = headerParts.join('');

  // --- Draw game board on canvas ---
  c.fillStyle = '#000000';
  c.fillRect(0, 0, w, w);

  // Draw sun with glow
  const sunX = CENTER * scale;
  const sunY = CENTER * scale;
  const sunR = SUN_RADIUS * scale;

  const glow = c.createRadialGradient(sunX, sunY, sunR * 0.5, sunX, sunY, sunR * 2.5);
  glow.addColorStop(0, 'rgba(255, 200, 50, 0.6)');
  glow.addColorStop(0.5, 'rgba(255, 150, 20, 0.2)');
  glow.addColorStop(1, 'rgba(255, 100, 0, 0)');
  c.fillStyle = glow;
  c.fillRect(0, 0, w, w);

  // Sun body
  c.beginPath();
  c.arc(sunX, sunY, sunR, 0, Math.PI * 2);
  c.fillStyle = '#FFB800';
  c.fill();
  c.strokeStyle = '#FFD700';
  c.lineWidth = 1;
  c.stroke();

  // Draw comet trails
  if (obs.comets) {
    for (const group of obs.comets) {
      const idx = group.path_index;
      for (let i = 0; i < group.planet_ids.length; i++) {
        const path = group.paths[i];
        const tailLen = Math.min(idx + 1, path.length, 5);
        if (tailLen < 2) continue;
        for (let t = 1; t < tailLen; t++) {
          const pi = idx - t;
          if (pi < 0) break;
          const alpha = 0.4 * (1 - t / tailLen);
          c.beginPath();
          c.moveTo(path[pi + 1][0] * scale, path[pi + 1][1] * scale);
          c.lineTo(path[pi][0] * scale, path[pi][1] * scale);
          c.strokeStyle = `rgba(200, 220, 255, ${alpha})`;
          c.lineWidth = ((2.5 - (1.5 * t) / tailLen) * scale) / 5;
          c.lineCap = 'round';
          c.stroke();
        }
      }
    }
  }

  // Draw planets
  for (const planet of planets) {
    const px = planet.x * scale;
    const py = planet.y * scale;
    const pr = planet.radius * scale;
    const color = getPlayerColor(planet.owner);
    const isComet = cometPlanetIds.has(planet.id);

    // Check if ownership changed from previous step
    const prev = prevPlanetMap.get(planet.id);
    const ownerChanged = prev && prev.owner !== planet.owner;

    // Planet body
    c.beginPath();
    c.arc(px, py, pr, 0, Math.PI * 2);
    c.fillStyle = color;
    c.globalAlpha = planet.owner >= 0 ? 0.85 : 0.5;
    c.fill();
    c.globalAlpha = 1;

    // Border
    c.beginPath();
    c.arc(px, py, pr, 0, Math.PI * 2);
    c.strokeStyle = isComet ? '#88ccff' : '#555';
    c.lineWidth = isComet ? 2 : 1;
    c.stroke();

    // Ownership change highlight
    if (ownerChanged) {
      c.beginPath();
      c.arc(px, py, pr + 3, 0, Math.PI * 2);
      c.strokeStyle = color;
      c.lineWidth = 2;
      c.stroke();
    }

    // Production dots (small dots around planet)
    if (settings.showProductionDots && planet.owner >= 0 && planet.production > 0) {
      const dotR = Math.max(1, scale * 0.3);
      for (let d = 0; d < planet.production; d++) {
        const dotAngle = (d / planet.production) * Math.PI * 2 - Math.PI / 2;
        const dotDist = pr + dotR + 2;
        const dx = px + Math.cos(dotAngle) * dotDist;
        const dy = py + Math.sin(dotAngle) * dotDist;
        c.beginPath();
        c.arc(dx, dy, dotR, 0, Math.PI * 2);
        c.fillStyle = '#aaa';
        c.fill();
      }
    }
  }

  // Draw fleets as chevrons
  for (const fleet of fleets) {
    const fx = fleet.x * scale;
    const fy = fleet.y * scale;
    const color = getPlayerColor(fleet.owner);
    const sz = (0.4 + (2.0 * Math.log(Math.max(1, fleet.ships))) / Math.log(1000)) * scale;

    c.save();
    c.translate(fx, fy);
    c.rotate(fleet.angle);
    c.beginPath();
    c.moveTo(sz, 0);
    c.lineTo(-sz, -sz * 0.6);
    c.lineTo(-sz * 0.3, 0);
    c.lineTo(-sz, sz * 0.6);
    c.closePath();
    c.fillStyle = color;
    c.globalAlpha = 0.85;
    c.fill();
    c.globalAlpha = 1;
    c.strokeStyle = '#222';
    c.lineWidth = 0.5;
    c.stroke();
    c.restore();
  }

  // Draw ship counts on planets
  const planetFontSize = Math.max(8, scale * 1.8 * textScale);
  const deltaFontSize = Math.max(6, scale * 1.2 * textScale);
  c.font = `bold ${planetFontSize}px Inter, sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  for (const planet of planets) {
    const px = planet.x * scale;
    const py = planet.y * scale;
    const shipText = Math.floor(planet.ships).toString();

    c.font = `bold ${planetFontSize}px Inter, sans-serif`;
    c.fillStyle = '#000000';
    c.fillText(shipText, px + 0.5, py + 0.5);
    c.fillStyle = '#ffffff';
    c.fillText(shipText, px, py);

    // Ship count delta (only when production display is on)
    if (settings.showProductionDots) {
      const prev = prevPlanetMap.get(planet.id);
      if (prev) {
        const delta = Math.floor(planet.ships) - Math.floor(prev.ships);
        if (delta !== 0) {
          const deltaText = delta > 0 ? `+${delta}` : `${delta}`;
          c.font = `bold ${deltaFontSize}px Inter, sans-serif`;
          c.fillStyle = delta > 0 ? '#44FF44' : '#FF4444';
          c.fillText(deltaText, px, py - planet.radius * scale - deltaFontSize);
        }
      }
    }
  }

  // Fleet ship counts
  if (settings.showFleetNumbers) {
    const fleetFontSize = Math.max(6, scale * 1.2 * textScale);
    c.font = `${fleetFontSize}px Inter, sans-serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    for (const fleet of fleets) {
      const fx = fleet.x * scale;
      const fy = fleet.y * scale;
      const labelOffset = fleet.y >= 50 ? -scale * 2.5 : scale * 2.5;
      c.fillStyle = getPlayerColor(fleet.owner);
      c.fillText(Math.floor(fleet.ships).toString(), fx, fy + labelOffset);
    }
  }

  // Step indicator
  const stepFontSize = Math.max(8, scale * 1.5 * textScale);
  c.font = `${stepFontSize}px Inter, sans-serif`;
  c.textAlign = 'left';
  c.textBaseline = 'top';
  c.fillStyle = '#888';
  c.fillText(`Step ${step}`, 6, 6);

  // Game over overlay
  if (isGameOver) {
    const maxScore = Math.max(...playerScores);
    const winners = playerScores.reduce<number[]>((acc, s, i) => {
      if (s === maxScore) acc.push(i);
      return acc;
    }, []);
    const winnerText = winners.length > 1 ? 'Draw!' : `${playerNames[winners[0]]} wins!`;

    const overlay = document.createElement('div');
    overlay.className = 'game-over-overlay';
    overlay.innerHTML = `
      <div class="game-over-modal">
        <h2>Game Over</h2>
        <div class="result-text">${winnerText}</div>
        <div style="margin-top: 8px; font-size: 0.85rem; color: #888;">
          ${playerScores.map((s, i) => `${playerNames[i]}: ${s}`).join(' &mdash; ')}
        </div>
      </div>
    `;
    canvasWrapper.appendChild(overlay);
  }
}
