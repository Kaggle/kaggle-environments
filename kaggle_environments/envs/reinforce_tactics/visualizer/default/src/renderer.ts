import type { RendererOptions } from '@kaggle-environments/core';
import {
  getSpriteTheme,
  getStructureSprite,
  getTerrainSprite,
  getUnitSprite,
  isReady,
  onSpritesLoad,
  setSpriteTheme,
} from './sprites';
import type { SpriteTheme } from './sprites';

interface Unit {
  type: string;
  owner: number;
  x: number;
  y: number;
  hp: number;
  maxHp: number;
  canMove?: boolean;
  canAttack?: boolean;
  paralyzedTurns?: number;
  isHasted?: boolean;
  distanceMoved?: number;
  defenceBuffTurns?: number;
  attackBuffTurns?: number;
}

interface Structure {
  x: number;
  y: number;
  type: string;
  owner: number;
  hp: number;
  maxHp: number;
}

interface RTObservation {
  board: string[][];
  units: Unit[];
  structures: Structure[];
  gold: [number, number];
  turnNumber: number;
  mapWidth: number;
  mapHeight: number;
}

interface RTAgentStep {
  observation: Partial<RTObservation> & Record<string, any>;
  action: any;
  reward: number | null;
  status: string;
}

type RTStep = RTAgentStep[];

const TERRAIN_COLORS: Record<string, string> = {
  p: '#dfe9c0', // grass
  f: '#90b779', // forest
  m: '#c2a987', // mountain
  w: '#8cc1de', // water
  r: '#e2d4ad', // road
  b: '#f0dba1', // building (base)
  h: '#f5cf72', // HQ (base)
  t: '#dcd2ad', // tower (base)
  o: '#6093b4', // ocean
};

// Player accent colours per art set. The game art follows the main
// repository's PLAYER_COLORS (Player 1 red, Player 2 blue); the kaggle
// placeholder art keeps the colours it originally shipped with.
const THEME_PLAYER_COLORS: Record<SpriteTheme, Record<number, string>> = {
  game: {
    0: '#888888',
    1: '#ff3232', // red (255, 50, 50)
    2: '#4d79ff', // blue (77, 121, 255)
  },
  kaggle: {
    0: '#888888',
    1: '#2f5fa1', // blue
    2: '#b03939', // red
  },
};

const THEME_PLAYER_LIGHT: Record<SpriteTheme, Record<number, string>> = {
  game: {
    0: '#dcdcdc',
    1: '#ecbcbc',
    2: '#bcd0ec',
  },
  kaggle: {
    0: '#dcdcdc',
    1: '#bcd0ec',
    2: '#ecbcbc',
  },
};

const STRUCT_NAMES: Record<string, string> = {
  h: '★',
  b: '■',
  t: '▲',
};

// Board codes that are capturable structures rather than plain terrain.
const STRUCT_TILE_CODES = new Set(['h', 'b', 't']);

// Latest renderer options, used to re-render once async sprite loads
// finish and when the sprite art set is toggled.
let lastOptions: RendererOptions | null = null;
onSpritesLoad(() => {
  if (lastOptions) renderer(lastOptions);
});

function escapeHtml(s: string): string {
  return s.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string
  );
}

function getObservation(step: RTStep | undefined): RTObservation | null {
  if (!step || !step[0]) return null;
  const obs = step[0].observation as RTObservation | undefined;
  if (!obs || !obs.board) return null;
  return obs;
}

function activePlayerIndex(step: RTStep | undefined): number | null {
  if (!step) return null;
  for (let i = 0; i < step.length; i++) {
    if (step[i]?.status === 'ACTIVE') return i;
  }
  return null;
}

function lastActorActions(step: RTStep | undefined): { actorIdx: number | null; actions: any[] } {
  if (!step) return { actorIdx: null, actions: [] };
  let actorIdx: number | null = null;
  let actions: any[] = [];
  for (let i = 0; i < step.length; i++) {
    const a = step[i]?.action;
    if (Array.isArray(a) && a.length > 0) {
      actorIdx = i;
      actions = a;
      break;
    }
  }
  return { actorIdx, actions };
}

function summariseActions(actions: any[]): string {
  if (!actions.length) return '';
  const counts: Record<string, number> = {};
  for (const a of actions) {
    if (!a || typeof a !== 'object') continue;
    const t = String(a.type ?? 'unknown');
    if (t === 'end_turn') continue;
    counts[t] = (counts[t] ?? 0) + 1;
  }
  const order = [
    'create_unit',
    'move',
    'attack',
    'seize',
    'heal',
    'cure',
    'paralyze',
    'haste',
    'defence_buff',
    'attack_buff',
  ];
  const parts: string[] = [];
  for (const t of order) {
    if (counts[t]) parts.push(`${counts[t]} ${t.replace('_', ' ')}`);
  }
  for (const t of Object.keys(counts)) {
    if (!order.includes(t)) parts.push(`${counts[t]} ${t}`);
  }
  return parts.join(', ') || 'end turn';
}

function buildHighlights(prevObs: RTObservation | null, curObs: RTObservation, actions: any[]) {
  const moved: Array<{ fromX: number; fromY: number; toX: number; toY: number }> = [];
  const attacked: Array<{ x: number; y: number }> = [];
  const created: Array<{ x: number; y: number }> = [];
  const seized: Array<{ x: number; y: number }> = [];

  for (const a of actions) {
    if (!a || typeof a !== 'object') continue;
    switch (a.type) {
      case 'move':
        moved.push({ fromX: a.from_x, fromY: a.from_y, toX: a.to_x, toY: a.to_y });
        break;
      case 'attack':
      case 'paralyze':
      case 'heal':
      case 'cure':
      case 'haste':
      case 'defence_buff':
      case 'attack_buff':
        attacked.push({ x: a.to_x, y: a.to_y });
        break;
      case 'create_unit':
        created.push({ x: a.x, y: a.y });
        break;
      case 'seize':
        seized.push({ x: a.x, y: a.y });
        break;
    }
  }

  // Fallback: if no actions list (e.g. step 0), diff state
  if (!actions.length && prevObs) {
    const prevByPos = new Map<string, Unit>();
    for (const u of prevObs.units) prevByPos.set(`${u.x},${u.y}`, u);
    for (const u of curObs.units) {
      const key = `${u.x},${u.y}`;
      if (!prevByPos.has(key)) created.push({ x: u.x, y: u.y });
    }
  }

  return { moved, attacked, created, seized };
}

export function renderer(options: RendererOptions) {
  lastOptions = options;
  const { step, replay, parent, agents } = options;
  const steps = (replay.steps as unknown as RTStep[]) || [];
  const curStep = steps[step];
  const prevStep = step > 0 ? steps[step - 1] : null;

  const artTheme = getSpriteTheme();
  const PLAYER_COLORS = THEME_PLAYER_COLORS[artTheme];
  const PLAYER_LIGHT = THEME_PLAYER_LIGHT[artTheme];
  const gameArt = artTheme === 'game';

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

  const curObs = getObservation(curStep);
  if (!canvas || !curObs) return;

  // --- Header ---
  const activeIdx = activePlayerIndex(curStep);
  const finalStep = curStep?.some((s) => s.status === 'DONE');
  const unitCounts: [number, number] = [0, 0];
  for (const u of curObs.units) {
    if (u.owner === 1) unitCounts[0]++;
    if (u.owner === 2) unitCounts[1]++;
  }
  const structCounts: [number, number] = [0, 0];
  for (const s of curObs.structures) {
    if (s.owner === 1) structCounts[0]++;
    if (s.owner === 2) structCounts[1]++;
  }
  const gold = curObs.gold ?? [0, 0];

  header.innerHTML = '';
  for (let i = 0; i < 2; i++) {
    const card = document.createElement('div');
    card.className = `player-card sketched-border${activeIdx === i && !finalStep ? ' active' : ''}`;
    const nameColor = PLAYER_COLORS[i + 1];
    const displayName = agents?.[i]?.name || `Player ${i + 1}`;
    card.innerHTML = `
      <div class="player-name" style="color: ${nameColor};">${escapeHtml(displayName)}</div>
      <div class="player-stats">
        <span>${gold[i]}g</span>
        <span>${unitCounts[i]} units</span>
        <span>${structCounts[i]} bldgs</span>
      </div>
    `;
    header.appendChild(card);
    if (i === 0) {
      const vs = document.createElement('span');
      vs.className = 'vs-label';
      vs.textContent = 'vs';
      header.appendChild(vs);
    }
  }

  // --- Canvas board ---
  canvas.width = 0;
  canvas.height = 0;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const c = canvas.getContext('2d');
  if (!c) return;
  c.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const mapW = curObs.mapWidth;
  const mapH = curObs.mapHeight;
  const cell = Math.floor(Math.min(W / mapW, H / mapH));
  const boardW = cell * mapW;
  const boardH = cell * mapH;
  const xOff = (W - boardW) / 2;
  const yOff = (H - boardH) / 2;

  c.clearRect(0, 0, W, H);

  // Lookup structure by position to colour terrain by ownership
  const structByPos = new Map<string, Structure>();
  for (const s of curObs.structures) structByPos.set(`${s.x},${s.y}`, s);

  // Draw terrain — sprite per tile, with the solid color as a fallback
  // while images are still loading. The game art is pixel art, so it is
  // scaled nearest-neighbour for a crisp look (matching the main game);
  // the placeholder art is painted at high resolution and looks better
  // smoothed.
  c.imageSmoothingEnabled = !gameArt;
  if (!gameArt) c.imageSmoothingQuality = 'medium';
  for (let y = 0; y < mapH; y++) {
    for (let x = 0; x < mapW; x++) {
      const tile = curObs.board[y]?.[x] ?? 'o';
      const px = xOff + x * cell;
      const py = yOff + y * cell;
      const struct = structByPos.get(`${x},${y}`);

      if (gameArt && STRUCT_TILE_CODES.has(tile)) {
        // Game art: a structure IS the tile — draw it full-cell, already
        // team-coloured by the palette swap (gray when neutral), exactly
        // like the main game's renderer.
        const sprite = getStructureSprite(tile, struct?.owner ?? 0);
        if (isReady(sprite)) {
          c.drawImage(sprite!, px, py, cell, cell);
          continue;
        }
      }

      const terrain = getTerrainSprite(tile);
      if (isReady(terrain)) {
        c.drawImage(terrain!, px, py, cell, cell);
      } else {
        c.fillStyle = TERRAIN_COLORS[tile] ?? '#cccccc';
        c.fillRect(px, py, cell, cell);
      }

      // Ownership tint for capturable structures (placeholder art only —
      // game-art structures are team-coloured by the sprite itself, and
      // this also covers the brief window before sprites finish loading).
      if (struct && struct.owner) {
        c.fillStyle = PLAYER_LIGHT[struct.owner] ?? '#dddddd';
        c.globalAlpha = 0.55;
        c.fillRect(px, py, cell, cell);
        c.globalAlpha = 1;
      }
    }
  }

  // Grid lines
  c.strokeStyle = '#3c3b37';
  c.lineWidth = 1;
  c.globalAlpha = 0.3;
  c.beginPath();
  for (let x = 0; x <= mapW; x++) {
    const px = xOff + x * cell;
    c.moveTo(px, yOff);
    c.lineTo(px, yOff + boardH);
  }
  for (let y = 0; y <= mapH; y++) {
    const py = yOff + y * cell;
    c.moveTo(xOff, py);
    c.lineTo(xOff + boardW, py);
  }
  c.stroke();
  c.globalAlpha = 1;

  // Structure sprites. With the game art the structure was already drawn
  // as its tile above; the placeholder art draws an icon on top of the
  // tinted tile here. Either way, fall back to a glyph while the PNG
  // hasn't finished loading.
  const structSize = cell * 0.78;
  c.font = `${Math.max(8, Math.floor(cell * 0.36))}px 'Inter', sans-serif`;
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  for (const s of curObs.structures) {
    const cx = xOff + s.x * cell + cell / 2;
    const cy = yOff + s.y * cell + cell / 2;
    const sprite = getStructureSprite(s.type, s.owner);
    if (isReady(sprite)) {
      if (!gameArt) {
        c.drawImage(sprite!, cx - structSize / 2, cy - structSize / 2, structSize, structSize);
      }
    } else {
      const label = STRUCT_NAMES[s.type] ?? '';
      if (label) {
        c.fillStyle = s.owner ? PLAYER_COLORS[s.owner] : '#555';
        c.fillText(label, cx, cy);
      }
    }
  }

  // Action highlights
  const prevObs = getObservation(prevStep ?? undefined);
  const { actorIdx: lastActorIdx, actions: acts } = lastActorActions(curStep);
  const hl = buildHighlights(prevObs, curObs, acts);

  // Move arrows
  c.lineWidth = Math.max(2, cell * 0.08);
  c.strokeStyle = 'rgba(50, 50, 50, 0.7)';
  for (const m of hl.moved) {
    const fx = xOff + m.fromX * cell + cell / 2;
    const fy = yOff + m.fromY * cell + cell / 2;
    const tx = xOff + m.toX * cell + cell / 2;
    const ty = yOff + m.toY * cell + cell / 2;
    c.beginPath();
    c.moveTo(fx, fy);
    c.lineTo(tx, ty);
    c.stroke();
    // origin marker
    c.fillStyle = 'rgba(50, 50, 50, 0.4)';
    c.beginPath();
    c.arc(fx, fy, cell * 0.12, 0, Math.PI * 2);
    c.fill();
  }

  // Attack/cast targets
  for (const t of hl.attacked) {
    const tx = xOff + t.x * cell + cell / 2;
    const ty = yOff + t.y * cell + cell / 2;
    c.strokeStyle = '#c62828';
    c.lineWidth = Math.max(2, cell * 0.08);
    c.beginPath();
    c.arc(tx, ty, cell * 0.45, 0, Math.PI * 2);
    c.stroke();
  }

  // Created units
  for (const t of hl.created) {
    const px = xOff + t.x * cell;
    const py = yOff + t.y * cell;
    c.strokeStyle = '#2e7d32';
    c.setLineDash([4, 3]);
    c.lineWidth = Math.max(1.5, cell * 0.06);
    c.strokeRect(px + 1, py + 1, cell - 2, cell - 2);
    c.setLineDash([]);
  }

  // Seized targets
  for (const t of hl.seized) {
    const px = xOff + t.x * cell;
    const py = yOff + t.y * cell;
    c.strokeStyle = '#fbc02d';
    c.lineWidth = Math.max(2, cell * 0.1);
    c.strokeRect(px + 1, py + 1, cell - 2, cell - 2);
  }

  // Units.
  //
  // Game art: team-coloured sprite framed by a player-coloured border,
  // matching the main game's renderer (sprites are palette-swapped, the
  // border marks ownership).
  //
  // Placeholder art: player-coloured disc as ownership indicator, then
  // the sprite on top.
  //
  // Both fall back to a disc with the letter glyph while sprites are
  // still loading.
  const unitRadius = cell * 0.4;
  const spriteSize = gameArt ? cell * 0.875 : cell * 0.78;
  const unitFont = Math.max(9, Math.floor(cell * 0.55));
  for (const u of curObs.units) {
    const px = xOff + u.x * cell + cell / 2;
    const py = yOff + u.y * cell + cell / 2;
    const color = PLAYER_COLORS[u.owner] ?? '#666';

    const sprite = getUnitSprite(u.type, u.owner);
    const spriteReady = isReady(sprite);

    if (gameArt && spriteReady) {
      // Player-coloured border around the unit's tile (main repo style)
      c.lineWidth = Math.max(1.5, cell * 0.0625);
      c.strokeStyle = color;
      c.strokeRect(
        xOff + u.x * cell + cell * 0.03 + 1,
        yOff + u.y * cell + cell * 0.03 + 1,
        cell * 0.94 - 2,
        cell * 0.94 - 2
      );
    } else {
      // Ownership disc
      c.fillStyle = color;
      c.beginPath();
      c.arc(px, py, unitRadius, 0, Math.PI * 2);
      c.fill();
      c.lineWidth = Math.max(1, cell * 0.04);
      c.strokeStyle = '#050001';
      c.stroke();
    }

    if (spriteReady) {
      c.drawImage(sprite!, px - spriteSize / 2, py - spriteSize / 2, spriteSize, spriteSize);
    } else {
      // Letter fallback
      c.fillStyle = '#ffffff';
      c.font = `700 ${unitFont}px 'Inter', sans-serif`;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(u.type, px, py + 1);
    }

    // HP bar
    const hpFrac = Math.max(0, Math.min(1, u.hp / Math.max(1, u.maxHp)));
    const barW = cell * 0.7;
    const barH = Math.max(2, cell * 0.08);
    const bx = px - barW / 2;
    const by = py + unitRadius + 1;
    c.fillStyle = '#3c3b37';
    c.fillRect(bx, by, barW, barH);
    c.fillStyle = hpFrac > 0.5 ? '#4caf50' : hpFrac > 0.25 ? '#ff9800' : '#e53935';
    c.fillRect(bx, by, barW * hpFrac, barH);

    // status badges (paralyzed, hasted, buffs)
    const badges: string[] = [];
    if (u.paralyzedTurns && u.paralyzedTurns > 0) badges.push('P');
    if (u.isHasted) badges.push('H');
    if (u.attackBuffTurns && u.attackBuffTurns > 0) badges.push('A');
    if (u.defenceBuffTurns && u.defenceBuffTurns > 0) badges.push('D');
    if (badges.length) {
      c.font = `700 ${Math.max(8, Math.floor(cell * 0.25))}px 'Inter', sans-serif`;
      c.fillStyle = '#050001';
      c.textAlign = 'left';
      c.fillText(badges.join(''), px + unitRadius * 0.6, py - unitRadius * 0.7);
    }
  }

  // --- Status ---
  const turnNum = curObs.turnNumber ?? 0;
  let statusHtml = '';
  if (finalStep) {
    const rewards = curStep.map((s) => s.reward ?? 0);
    let msg = '';
    let cls = 'status-container sketched-border winner';
    if (rewards[0] === 1) msg = `Player 1 wins!`;
    else if (rewards[1] === 1) msg = `Player 2 wins!`;
    else {
      msg = 'Draw';
      cls = 'status-container sketched-border';
    }
    statusContainer.className = cls;
    statusHtml = `<span>${msg}</span><span class="annotation">turn ${turnNum}</span>`;
  } else {
    statusContainer.className = 'status-container sketched-border';
    const summary = summariseActions(acts);
    const lastLabel =
      lastActorIdx !== null && summary ? `<span class="annotation">P${lastActorIdx + 1}: ${summary}</span>` : '';
    const actorLabel =
      activeIdx !== null
        ? `<span style="color: ${PLAYER_COLORS[activeIdx + 1]}; font-weight: 700;">P${activeIdx + 1}</span> to act`
        : '';
    statusHtml = `<span>Turn ${turnNum}</span>${actorLabel ? ' · ' + actorLabel : ''}${lastLabel ? ' · ' + lastLabel : ''}`;
  }
  statusContainer.innerHTML = statusHtml;

  // Sprite art toggle — switches between the main game's pixel art and
  // the original placeholder art.
  const toggle = document.createElement('button');
  toggle.className = 'art-toggle';
  toggle.type = 'button';
  toggle.textContent = gameArt ? 'Art: game' : 'Art: classic';
  toggle.title = 'Switch between the main game pixel art and the original placeholder art';
  toggle.onclick = () => {
    setSpriteTheme(gameArt ? 'kaggle' : 'game');
    if (lastOptions) renderer(lastOptions);
  };
  statusContainer.appendChild(toggle);
}
