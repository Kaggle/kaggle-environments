import { BOARD_SIZE, CENTER, SUN_RADIUS } from '../engine/constants';
import type { GameState } from '../engine/types';
import { ownerColor } from './colors';

export interface RenderSettings {
  showFleetNumbers: boolean;
  showProductionDots: boolean;
  /** If set, subtract these ships from the displayed planet count (queued human launches). */
  pendingShipsByPlanet?: Map<number, number>;
}

export const DEFAULT_RENDER_SETTINGS: RenderSettings = {
  showFleetNumbers: true,
  showProductionDots: true,
};

/** Draws a single frame of GameState into ctx. ctx is sized cssSize x cssSize, DPR already applied. */
export function renderGame(
  ctx: CanvasRenderingContext2D,
  state: GameState,
  cssSize: number,
  settings: RenderSettings = DEFAULT_RENDER_SETTINGS
): void {
  const w = cssSize;
  const scale = w / BOARD_SIZE;

  // Clear
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, w, w);

  // Sun glow + body
  const sunX = CENTER * scale;
  const sunY = CENTER * scale;
  const sunR = SUN_RADIUS * scale;
  const glow = ctx.createRadialGradient(sunX, sunY, sunR * 0.5, sunX, sunY, sunR * 2.5);
  glow.addColorStop(0, 'rgba(255, 200, 50, 0.6)');
  glow.addColorStop(0.5, 'rgba(255, 150, 20, 0.2)');
  glow.addColorStop(1, 'rgba(255, 100, 0, 0)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, w, w);
  ctx.beginPath();
  ctx.arc(sunX, sunY, sunR, 0, Math.PI * 2);
  ctx.fillStyle = '#FFB800';
  ctx.fill();
  ctx.strokeStyle = '#FFD700';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Comet trails
  for (const group of state.comets) {
    const idx = group.pathIndex;
    for (let i = 0; i < group.planetIds.length; i++) {
      const path = group.paths[i];
      const tailLen = Math.min(idx + 1, path.length, 5);
      if (tailLen < 2) continue;
      for (let t = 1; t < tailLen; t++) {
        const pi = idx - t;
        if (pi < 0) break;
        const alpha = 0.4 * (1 - t / tailLen);
        ctx.beginPath();
        ctx.moveTo(path[pi + 1][0] * scale, path[pi + 1][1] * scale);
        ctx.lineTo(path[pi][0] * scale, path[pi][1] * scale);
        ctx.strokeStyle = `rgba(200, 220, 255, ${alpha})`;
        ctx.lineWidth = ((2.5 - (1.5 * t) / tailLen) * scale) / 5;
        ctx.lineCap = 'round';
        ctx.stroke();
      }
    }
  }

  const cometSet = new Set(state.cometPlanetIds);

  // Planets
  for (const planet of state.planets) {
    const px = planet.x * scale;
    const py = planet.y * scale;
    const pr = planet.radius * scale;
    const color = ownerColor(planet.owner);
    const isComet = cometSet.has(planet.id);

    ctx.beginPath();
    ctx.arc(px, py, pr, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = planet.owner >= 0 ? 0.85 : 0.5;
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.beginPath();
    ctx.arc(px, py, pr, 0, Math.PI * 2);
    ctx.strokeStyle = isComet ? '#88ccff' : '#555';
    ctx.lineWidth = isComet ? 2 : 1;
    ctx.stroke();

    if (settings.showProductionDots && planet.owner >= 0 && planet.production > 0) {
      const dotR = Math.max(1, scale * 0.3);
      for (let d = 0; d < planet.production; d++) {
        const dotAngle = (d / planet.production) * Math.PI * 2 - Math.PI / 2;
        const dotDist = pr + dotR + 2;
        const dx = px + Math.cos(dotAngle) * dotDist;
        const dy = py + Math.sin(dotAngle) * dotDist;
        ctx.beginPath();
        ctx.arc(dx, dy, dotR, 0, Math.PI * 2);
        ctx.fillStyle = '#aaa';
        ctx.fill();
      }
    }
  }

  // Fleets as chevrons
  for (const fleet of state.fleets) {
    const fx = fleet.x * scale;
    const fy = fleet.y * scale;
    const color = ownerColor(fleet.owner);
    const sz = (0.4 + (2.0 * Math.log(Math.max(1, fleet.ships))) / Math.log(1000)) * scale;

    ctx.save();
    ctx.translate(fx, fy);
    ctx.rotate(fleet.angle);

    ctx.beginPath();
    ctx.moveTo(sz, 0);
    ctx.lineTo(-sz, -sz * 0.6);
    ctx.lineTo(-sz * 0.3, 0);
    ctx.lineTo(-sz, sz * 0.6);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Per-player marking lines (colorblind accessibility)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = sz * 0.15;
    ctx.lineCap = 'round';
    if (fleet.owner === 1 || fleet.owner === 3) {
      ctx.beginPath();
      ctx.moveTo(sz * 0.8, 0);
      ctx.lineTo(-sz * 0.2, 0);
      ctx.stroke();
    }
    if (fleet.owner === 2 || fleet.owner === 3) {
      ctx.beginPath();
      ctx.moveTo(sz * 0.6, -sz * 0.15);
      ctx.lineTo(-sz * 0.7, -sz * 0.45);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(sz * 0.6, sz * 0.15);
      ctx.lineTo(-sz * 0.7, sz * 0.45);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Ship count labels
  const planetFontSize = Math.max(8, scale * 1.8);
  ctx.font = `bold ${planetFontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const planet of state.planets) {
    const px = planet.x * scale;
    const py = planet.y * scale;
    const queued = settings.pendingShipsByPlanet?.get(planet.id) ?? 0;
    const remaining = Math.max(0, Math.floor(planet.ships) - queued);
    const txt = queued > 0 ? `${remaining} (-${queued})` : `${Math.floor(planet.ships)}`;
    ctx.fillStyle = '#000000';
    ctx.fillText(txt, px + 0.5, py + 0.5);
    ctx.fillStyle = queued > 0 ? '#ffd166' : '#ffffff';
    ctx.fillText(txt, px, py);
  }

  if (settings.showFleetNumbers) {
    const fleetFontSize = Math.max(6, scale * 1.2);
    ctx.font = `${fleetFontSize}px sans-serif`;
    for (const fleet of state.fleets) {
      const fx = fleet.x * scale;
      const fy = fleet.y * scale;
      const labelOffset = fleet.y >= 50 ? -scale * 2.5 : scale * 2.5;
      ctx.fillStyle = ownerColor(fleet.owner);
      ctx.fillText(Math.floor(fleet.ships).toString(), fx, fy + labelOffset);
    }
  }

  // Step indicator
  const stepFontSize = Math.max(8, scale * 1.5);
  ctx.font = `${stepFontSize}px sans-serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#888';
  ctx.fillText(`Step ${state.step}`, 6, 6);
}
