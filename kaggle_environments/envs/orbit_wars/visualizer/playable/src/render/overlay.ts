import { BOARD_SIZE } from '../engine/constants';
import type { GameState, Planet } from '../engine/types';
import { ownerColor } from './colors';
import { fleetCollisionScore, predictPlanetPosition } from './predict';

/** Lerp dark green → bright green by confidence in [0,1]. */
function hitGreen(confidence: number): string {
  const t = Math.max(0, Math.min(1, confidence));
  const r = Math.round(26 + (34 - 26) * t);
  const g = Math.round(80 + (221 - 80) * t);
  const b = Math.round(46 + (102 - 46) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Largest t > 0 such that (x + t*cos(a), y + t*sin(a)) stays inside [0, BOARD_SIZE]^2. */
function distanceToBoardEdge(x: number, y: number, angle: number): number {
  const cx = Math.cos(angle);
  const sy = Math.sin(angle);
  const tx = cx > 1e-9 ? (BOARD_SIZE - x) / cx : cx < -1e-9 ? -x / cx : Infinity;
  const ty = sy > 1e-9 ? (BOARD_SIZE - y) / sy : sy < -1e-9 ? -y / sy : Infinity;
  return Math.max(0, Math.min(tx, ty));
}

/** Draw selection ring, aim line to board edge, per-tick dots, and ghost-future planet positions. */
export const TARGET_HIT_RADIUS = 2.2; // board units; how close a click must be to "grab" the target

/** Draw the placed aim-target reticle at board coords (target.x, target.y). */
function drawTarget(
  ctx: CanvasRenderingContext2D,
  scale: number,
  target: { x: number; y: number },
  color: string
): void {
  const tx = target.x * scale;
  const ty = target.y * scale;
  const r = Math.max(6, scale * 1.4);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(tx, ty, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tx - r * 1.4, ty);
  ctx.lineTo(tx - r * 0.4, ty);
  ctx.moveTo(tx + r * 0.4, ty);
  ctx.lineTo(tx + r * 1.4, ty);
  ctx.moveTo(tx, ty - r * 1.4);
  ctx.lineTo(tx, ty - r * 0.4);
  ctx.moveTo(tx, ty + r * 0.4);
  ctx.lineTo(tx, ty + r * 1.4);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(tx, ty, Math.max(1.5, scale * 0.25), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  cssSize: number,
  state: GameState,
  selected: Planet | null,
  aimAngle: number | null,
  shipsForSpeed: number,
  stepSpeed: number,
  mouseBoard: { x: number; y: number } | null,
  aimTarget: { x: number; y: number } | null
): void {
  if (!selected) return;
  const scale = cssSize / BOARD_SIZE;
  const px = selected.x * scale;
  const py = selected.y * scale;
  const pr = selected.radius * scale;
  const color = ownerColor(selected.owner);

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.arc(px, py, pr + 4, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  if (aimAngle !== null && stepSpeed > 0 && shipsForSpeed > 0) {
    const tEdge = distanceToBoardEdge(selected.x, selected.y, aimAngle);
    const ex = px + Math.cos(aimAngle) * tEdge * scale;
    const ey = py + Math.sin(aimAngle) * tEdge * scale;

    // Aim line to the edge.
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // How many ticks until the fleet leaves the board.
    const startOffset = selected.radius + 0.1;
    const numTicks = Math.max(0, Math.floor((tEdge - startOffset) / stepSpeed));

    // Tick dots: end-of-tick fleet positions (matches the engine's per-step move).
    ctx.fillStyle = color;
    for (let i = 1; i <= numTicks; i++) {
      const t = startOffset + i * stepSpeed;
      if (t > tEdge) break;
      const dx = px + Math.cos(aimAngle) * t * scale;
      const dy = py + Math.sin(aimAngle) * t * scale;
      ctx.globalAlpha = Math.max(0.25, 1 - i / Math.max(numTicks, 8));
      ctx.beginPath();
      ctx.arc(dx, dy, Math.max(2, scale * 0.35), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Ghost-future planet positions, one per tick. Hovering one tints it.
    if (numTicks > 0) {
      for (const planet of state.planets) {
        if (planet.id === selected.id) continue;
        if (state.cometPlanetIds.includes(planet.id)) continue;

        // Determine if the user's mouse is over any of this planet's ghosts.
        let hovered = false;
        if (mouseBoard) {
          for (let k = 1; k <= numTicks; k++) {
            const gp = predictPlanetPosition(state, planet, k);
            if (!gp) break;
            const ddx = mouseBoard.x - gp.x;
            const ddy = mouseBoard.y - gp.y;
            if (ddx * ddx + ddy * ddy <= planet.radius * planet.radius) {
              hovered = true;
              break;
            }
          }
        }

        let strokeColor: string;
        if (hovered) {
          const score = fleetCollisionScore(state, selected, aimAngle, stepSpeed, planet, numTicks);
          strokeColor = score.hit ? hitGreen(score.confidence) : '#ff4444';
        } else {
          strokeColor = ownerColor(planet.owner);
        }

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = hovered ? 1.5 : 1;
        for (let k = 1; k <= numTicks; k++) {
          const gp = predictPlanetPosition(state, planet, k);
          if (!gp) break;
          const gx = gp.x * scale;
          const gy = gp.y * scale;
          const gr = planet.radius * scale;
          ctx.globalAlpha = hovered ? 0.55 : Math.max(0.08, 0.32 - k * 0.015);
          ctx.beginPath();
          ctx.arc(gx, gy, gr, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
    }
  }

  if (aimTarget) {
    drawTarget(ctx, scale, aimTarget, color);
  }
  ctx.restore();
}
