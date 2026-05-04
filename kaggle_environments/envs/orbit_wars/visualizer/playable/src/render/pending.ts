import { BOARD_SIZE } from '../engine/constants';
import type { Action, GameState } from '../engine/types';
import { ownerColor } from './colors';

/** Draws faded "ghost fleet" chevrons just outside the source planet for each pending action. */
export function drawPendingFleets(
  ctx: CanvasRenderingContext2D,
  cssSize: number,
  state: GameState,
  pendingActions: Action[],
  ownerId: number
): void {
  if (pendingActions.length === 0) return;
  const scale = cssSize / BOARD_SIZE;
  const color = ownerColor(ownerId);

  for (const [planetId, angle, ships] of pendingActions) {
    const planet = state.planets.find((p) => p.id === planetId);
    if (!planet) continue;

    const offsetBoard = planet.radius + 1.2;
    const gx = (planet.x + Math.cos(angle) * offsetBoard) * scale;
    const gy = (planet.y + Math.sin(angle) * offsetBoard) * scale;
    const sz = (0.4 + (2.0 * Math.log(Math.max(1, ships))) / Math.log(1000)) * scale;

    // Chevron rotated to face launch direction.
    ctx.save();
    ctx.translate(gx, gy);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(sz, 0);
    ctx.lineTo(-sz, -sz * 0.6);
    ctx.lineTo(-sz * 0.3, 0);
    ctx.lineTo(-sz, sz * 0.6);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.35;
    ctx.fill();
    ctx.globalAlpha = 0.9;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 2]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Ship-count label beside the ghost (axis-aligned for readability).
    const labelFontSize = Math.max(7, scale * 1.1);
    ctx.font = `bold ${labelFontSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const lx = gx + Math.cos(angle) * sz * 1.5;
    const ly = gy + Math.sin(angle) * sz * 1.5;
    ctx.fillStyle = '#000';
    ctx.fillText(String(ships), lx + 0.5, ly + 0.5);
    ctx.fillStyle = color;
    ctx.fillText(String(ships), lx, ly);
  }
}
