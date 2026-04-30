import { CENTER, ROTATION_RADIUS_LIMIT } from '../engine/constants';
import { pointToSegmentDistance, type Vec2 } from '../engine/geometry';
import type { GameState, Planet } from '../engine/types';

/**
 * Predict where a planet will be `ticksAhead` steps from now (0 = current step).
 * Returns null for comet planets (their precomputed paths aren't expanded here).
 */
export function predictPlanetPosition(
  state: GameState,
  planet: Planet,
  ticksAhead: number
): { x: number; y: number } | null {
  if (state.cometPlanetIds.includes(planet.id)) return null;

  const initial = state.initialPlanets.find((p) => p.id === planet.id);
  if (!initial) return { x: planet.x, y: planet.y };

  const dx = initial.x - CENTER;
  const dy = initial.y - CENTER;
  const r = Math.sqrt(dx * dx + dy * dy);

  if (r + planet.radius >= ROTATION_RADIUS_LIMIT) {
    return { x: planet.x, y: planet.y };
  }

  const initialAngle = Math.atan2(dy, dx);
  const futureAngle = initialAngle + state.angularVelocity * (state.step + ticksAhead);
  return {
    x: CENTER + r * Math.cos(futureAngle),
    y: CENTER + r * Math.sin(futureAngle),
  };
}

/** Fleet position at end of tick k after launch (k=1 = first move). */
export function predictFleetPosition(
  from: Planet,
  angle: number,
  stepSpeed: number,
  ticksAhead: number
): { x: number; y: number } {
  const t = from.radius + 0.1 + stepSpeed * ticksAhead;
  return {
    x: from.x + Math.cos(angle) * t,
    y: from.y + Math.sin(angle) * t,
  };
}

export interface CollisionScore {
  /** True if our predictor says the fleet will hit the planet at some tick. */
  hit: boolean;
  /**
   * 0..1 depth of the closest approach inside the planet:
   *   1 = passes through the center, 0 = just barely grazes the radius.
   * For misses this stays at 0.
   */
  confidence: number;
}

/**
 * Predict whether a fleet launched now (with `stepSpeed`/`angle` from `from`) will
 * collide with `target` within `maxTicks` ticks. Mirrors the engine's two-phase
 * continuous collision check (fleet segment vs planet point, then post-move fleet
 * point vs planet swept segment) so the result matches what the simulator will do.
 */
export function fleetCollisionScore(
  state: GameState,
  from: Planet,
  angle: number,
  stepSpeed: number,
  target: Planet,
  maxTicks: number
): CollisionScore {
  const cx = Math.cos(angle);
  const sy = Math.sin(angle);
  const sx0 = from.x + cx * (from.radius + 0.1);
  const sy0 = from.y + sy * (from.radius + 0.1);
  const radius = target.radius;

  let minRatio = Infinity;

  for (let k = 1; k <= maxTicks; k++) {
    const oldFleet: Vec2 = [sx0 + cx * (k - 1) * stepSpeed, sy0 + sy * (k - 1) * stepSpeed];
    const newFleet: Vec2 = [sx0 + cx * k * stepSpeed, sy0 + sy * k * stepSpeed];

    const before = predictPlanetPosition(state, target, k - 1);
    const after = predictPlanetPosition(state, target, k);
    if (!before || !after) break;

    // Phase 3: fleet sweeps past the (pre-rotation) planet.
    const d1 = pointToSegmentDistance([before.x, before.y], oldFleet, newFleet);
    if (d1 / radius < minRatio) minRatio = d1 / radius;

    // Phase 4: planet rotates onto/past the (post-move) fleet.
    const d2 = pointToSegmentDistance(newFleet, [before.x, before.y], [after.x, after.y]);
    if (d2 / radius < minRatio) minRatio = d2 / radius;
  }

  const hit = minRatio < 1;
  // confidence = how deep inside the radius the closest approach goes.
  // Clamp at 0.05 so even a near-grazing hit stays visibly green (not pure black).
  const confidence = hit ? Math.max(0.05, 1 - minRatio) : 0;
  return { hit, confidence };
}
