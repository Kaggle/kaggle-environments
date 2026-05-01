/**
 * Port of agent4.py — strongest agent. agent3 base + smooth distance scoring +
 * close-enemy bonus + send-sizing that accounts for inbound allied fleets.
 */
import type { Action, Fleet, Planet } from '../engine/types';
import type { AgentFn } from './types';

const CENTER = 50.0;
const SUN_RADIUS = 10.0;
const ROTATION_RADIUS_LIMIT = 50.0;
const BOARD_SIZE = 100.0;
const MAX_SPEED = 6.0;

function fleetSpeed(ships: number): number {
  if (ships <= 1) return 1.0;
  return Math.min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * Math.pow(Math.log(ships) / Math.log(1000), 1.5));
}

function dist(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function pointToSegmentDistance(p: [number, number], v: [number, number], w: [number, number]): number {
  const l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2;
  if (l2 === 0) return dist(p, v);
  const t = Math.max(0, Math.min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2));
  return dist(p, [v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1])]);
}

function isOrbiting(p: Planet): boolean {
  return Math.hypot(p.x - CENTER, p.y - CENTER) + p.radius < ROTATION_RADIUS_LIMIT;
}

function rotateAroundCenter(pos: [number, number], dtheta: number): [number, number] {
  const dx = pos[0] - CENTER;
  const dy = pos[1] - CENTER;
  const c = Math.cos(dtheta);
  const s = Math.sin(dtheta);
  return [CENTER + dx * c - dy * s, CENTER + dx * s + dy * c];
}

function predictPosition(planet: Planet, av: number, t: number): [number, number] {
  if (isOrbiting(planet)) return rotateAroundCenter([planet.x, planet.y], av * t);
  return [planet.x, planet.y];
}

function computeIntercept(
  srcPos: [number, number],
  target: Planet,
  av: number,
  ships: number
): [[number, number], number] {
  const speed = fleetSpeed(ships);
  let pos: [number, number] = [target.x, target.y];
  if (!isOrbiting(target)) return [pos, Math.max(1, dist(srcPos, pos) / speed)];
  for (let i = 0; i < 20; i++) {
    const d = dist(srcPos, pos);
    const t = Math.max(1, d / speed);
    const newPos = predictPosition(target, av, t);
    if (dist(newPos, pos) < 0.1) {
      pos = newPos;
      break;
    }
    pos = newPos;
  }
  return [pos, Math.max(1, dist(srcPos, pos) / speed)];
}

function safeAngle(srcPos: [number, number], targetPos: [number, number]): number | null {
  if (pointToSegmentDistance([CENTER, CENTER], srcPos, targetPos) >= SUN_RADIUS + 1.5) {
    return Math.atan2(targetPos[1] - srcPos[1], targetPos[0] - srcPos[0]);
  }
  return null;
}

function estimateIncoming(fleets: Fleet[], planets: Planet[], av: number, player: number): Map<number, number> {
  const threat = new Map<number, number>();
  for (const p of planets) if (p.owner === player) threat.set(p.id, 0);
  for (const f of fleets) {
    const speed = fleetSpeed(f.ships);
    let prev: [number, number] = [f.x, f.y];
    for (let t = 1; t < 50; t++) {
      const nx = f.x + Math.cos(f.angle) * speed * t;
      const ny = f.y + Math.sin(f.angle) * speed * t;
      if (nx < 0 || nx > BOARD_SIZE || ny < 0 || ny > BOARD_SIZE) break;
      if (pointToSegmentDistance([CENTER, CENTER], prev, [nx, ny]) < SUN_RADIUS) break;
      let hit: Planet | null = null;
      for (const p of planets) {
        const pos: [number, number] = isOrbiting(p) ? predictPosition(p, av, t) : [p.x, p.y];
        if (pointToSegmentDistance(pos, prev, [nx, ny]) < p.radius + 0.5) {
          hit = p;
          break;
        }
      }
      if (hit !== null) {
        if (hit.owner === player && f.owner !== player) {
          threat.set(hit.id, (threat.get(hit.id) ?? 0) + f.ships);
        } else if (hit.owner === player && f.owner === player) {
          threat.set(hit.id, (threat.get(hit.id) ?? 0) - f.ships);
        }
        break;
      }
      prev = [nx, ny];
    }
  }
  return threat;
}

export const agent4Agent: AgentFn = (obs) => {
  const planets = obs.planets;
  const fleets = obs.fleets;
  const player = obs.player;
  const av = obs.angularVelocity;
  const step = obs.step;
  const remaining = Math.max(1, 500 - step);

  const myPlanets = planets.filter((p) => p.owner === player);
  if (myPlanets.length === 0) return [];

  const threat = estimateIncoming(fleets, planets, av, player);

  // Allied inbound prediction
  const alliedInbound = new Map<number, number>();
  for (const p of planets) alliedInbound.set(p.id, 0);
  for (const f of fleets) {
    if (f.owner !== player) continue;
    const speed = fleetSpeed(f.ships);
    let prev: [number, number] = [f.x, f.y];
    for (let t = 1; t < 90; t++) {
      const nx = f.x + Math.cos(f.angle) * speed * t;
      const ny = f.y + Math.sin(f.angle) * speed * t;
      if (nx < 0 || nx > BOARD_SIZE || ny < 0 || ny > BOARD_SIZE) break;
      if (pointToSegmentDistance([CENTER, CENTER], prev, [nx, ny]) < SUN_RADIUS) break;
      let hit: number | null = null;
      for (const p of planets) {
        const pos: [number, number] = isOrbiting(p) ? predictPosition(p, av, t) : [p.x, p.y];
        if (pointToSegmentDistance(pos, prev, [nx, ny]) < p.radius + 0.5) {
          hit = p.id;
          break;
        }
      }
      if (hit !== null) {
        alliedInbound.set(hit, (alliedInbound.get(hit) ?? 0) + f.ships);
        break;
      }
      prev = [nx, ny];
    }
  }

  const moves: Action[] = [];
  const committed = new Map<number, number>();

  const sortedSrc = [...myPlanets].sort((a, b) => b.ships - a.ships);
  for (const src of sortedSrc) {
    if (src.ships < 20) continue;
    const netThreat = Math.max(0, threat.get(src.id) ?? 0);

    let best: { tgt: Planet; ipos: [number, number]; angle: number; turns: number; cost: number } | null = null;
    let bestScore = -1;

    for (const tgt of planets) {
      if (tgt.id === src.id || tgt.owner === player) continue;
      const ships = Math.max(20, Math.floor(src.ships / 2));
      const [ipos, tf] = computeIntercept([src.x, src.y], tgt, av, ships);
      const turns = Math.max(1, Math.ceil(tf));
      if (turns > 80) continue;
      const angle = safeAngle([src.x, src.y], ipos);
      if (angle === null) continue;
      const d = dist([src.x, src.y], ipos);
      const cost = tgt.owner === -1 ? tgt.ships + 1 : tgt.ships + tgt.production * turns + 1;
      const already = (alliedInbound.get(tgt.id) ?? 0) + (committed.get(tgt.id) ?? 0);
      if (tgt.owner === -1 && already >= cost) continue;
      const payoff = tgt.production * Math.max(0, remaining - turns);
      let score = payoff / (cost + 5) / (1 + 0.02 * turns);
      if (tgt.owner !== -1) score *= 3.0;
      score *= 30.0 / (30.0 + d);
      if (tgt.owner !== -1 && tgt.owner !== player && already > 0 && already < cost) score *= 1.5;
      if (tgt.owner !== -1 && tgt.owner !== player) {
        if (d < 25) score *= 1.7;
        else if (d < 40) score *= 1.3;
      }
      if (score > bestScore) {
        bestScore = score;
        best = { tgt, ipos, angle, turns, cost };
      }
    }

    if (!best) continue;
    const { tgt, angle, cost } = best;
    const alreadyIn = (alliedInbound.get(tgt.id) ?? 0) + (committed.get(tgt.id) ?? 0);
    let need = Math.max(20, cost - alreadyIn + 5);
    let send = Math.max(need, Math.floor(src.ships / 2));
    const threatBuf = Math.floor(netThreat * 1.1) + (netThreat > 0 ? 5 : 5);
    const usable = Math.max(0, src.ships - threatBuf);
    send = Math.min(send, usable);
    if (send < 20) continue;
    moves.push([src.id, angle, Math.floor(send)]);
    committed.set(tgt.id, (committed.get(tgt.id) ?? 0) + send);
  }
  return moves;
};
