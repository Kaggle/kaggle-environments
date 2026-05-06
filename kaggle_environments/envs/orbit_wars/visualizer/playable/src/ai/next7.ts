/**
 * Port of next7.py — the strongest of the next* progression.
 *
 * Key ideas:
 *   - Defensive: detect inbound enemy fleets, divert reinforcements that
 *     can arrive in time.
 *   - Offensive: for each owned planet, pick best target by production/distance.
 *     Account for incoming friendly ships (don't double-target neutrals).
 *     Use iterative intercept refinement to handle moving targets.
 */
import type { Action, Fleet, Planet } from '../engine/types';
import type { AgentFn } from './types';

const CENTER = 50.0;
const ROTATION_RADIUS_LIMIT = 50.0;
const SUN_RADIUS = 10.0;
const MAX_SPEED = 6.0;

function fleetSpeed(ships: number): number {
  if (ships <= 1) return 1.0;
  return Math.min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * Math.pow(Math.log(ships) / Math.log(1000), 1.5));
}

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function pointToSegmentDistance(p: [number, number], v: [number, number], w: [number, number]): number {
  const l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2;
  if (l2 === 0) return Math.hypot(p[0] - v[0], p[1] - v[1]);
  const t = Math.max(0, Math.min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2));
  return Math.hypot(p[0] - (v[0] + t * (w[0] - v[0])), p[1] - (v[1] + t * (w[1] - v[1])));
}

function isOrbiting(p: Planet): boolean {
  const r = Math.hypot(p.x - CENTER, p.y - CENTER);
  return r + p.radius < ROTATION_RADIUS_LIMIT;
}

function getIntercept(
  src: Planet,
  target: Planet,
  omega: number,
  ships: number
): [number | null, number | null, number | null] {
  const speed = fleetSpeed(ships);
  if (!isOrbiting(target)) {
    return [dist(src, target) / speed, target.x, target.y];
  }
  const orbitalR = Math.hypot(target.x - CENTER, target.y - CENTER);
  const initAngle = Math.atan2(target.y - CENTER, target.x - CENTER);
  for (let t = 1; t < 100; t++) {
    const cur = initAngle + omega * t;
    const tx = CENTER + orbitalR * Math.cos(cur);
    const ty = CENTER + orbitalR * Math.sin(cur);
    const d = Math.hypot(tx - src.x, ty - src.y);
    if (d <= speed * t) return [t, tx, ty];
  }
  return [null, null, null];
}

function detectAttacks(
  fleets: Fleet[],
  myPlanets: Planet[],
  omega: number,
  player: number
): Map<number, Array<[Fleet, number]>> {
  const attacks = new Map<number, Array<[Fleet, number]>>();
  for (const f of fleets) {
    if (f.owner === player) continue;
    const speed = fleetSpeed(f.ships);
    for (const mp of myPlanets) {
      const orbitalR = Math.hypot(mp.x - CENTER, mp.y - CENTER);
      const isStatic = orbitalR + mp.radius >= ROTATION_RADIUS_LIMIT;
      const initAngle = Math.atan2(mp.y - CENTER, mp.x - CENTER);
      for (let t = 1; t < 100; t++) {
        let px: number, py: number;
        if (isStatic) {
          px = mp.x;
          py = mp.y;
        } else {
          const cur = initAngle + omega * t;
          px = CENTER + orbitalR * Math.cos(cur);
          py = CENTER + orbitalR * Math.sin(cur);
        }
        const fx = f.x + t * speed * Math.cos(f.angle);
        const fy = f.y + t * speed * Math.sin(f.angle);
        if (Math.hypot(px - fx, py - fy) < mp.radius) {
          if (!attacks.has(mp.id)) attacks.set(mp.id, []);
          attacks.get(mp.id)!.push([f, t]);
          break;
        }
      }
    }
  }
  return attacks;
}

function trackFriendlyFleets(
  fleets: Fleet[],
  targets: Planet[],
  omega: number,
  player: number
): Map<number, Array<[number, number]>> {
  const incoming = new Map<number, Array<[number, number]>>();
  for (const f of fleets) {
    if (f.owner !== player) continue;
    const speed = fleetSpeed(f.ships);
    for (const tp of targets) {
      const orbitalR = Math.hypot(tp.x - CENTER, tp.y - CENTER);
      const isStatic = orbitalR + tp.radius >= ROTATION_RADIUS_LIMIT;
      const initAngle = Math.atan2(tp.y - CENTER, tp.x - CENTER);
      for (let t = 1; t < 100; t++) {
        let px: number, py: number;
        if (isStatic) {
          px = tp.x;
          py = tp.y;
        } else {
          const cur = initAngle + omega * t;
          px = CENTER + orbitalR * Math.cos(cur);
          py = CENTER + orbitalR * Math.sin(cur);
        }
        const fx = f.x + t * speed * Math.cos(f.angle);
        const fy = f.y + t * speed * Math.sin(f.angle);
        if (Math.hypot(px - fx, py - fy) < tp.radius) {
          if (!incoming.has(tp.id)) incoming.set(tp.id, []);
          incoming.get(tp.id)!.push([f.ships, t]);
          break;
        }
      }
    }
  }
  return incoming;
}

export const next7Agent: AgentFn = (obs) => {
  const moves: Action[] = [];
  const player = obs.player;
  const omega = obs.angularVelocity;
  const myPlanets = obs.planets.filter((p) => p.owner === player);
  let targets = obs.planets.filter((p) => p.owner !== player);

  const availableShips = new Map<number, number>();
  for (const mp of myPlanets) availableShips.set(mp.id, mp.ships);

  // 1. Defense
  const attacks = detectAttacks(obs.fleets, myPlanets, omega, player);
  for (const [pid, attackList] of attacks) {
    const mp = myPlanets.find((p) => p.id === pid);
    if (!mp) continue;
    attackList.sort((a, b) => a[1] - b[1]);
    const [f, arrTime] = attackList[0];
    const shipsAtArrival = mp.ships + mp.production * arrTime;
    if (shipsAtArrival < f.ships + 5) {
      let neededHelp = f.ships + 5 - shipsAtArrival;

      const orbitalR = Math.hypot(mp.x - CENTER, mp.y - CENTER);
      const isStatic = orbitalR + mp.radius >= ROTATION_RADIUS_LIMIT;
      let targetX: number, targetY: number;
      if (isStatic) {
        targetX = mp.x;
        targetY = mp.y;
      } else {
        const initAngle = Math.atan2(mp.y - CENTER, mp.x - CENTER);
        const cur = initAngle + omega * arrTime;
        targetX = CENTER + orbitalR * Math.cos(cur);
        targetY = CENTER + orbitalR * Math.sin(cur);
      }

      for (const helper of myPlanets) {
        if (helper.id === mp.id) continue;
        const avail = availableShips.get(helper.id) ?? 0;
        if (avail <= 0) continue;
        const d = Math.hypot(targetX - helper.x, targetY - helper.y);
        const speed = fleetSpeed(avail);
        const helpArr = d / speed;
        if (helpArr < arrTime) {
          const ships = Math.min(Math.floor(neededHelp), avail);
          if (ships > 0) {
            const angle = Math.atan2(targetY - helper.y, targetX - helper.x);
            moves.push([helper.id, angle, ships]);
            availableShips.set(helper.id, avail - ships);
            neededHelp -= ships;
            if (neededHelp <= 0) break;
          }
        }
      }
    }
  }

  // 2. Attack
  const incomingHelp = trackFriendlyFleets(obs.fleets, targets, omega, player);

  for (const mp of myPlanets) {
    const avail = availableShips.get(mp.id) ?? 0;
    if (avail <= 0) continue;

    const validTargets: Array<[Planet, number, number, number, number]> = [];
    for (const t of targets) {
      const buf = t.owner === -1 ? 5 : 15;
      const helpList = incomingHelp.get(t.id) ?? [];

      const estSpeed = fleetSpeed(avail);
      const d = Math.hypot(mp.x - t.x, mp.y - t.y);
      const estTravel = d / estSpeed;

      let validHelp = 0;
      for (const [hShips, arr] of helpList) {
        if (arr <= estTravel) validHelp += hShips;
      }
      if (validHelp >= t.ships + buf) continue;

      let needed = Math.max(1, t.ships + buf - validHelp);
      let bestT: number | null = null;
      let bestX: number | null = null;
      let bestY: number | null = null;

      for (let it = 0; it < 5; it++) {
        const [travelTime, tx, ty] = getIntercept(mp, t, omega, needed);
        if (travelTime === null || tx === null || ty === null) break;
        if (pointToSegmentDistance([CENTER, CENTER], [mp.x, mp.y], [tx, ty]) < SUN_RADIUS + 1.0) break;

        let newNeeded = t.ships + buf - validHelp;
        if (t.owner !== -1) newNeeded += t.production * travelTime;

        if (Math.abs(newNeeded - needed) < 1) {
          needed = newNeeded;
          bestT = travelTime;
          bestX = tx;
          bestY = ty;
          break;
        }
        needed = newNeeded;
        bestT = travelTime;
        bestX = tx;
        bestY = ty;
      }

      if (bestT !== null && bestX !== null && bestY !== null && avail > needed) {
        const distToIntercept = Math.hypot(bestX - mp.x, bestY - mp.y);
        let score = t.production / Math.max(distToIntercept, 0.001);
        const orbR = Math.hypot(t.x - CENTER, t.y - CENTER);
        const isStatic = orbR + t.radius >= ROTATION_RADIUS_LIMIT;
        if (!isStatic) {
          if (t.production < 3) continue;
          score *= 0.7;
        }
        validTargets.push([t, score, needed, bestX, bestY]);
      }
    }

    validTargets.sort((a, b) => b[1] - a[1]);
    if (validTargets.length > 0) {
      const [bestTgt, , neededShips, tx, ty] = validTargets[0];
      const angle = Math.atan2(ty - mp.y, tx - mp.x);
      const send = Math.min(Math.floor(neededShips), avail);
      if (send > 0) {
        moves.push([mp.id, angle, send]);
        availableShips.set(mp.id, avail - send);
        targets = targets.filter((tt) => tt.id !== bestTgt.id);
      }
    }
  }

  return moves;
};
