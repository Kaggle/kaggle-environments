import {
  BOARD_SIZE,
  CENTER,
  COMET_RADIUS,
  COMET_PRODUCTION,
  MAX_PLANET_GROUPS,
  MIN_PLANET_GROUPS,
  MIN_STATIC_GROUPS,
  PLANET_CLEARANCE,
  ROTATION_RADIUS_LIMIT,
  SUN_RADIUS,
} from './constants';
import { distance, type Vec2 } from './geometry';
import type { CometGroup, Planet } from './types';
import type { PyRandom } from './rng';

/** Direct port of generate_planets() in orbit_wars.py. */
export function generatePlanets(rng: PyRandom): Planet[] {
  const planets: Planet[] = [];
  const numQ1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS);
  let id = 0;

  // Phase 1: guaranteed static planet groups.
  let staticGroups = 0;
  for (let attempt = 0; attempt < 5000; attempt++) {
    if (staticGroups >= MIN_STATIC_GROUPS) break;

    const prod = rng.randint(1, 5);
    const r = 1 + Math.log(prod);
    const angle = rng.uniform(0, Math.PI / 2);
    const minOrbital = ROTATION_RADIUS_LIMIT - r;
    const maxOrbital = (BOARD_SIZE - CENTER - r) / Math.max(Math.cos(angle), Math.sin(angle));
    if (minOrbital > maxOrbital) continue;
    const orbitalR = rng.uniform(minOrbital, maxOrbital);
    const x = CENTER + orbitalR * Math.cos(angle);
    const y = CENTER + orbitalR * Math.sin(angle);

    if (x + r > BOARD_SIZE || x - r < 0 || y + r > BOARD_SIZE || y - r < 0) continue;
    if (BOARD_SIZE - x - r < 0 || BOARD_SIZE - y - r < 0) continue;
    if (x - CENTER < r + 5 || y - CENTER < r + 5) continue;

    const ships = Math.min(rng.randint(5, 99), rng.randint(5, 99));
    // NOTE: matches Python: first symmetric copy uses (y, x) — the swap is
    // intentional and preserved because ports must reproduce the same layout.
    const temp: Planet[] = [
      { id: id, owner: -1, x: y, y: x, radius: r, ships, production: prod },
      { id: id + 1, owner: -1, x: BOARD_SIZE - x, y: y, radius: r, ships, production: prod },
      { id: id + 2, owner: -1, x: x, y: BOARD_SIZE - y, radius: r, ships, production: prod },
      { id: id + 3, owner: -1, x: BOARD_SIZE - y, y: BOARD_SIZE - x, radius: r, ships, production: prod },
    ];

    let valid = true;
    for (const tp of temp) {
      for (const p of planets) {
        if (distance([p.x, p.y], [tp.x, tp.y]) < p.radius + tp.radius + PLANET_CLEARANCE) {
          valid = false;
          break;
        }
      }
      if (!valid) break;
    }

    if (valid) {
      planets.push(...temp);
      id += 4;
      staticGroups++;
    }
  }

  // Phase 2: fill with random groups, ensure at least one orbiting.
  let attempts = 0;
  const maxAttempts = 5000;
  let hasOrbiting = false;

  while (planets.length < numQ1 * 4 || (!hasOrbiting && attempts < maxAttempts)) {
    attempts++;
    if (attempts >= maxAttempts) break;
    const prod = rng.randint(1, 5);
    const r = 1 + Math.log(prod);
    const x = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5);
    const y = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5);

    const orbitalRadius = distance([x, y], [CENTER, CENTER]);
    if (orbitalRadius < SUN_RADIUS + r + 10) continue;
    if (orbitalRadius + r >= ROTATION_RADIUS_LIMIT) {
      if (x + r > BOARD_SIZE || x - r < 0 || y + r > BOARD_SIZE || y - r < 0) continue;
    }

    const ships = rng.randint(5, 30);
    const temp: Planet[] = [
      { id: id, owner: -1, x: y, y: x, radius: r, ships, production: prod },
      { id: id + 1, owner: -1, x: BOARD_SIZE - x, y: y, radius: r, ships, production: prod },
      { id: id + 2, owner: -1, x: x, y: BOARD_SIZE - y, radius: r, ships, production: prod },
      { id: id + 3, owner: -1, x: BOARD_SIZE - y, y: BOARD_SIZE - x, radius: r, ships, production: prod },
    ];

    let valid = true;
    for (const tp of temp) {
      const tpOrbital = distance([tp.x, tp.y], [CENTER, CENTER]);
      const tpIsRotating = tpOrbital + tp.radius < ROTATION_RADIUS_LIMIT;

      for (const p of planets) {
        const pOrbital = distance([p.x, p.y], [CENTER, CENTER]);
        const pIsRotating = pOrbital + p.radius < ROTATION_RADIUS_LIMIT;

        if (distance([p.x, p.y], [tp.x, tp.y]) < p.radius + tp.radius + PLANET_CLEARANCE) {
          valid = false;
          break;
        }
        if (tpIsRotating !== pIsRotating) {
          if (Math.abs(tpOrbital - pOrbital) < tp.radius + p.radius + PLANET_CLEARANCE) {
            valid = false;
            break;
          }
        }
      }
      if (!valid) break;
    }

    if (valid) {
      if (orbitalRadius + r < ROTATION_RADIUS_LIMIT) hasOrbiting = true;
      planets.push(...temp);
      id += 4;
    }
  }

  return planets;
}

/**
 * Direct port of generate_comet_paths(). Returns null if no valid orbit found
 * after 300 attempts. The 4-element array is in the same order as the planet
 * symmetric copies above.
 */
export function generateCometPaths(
  initialPlanets: Planet[],
  angularVelocity: number,
  spawnStep: number,
  cometPlanetIds: Set<number>,
  cometSpeed: number,
  rng: PyRandom
): [number, number][][] | null {
  for (let attempt = 0; attempt < 300; attempt++) {
    const e = rng.uniform(0.75, 0.93);
    const a = rng.uniform(60, 150);
    const perihelion = a * (1 - e);
    if (perihelion < SUN_RADIUS + COMET_RADIUS) continue;

    const b = a * Math.sqrt(1 - e * e);
    const cVal = a * e;
    const phi = rng.uniform(Math.PI / 6, Math.PI / 3);

    // Dense sample around perihelion half of orbit.
    const dense: Vec2[] = [];
    const num = 5000;
    for (let i = 0; i < num; i++) {
      const t = 0.3 * Math.PI + (1.4 * Math.PI * i) / (num - 1);
      const ex = cVal + a * Math.cos(t);
      const ey = b * Math.sin(t);
      const x = CENTER + ex * Math.cos(phi) - ey * Math.sin(phi);
      const y = CENTER + ex * Math.sin(phi) + ey * Math.cos(phi);
      dense.push([x, y]);
    }

    // Resample at constant cometSpeed arc-length intervals.
    const path: Vec2[] = [dense[0]];
    let cum = 0;
    let target = cometSpeed;
    for (let i = 1; i < dense.length; i++) {
      cum += distance(dense[i], dense[i - 1]);
      if (cum >= target) {
        path.push(dense[i]);
        target += cometSpeed;
      }
    }

    // Extract contiguous on-board segment.
    let boardStart: number | null = null;
    let boardEnd: number | null = null;
    for (let i = 0; i < path.length; i++) {
      const [x, y] = path[i];
      if (x >= 0 && x <= BOARD_SIZE && y >= 0 && y <= BOARD_SIZE) {
        if (boardStart === null) boardStart = i;
        boardEnd = i;
      }
    }
    if (boardStart === null || boardEnd === null) continue;
    const visible = path.slice(boardStart, boardEnd + 1);
    if (visible.length < 5 || visible.length > 40) continue;

    // 4 rotationally symmetric paths.
    const paths: [number, number][][] = [
      visible.map(([x, y]) => [y, x] as [number, number]),
      visible.map(([x, y]) => [BOARD_SIZE - x, y] as [number, number]),
      visible.map(([x, y]) => [x, BOARD_SIZE - y] as [number, number]),
      visible.map(([x, y]) => [BOARD_SIZE - y, BOARD_SIZE - x] as [number, number]),
    ];

    // Separate planets into static and orbiting (skip other comets).
    const staticPlanets: Planet[] = [];
    const orbitingPlanets: Planet[] = [];
    for (const planet of initialPlanets) {
      if (cometPlanetIds.has(planet.id)) continue;
      const pr = distance([planet.x, planet.y], [CENTER, CENTER]);
      if (pr + planet.radius < ROTATION_RADIUS_LIMIT) orbitingPlanets.push(planet);
      else staticPlanets.push(planet);
    }

    let valid = true;
    const buf = COMET_RADIUS + 0.5;
    for (let k = 0; k < visible.length; k++) {
      const [cx, cy] = visible[k];
      if (distance([cx, cy], [CENTER, CENTER]) < SUN_RADIUS + COMET_RADIUS) {
        valid = false;
        break;
      }
      const symPts: Vec2[] = [
        [cy, cx],
        [BOARD_SIZE - cx, cy],
        [cx, BOARD_SIZE - cy],
        [BOARD_SIZE - cy, BOARD_SIZE - cx],
      ];
      for (const planet of staticPlanets) {
        let bad = false;
        for (const sp of symPts) {
          if (distance(sp, [planet.x, planet.y]) < planet.radius + buf) {
            bad = true;
            break;
          }
        }
        if (bad) {
          valid = false;
          break;
        }
      }
      if (!valid) break;

      const gameStep = spawnStep - 1 + k;
      for (const planet of orbitingPlanets) {
        const dx = planet.x - CENTER;
        const dy = planet.y - CENTER;
        const orbR = Math.sqrt(dx * dx + dy * dy);
        const initAngle = Math.atan2(dy, dx);
        const curAngle = initAngle + angularVelocity * gameStep;
        const px = CENTER + orbR * Math.cos(curAngle);
        const py = CENTER + orbR * Math.sin(curAngle);
        let bad = false;
        for (const sp of symPts) {
          if (distance(sp, [px, py]) < planet.radius + COMET_RADIUS) {
            bad = true;
            break;
          }
        }
        if (bad) {
          valid = false;
          break;
        }
      }
      if (!valid) break;
    }

    if (valid) return paths;
  }
  return null;
}

/** Make a CometGroup once paths have been generated; planets start off-board. */
export function buildCometGroup(
  paths: [number, number][][],
  startId: number,
  ships: number
): { group: CometGroup; planets: Planet[]; ids: number[] } {
  const ids = paths.map((_, i) => startId + i);
  const planets: Planet[] = ids.map((id) => ({
    id,
    owner: -1,
    x: -99,
    y: -99,
    radius: COMET_RADIUS,
    ships,
    production: COMET_PRODUCTION,
  }));
  return {
    group: { planetIds: ids, paths, pathIndex: -1 },
    planets,
    ids,
  };
}
