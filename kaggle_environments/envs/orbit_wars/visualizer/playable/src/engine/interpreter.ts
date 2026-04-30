import { BOARD_SIZE, CENTER, COMET_SPAWN_STEPS, ROTATION_RADIUS_LIMIT, SUN_RADIUS } from './constants';
import { distance, pointToSegmentDistance, type Vec2 } from './geometry';
import { buildCometGroup, generateCometPaths, generatePlanets } from './generate';
import { hashStringToSeed, PyRandom } from './rng';
import type { Action, ActionsByPlayer, Config, Fleet, GameState, Planet } from './types';

/** Speed scales log with fleet size; 1 ship = 1/turn, large fleets cap at maxSpeed. */
export function fleetSpeed(ships: number, maxSpeed: number): number {
  if (ships <= 1) return 1.0;
  const s = 1.0 + (maxSpeed - 1.0) * Math.pow(Math.log(ships) / Math.log(1000), 1.5);
  return Math.min(s, maxSpeed);
}

function clonePlanet(p: Planet): Planet {
  return { ...p };
}

/** Build the initial GameState — runs once per match. */
export function initGameState(numAgents: 2 | 4, config: Config): GameState {
  const initRng = new PyRandom(config.seed);
  const angularVelocity = initRng.uniform(0.025, 0.05);
  const planets = generatePlanets(initRng);
  const initialPlanets = planets.map(clonePlanet);

  // Assign home planets — pick a random symmetric group of 4.
  const numGroups = Math.floor(planets.length / 4);
  if (numGroups > 0) {
    const homeGroup = initRng.randint(0, numGroups - 1);
    const base = homeGroup * 4;
    if (numAgents === 2) {
      planets[base].owner = 0;
      planets[base].ships = 10;
      planets[base + 3].owner = 1;
      planets[base + 3].ships = 10;
    } else {
      for (let j = 0; j < 4; j++) {
        planets[base + j].owner = j;
        planets[base + j].ships = 10;
      }
    }
  }

  return {
    step: 0,
    numAgents,
    angularVelocity,
    planets,
    initialPlanets,
    fleets: [],
    nextFleetId: 0,
    comets: [],
    cometPlanetIds: [],
    done: false,
    scores: new Array(numAgents).fill(0),
    winners: [],
    seed: config.seed,
  };
}

/**
 * Advance one game step. Returns a NEW GameState — caller's state is not
 * mutated. Mirrors the Python `interpreter` body phase-for-phase:
 *   0. expire comets, spawn new comet group if scheduled
 *   1. process moves (launch fleets)
 *   2. production
 *   3. fleet movement w/ continuous collision (planets → bounds → sun)
 *   4. planet rotation + sweep
 *   5. comet movement + sweep
 *   6. combat resolution
 *   7. terminal check + scores
 */
export function step(prev: GameState, actions: ActionsByPlayer, config: Config): GameState {
  if (prev.done) return prev;

  // Deep-clone mutable bits so callers can keep `prev` around.
  const planets: Planet[] = prev.planets.map(clonePlanet);
  const fleets: Fleet[] = prev.fleets.map((f) => ({ ...f }));
  const comets = prev.comets.map((g) => ({
    planetIds: [...g.planetIds],
    paths: g.paths,
    pathIndex: g.pathIndex,
  }));
  const initialPlanets: Planet[] = prev.initialPlanets.map(clonePlanet);
  let cometPlanetIds = [...prev.cometPlanetIds];
  let nextFleetId = prev.nextFleetId;

  const { numAgents, angularVelocity, seed } = prev;
  const stepNum = prev.step + 1;

  // --- 0a. Expire comets whose paths have run out ---
  const expiredPids: number[] = [];
  for (const group of comets) {
    const idx = group.pathIndex;
    for (let i = 0; i < group.planetIds.length; i++) {
      const pid = group.planetIds[i];
      if (idx >= group.paths[i].length) expiredPids.push(pid);
    }
  }
  if (expiredPids.length > 0) {
    const expired = new Set(expiredPids);
    removeByIds(planets, expired);
    removeByIds(initialPlanets, expired);
    cometPlanetIds = cometPlanetIds.filter((pid) => !expired.has(pid));
    for (const g of comets) g.planetIds = g.planetIds.filter((pid) => !expired.has(pid));
    // Drop empty groups
    for (let i = comets.length - 1; i >= 0; i--) {
      if (comets[i].planetIds.length === 0) comets.splice(i, 1);
    }
  }

  // --- 0b. Spawn extra-solar comets at designated steps ---
  if (COMET_SPAWN_STEPS.includes(stepNum)) {
    const cometSeed = hashStringToSeed(`orbit_wars-comet-${seed}-${stepNum}`);
    const cometRng = new PyRandom(cometSeed);
    const paths = generateCometPaths(
      initialPlanets,
      angularVelocity,
      stepNum,
      new Set(cometPlanetIds),
      config.cometSpeed,
      cometRng
    );
    if (paths) {
      const nextId = Math.max(...planets.map((p) => p.id), -1) + 1;
      // Ship count: min of 4 randints, like Python.
      const ships = Math.min(
        cometRng.randint(1, 99),
        cometRng.randint(1, 99),
        cometRng.randint(1, 99),
        cometRng.randint(1, 99)
      );
      const built = buildCometGroup(paths, nextId, ships);
      comets.push(built.group);
      planets.push(...built.planets);
      initialPlanets.push(...built.planets.map(clonePlanet));
      cometPlanetIds.push(...built.ids);
    }
  }

  // --- 1. Process moves (launch fleets) ---
  const planetsById = new Map(planets.map((p) => [p.id, p]));
  for (let pid = 0; pid < numAgents; pid++) {
    const playerActions = actions[pid] ?? [];
    for (const move of playerActions) {
      if (!Array.isArray(move) || move.length !== 3) continue;
      const [fromId, angle, rawShips] = move;
      const ships = Math.floor(Number(rawShips));
      if (!Number.isFinite(ships) || ships <= 0) continue;
      const from = planetsById.get(fromId);
      if (!from || from.owner !== pid) continue;
      if (from.ships < ships) continue;
      from.ships -= ships;
      const startX = from.x + Math.cos(angle) * (from.radius + 0.1);
      const startY = from.y + Math.sin(angle) * (from.radius + 0.1);
      fleets.push({
        id: nextFleetId++,
        owner: pid,
        x: startX,
        y: startY,
        angle,
        fromPlanetId: fromId,
        ships,
      });
    }
  }

  // --- 2. Production ---
  for (const p of planets) {
    if (p.owner !== -1) p.ships += p.production;
  }

  // --- 3. Fleet movement w/ continuous collision detection ---
  const fleetsToRemove = new Set<number>();
  const combatLists = new Map<number, Fleet[]>();
  for (const p of planets) combatLists.set(p.id, []);

  for (const fleet of fleets) {
    const speed = fleetSpeed(fleet.ships, config.shipSpeed);
    const oldPos: Vec2 = [fleet.x, fleet.y];
    fleet.x += Math.cos(fleet.angle) * speed;
    fleet.y += Math.sin(fleet.angle) * speed;
    const newPos: Vec2 = [fleet.x, fleet.y];

    // Planet collision first (so fast fleets that overshoot still hit).
    let hit = false;
    for (const planet of planets) {
      if (pointToSegmentDistance([planet.x, planet.y], oldPos, newPos) < planet.radius) {
        combatLists.get(planet.id)!.push(fleet);
        fleetsToRemove.add(fleet.id);
        hit = true;
        break;
      }
    }
    if (hit) continue;

    if (fleet.x < 0 || fleet.x > BOARD_SIZE || fleet.y < 0 || fleet.y > BOARD_SIZE) {
      fleetsToRemove.add(fleet.id);
      continue;
    }
    if (pointToSegmentDistance([CENTER, CENTER], oldPos, newPos) < SUN_RADIUS) {
      fleetsToRemove.add(fleet.id);
      continue;
    }
  }

  // --- 4. Planet rotation + sweep ---
  const cometPidSet = new Set(cometPlanetIds);
  const initialById = new Map(initialPlanets.map((p) => [p.id, p]));

  const sweepFleets = (planet: Planet, oldPos: Vec2, newPos: Vec2) => {
    if (oldPos[0] === newPos[0] && oldPos[1] === newPos[1]) return;
    for (const fleet of fleets) {
      if (fleetsToRemove.has(fleet.id)) continue;
      if (pointToSegmentDistance([fleet.x, fleet.y], oldPos, newPos) < planet.radius) {
        combatLists.get(planet.id)!.push(fleet);
        fleetsToRemove.add(fleet.id);
      }
    }
  };

  for (const planet of planets) {
    if (cometPidSet.has(planet.id)) continue;
    const initialP = initialById.get(planet.id);
    if (!initialP) continue;
    const dx = initialP.x - CENTER;
    const dy = initialP.y - CENTER;
    const r = Math.sqrt(dx * dx + dy * dy);
    const oldPos: Vec2 = [planet.x, planet.y];

    if (r + planet.radius < ROTATION_RADIUS_LIMIT) {
      const initialAngle = Math.atan2(dy, dx);
      const currentAngle = initialAngle + angularVelocity * stepNum;
      planet.x = CENTER + r * Math.cos(currentAngle);
      planet.y = CENTER + r * Math.sin(currentAngle);
    }
    sweepFleets(planet, oldPos, [planet.x, planet.y]);
  }

  // --- 5. Comet movement + sweep + expire ---
  const expiredComet: number[] = [];
  for (const group of comets) {
    group.pathIndex += 1;
    const idx = group.pathIndex;
    for (let i = 0; i < group.planetIds.length; i++) {
      const pid = group.planetIds[i];
      const planet = planets.find((p) => p.id === pid);
      if (!planet) continue;
      const pPath = group.paths[i];
      if (idx >= pPath.length) {
        expiredComet.push(pid);
      } else {
        const oldPos: Vec2 = [planet.x, planet.y];
        planet.x = pPath[idx][0];
        planet.y = pPath[idx][1];
        // Skip sweep on first placement (off-board placeholder).
        if (oldPos[0] >= 0) sweepFleets(planet, oldPos, [planet.x, planet.y]);
      }
    }
  }
  if (expiredComet.length > 0) {
    const expired = new Set(expiredComet);
    removeByIds(planets, expired);
    removeByIds(initialPlanets, expired);
    cometPlanetIds = cometPlanetIds.filter((pid) => !expired.has(pid));
    for (const g of comets) g.planetIds = g.planetIds.filter((pid) => !expired.has(pid));
    for (let i = comets.length - 1; i >= 0; i--) {
      if (comets[i].planetIds.length === 0) comets.splice(i, 1);
    }
  }

  // Drop dead fleets after sweeps so combat sees the right contents.
  const aliveFleets = fleets.filter((f) => !fleetsToRemove.has(f.id));

  // --- 6. Combat resolution ---
  for (const [pid, planetFleets] of combatLists) {
    if (planetFleets.length === 0) continue;
    const planet = planets.find((p) => p.id === pid);
    if (!planet) continue;

    const playerShips = new Map<number, number>();
    for (const f of planetFleets) {
      playerShips.set(f.owner, (playerShips.get(f.owner) ?? 0) + f.ships);
    }
    if (playerShips.size === 0) continue;
    const sorted = [...playerShips.entries()].sort((a, b) => b[1] - a[1]);
    const [topPlayer, topShips] = sorted[0];

    let survivorOwner: number;
    let survivorShips: number;
    if (sorted.length > 1) {
      const secondShips = sorted[1][1];
      survivorShips = topShips - secondShips;
      if (sorted[0][1] === sorted[1][1]) survivorShips = 0;
      survivorOwner = survivorShips > 0 ? topPlayer : -1;
    } else {
      survivorOwner = topPlayer;
      survivorShips = topShips;
    }

    if (survivorShips > 0) {
      if (planet.owner === survivorOwner) {
        planet.ships += survivorShips;
      } else {
        planet.ships -= survivorShips;
        if (planet.ships < 0) {
          planet.owner = survivorOwner;
          planet.ships = Math.abs(planet.ships);
        }
      }
    }
  }

  // --- 7. Terminal check + scores ---
  let done = false;
  if (stepNum >= config.episodeSteps - 2) done = true;

  const aliveOwners = new Set<number>();
  for (const p of planets) if (p.owner !== -1) aliveOwners.add(p.owner);
  for (const f of aliveFleets) aliveOwners.add(f.owner);
  if (aliveOwners.size <= 1) done = true;

  const scores = new Array(numAgents).fill(0);
  for (const p of planets) if (p.owner !== -1 && p.owner < numAgents) scores[p.owner] += p.ships;
  for (const f of aliveFleets) if (f.owner < numAgents) scores[f.owner] += f.ships;

  let winners: number[] = [];
  if (done) {
    const max = Math.max(...scores);
    if (max > 0) {
      winners = scores.map((s, i) => (s === max ? i : -1)).filter((i) => i >= 0);
    }
  }

  return {
    step: stepNum,
    numAgents,
    angularVelocity,
    planets,
    initialPlanets,
    fleets: aliveFleets,
    nextFleetId,
    comets,
    cometPlanetIds,
    done,
    scores,
    winners,
    seed,
  };
}

function removeByIds(arr: Planet[], ids: Set<number>): void {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (ids.has(arr[i].id)) arr.splice(i, 1);
  }
}
