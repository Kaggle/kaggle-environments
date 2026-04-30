import { describe, it, expect } from 'vitest';
import { initGameState, step, fleetSpeed } from '../interpreter';
import type { Config, GameState, Planet } from '../types';
import { DEFAULT_COMET_SPEED, DEFAULT_EPISODE_STEPS, DEFAULT_SHIP_SPEED } from '../constants';

const baseConfig: Config = {
  episodeSteps: DEFAULT_EPISODE_STEPS,
  shipSpeed: DEFAULT_SHIP_SPEED,
  cometSpeed: DEFAULT_COMET_SPEED,
  seed: 12345,
};

function freshGame(seed = 12345): GameState {
  return initGameState(2, { ...baseConfig, seed });
}

describe('fleetSpeed', () => {
  it('returns 1 for tiny fleets', () => {
    expect(fleetSpeed(1, 6)).toBe(1);
  });
  it('caps at maxSpeed', () => {
    expect(fleetSpeed(1_000_000, 6)).toBe(6);
  });
  it('grows monotonically', () => {
    let prev = 0;
    for (const n of [2, 5, 10, 50, 200, 999]) {
      const s = fleetSpeed(n, 6);
      expect(s).toBeGreaterThanOrEqual(prev);
      prev = s;
    }
  });
});

describe('initGameState', () => {
  it('produces a valid 2p game with home planets', () => {
    const s = freshGame();
    expect(s.numAgents).toBe(2);
    expect(s.planets.length).toBeGreaterThan(0);
    const owners = new Set(s.planets.map((p) => p.owner));
    expect(owners.has(0)).toBe(true);
    expect(owners.has(1)).toBe(true);
  });

  it('reproduces the same layout for the same seed', () => {
    const a = freshGame(777);
    const b = freshGame(777);
    expect(a.planets.length).toBe(b.planets.length);
    for (let i = 0; i < a.planets.length; i++) {
      expect(a.planets[i]).toEqual(b.planets[i]);
    }
  });
});

describe('step phases', () => {
  it('production accrues for owned planets', () => {
    const s = freshGame();
    const home = s.planets.find((p) => p.owner === 0)!;
    const before = home.ships;
    const next = step(s, {}, baseConfig);
    const newHome = next.planets.find((p) => p.id === home.id)!;
    expect(newHome.ships).toBe(before + home.production);
  });

  it('launching a fleet deducts ships and creates a Fleet', () => {
    const s = freshGame();
    const home = s.planets.find((p) => p.owner === 0)!;
    const next = step(s, { 0: [[home.id, 0, 5]] }, baseConfig);
    const newHome = next.planets.find((p) => p.id === home.id)!;
    // production added, then 5 deducted
    expect(newHome.ships).toBe(home.ships + home.production - 5);
    expect(next.fleets.length).toBe(1);
    expect(next.fleets[0].owner).toBe(0);
    expect(next.fleets[0].ships).toBe(5);
  });

  it('rejects fleet from a planet not owned by the player', () => {
    const s = freshGame();
    const enemy = s.planets.find((p) => p.owner === 1)!;
    const next = step(s, { 0: [[enemy.id, 0, 5]] }, baseConfig);
    expect(next.fleets.length).toBe(0);
  });

  it('rejects oversized fleet (more ships than available)', () => {
    const s = freshGame();
    const home = s.planets.find((p) => p.owner === 0)!;
    const next = step(s, { 0: [[home.id, 0, home.ships + 1000]] }, baseConfig);
    expect(next.fleets.length).toBe(0);
  });

  it('combat: tied opposing forces leave the planet untouched', () => {
    // Two opposing fleets both within hit range of planet1 in step 1.
    // Place them right at the planet edge so guaranteed collision.
    const planet1: Planet = { id: 1, owner: -1, x: 90, y: 50, radius: 2, ships: 5, production: 0 };
    const synthetic: GameState = {
      step: 0,
      numAgents: 2,
      angularVelocity: 0,
      planets: [planet1],
      initialPlanets: [{ ...planet1 }],
      fleets: [
        // Player 0 fleet right next to planet, aimed at it
        { id: 0, owner: 0, x: 88.5, y: 50, angle: 0, fromPlanetId: -1, ships: 30 },
        // Player 1 fleet right next to planet, aimed at it from opposite side
        { id: 1, owner: 1, x: 91.5, y: 50, angle: Math.PI, fromPlanetId: -1, ships: 30 },
      ],
      nextFleetId: 2,
      comets: [],
      cometPlanetIds: [],
      done: false,
      scores: [0, 0],
      winners: [],
      seed: 1,
    };
    const next = step(synthetic, {}, { ...baseConfig, episodeSteps: 1000 });
    const target = next.planets.find((p) => p.id === 1);
    // Tie at planet → survivor_ships = 0 → planet untouched (still neutral, ships=5).
    expect(target?.owner).toBe(-1);
    expect(target?.ships).toBe(5);
  });

  it('combat: overwhelming force flips ownership', () => {
    const planet1: Planet = { id: 1, owner: 1, x: 90, y: 50, radius: 2, ships: 10, production: 0 };
    const synthetic: GameState = {
      step: 0,
      numAgents: 2,
      angularVelocity: 0,
      planets: [planet1],
      initialPlanets: [{ ...planet1 }],
      fleets: [{ id: 0, owner: 0, x: 88, y: 50, angle: 0, fromPlanetId: -1, ships: 50 }],
      nextFleetId: 1,
      comets: [],
      cometPlanetIds: [],
      done: false,
      scores: [0, 0],
      winners: [],
      seed: 1,
    };
    const next = step(synthetic, {}, { ...baseConfig, episodeSteps: 1000 });
    const target = next.planets.find((p) => p.id === 1)!;
    expect(target.owner).toBe(0);
    expect(target.ships).toBe(40); // |10 - 50|
  });

  it('terminates and assigns a winner when only one player remains', () => {
    const planet0: Planet = { id: 0, owner: 0, x: 10, y: 50, radius: 2, ships: 100, production: 1 };
    const synthetic: GameState = {
      step: 0,
      numAgents: 2,
      angularVelocity: 0,
      planets: [planet0],
      initialPlanets: [{ ...planet0 }],
      fleets: [],
      nextFleetId: 0,
      comets: [],
      cometPlanetIds: [],
      done: false,
      scores: [0, 0],
      winners: [],
      seed: 1,
    };
    const next = step(synthetic, {}, { ...baseConfig, episodeSteps: 1000 });
    expect(next.done).toBe(true);
    expect(next.winners).toEqual([0]);
  });

  it('does not advance after game is done', () => {
    let s = freshGame();
    // Force done
    s = { ...s, done: true };
    const next = step(s, {}, baseConfig);
    expect(next).toBe(s);
  });
});
