export interface Planet {
  id: number;
  owner: number; // -1 for neutral
  x: number;
  y: number;
  radius: number;
  ships: number;
  production: number;
}

export interface Fleet {
  id: number;
  owner: number;
  x: number;
  y: number;
  angle: number;
  fromPlanetId: number;
  ships: number;
}

export interface CometGroup {
  planetIds: number[];
  paths: [number, number][][];
  pathIndex: number;
}

/** [planetId, angleRad, ships] */
export type Action = [number, number, number];

export interface GameState {
  step: number;
  numAgents: 2 | 4;
  angularVelocity: number;
  planets: Planet[];
  initialPlanets: Planet[];
  fleets: Fleet[];
  nextFleetId: number;
  comets: CometGroup[];
  cometPlanetIds: number[];
  done: boolean;
  scores: number[];
  /** indexes into players that won the game (only set when done) */
  winners: number[];
  seed: number;
}

export interface Config {
  episodeSteps: number;
  shipSpeed: number;
  cometSpeed: number;
  seed: number;
}

export type ActionsByPlayer = Record<number, Action[]>;
