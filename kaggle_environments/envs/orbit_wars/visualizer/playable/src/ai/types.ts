import type { Action, Fleet, Planet } from '../engine/types';

export interface Observation {
  player: number;
  planets: Planet[];
  fleets: Fleet[];
  angularVelocity: number;
  step: number;
  numAgents: number;
}

export type AgentFn = (obs: Observation) => Action[];
