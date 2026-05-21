/**
 * Per-agent observation. Mirrors the dict passed to `random_agent` /
 * `starter_agent` in kaggriculture.py — the engine assembles one of these
 * per player from the shared GameState.
 */

import type { Farm, Market, PlayerAction, Private, Town } from '../engine/types';

export interface Observation {
  player: number;
  step: number;
  day: number;
  hour: number;
  numAgents: number;
  farms: Farm[];
  private: Private;
  market: Market;
  town: Town;
}

export type AgentFn = (obs: Observation) => PlayerAction;
