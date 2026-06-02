/**
 * Postmessage wire protocol between the main thread and the game worker.
 * Mirrors the orbit_wars shape — INIT once, then STEP per turn (passing in
 * only the human players' actions; the worker fills in AI actions itself).
 */

import type { Config, GameState, PlayerAction } from '../engine/types';

export type SlotConfig = { kind: 'human' } | { kind: 'ai'; agentId: string };

/** Sparse map: player id → action. Missing entries fall through to AI or PASS. */
export type HumanActions = Record<number, PlayerAction>;

export type Req =
  | { type: 'INIT'; reqId: string; config: Config; numAgents: number; slots: SlotConfig[] }
  | { type: 'STEP'; reqId: string; humanActions: HumanActions }
  | { type: 'RESET'; reqId: string }
  | { type: 'GET_STATE'; reqId: string };

export type Res =
  | { type: 'STATE'; reqId: string; state: GameState }
  | { type: 'ERROR'; reqId: string; message: string };
