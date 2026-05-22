/// <reference lib="webworker" />
/**
 * Game worker: owns the authoritative GameState and runs `step()` on demand.
 * AI slots are evaluated here so the main thread never sees their cost.
 * Mirrors `orbit_wars/visualizer/playable/src/worker/gameWorker.ts`.
 */

import { AGENTS } from '../ai';
import type { Observation } from '../ai/types';
import { initGameState, pickSeed } from '../engine/state';
import { step } from '../engine/interpreter';
import type { Config, GameState, PlayerAction } from '../engine/types';
import type { HumanActions, Req, Res, SlotConfig } from './protocol';

let state: GameState | null = null;
let config: Config | null = null;
let slots: SlotConfig[] = [];

function buildObservation(s: GameState, player: number): Observation {
  return {
    player,
    step: s.step,
    day: s.day,
    hour: s.hour,
    numAgents: s.numAgents,
    farms: s.farms,
    private: s.privates[player],
    market: s.market,
    town: s.town,
  };
}

function collectActions(s: GameState, humanActions: HumanActions): PlayerAction[] {
  const out: PlayerAction[] = [];
  for (let pid = 0; pid < slots.length; pid++) {
    const slot = slots[pid];
    if (slot.kind === 'human') {
      out.push(humanActions[pid] ?? { farmer: ['PASS'], hands: [], market: [] });
      continue;
    }
    const agent = AGENTS[slot.agentId];
    if (!agent) {
      out.push({ farmer: ['PASS'], hands: [], market: [] });
      continue;
    }
    try {
      out.push(agent.fn(buildObservation(s, pid)));
    } catch (err) {
      console.error('AI agent error', slot.agentId, err);
      out.push({ farmer: ['PASS'], hands: [], market: [] });
    }
  }
  return out;
}

function resolveSeed(cfg: Config): number {
  return cfg.seed ?? pickSeed();
}

function send(res: Res): void {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(res);
}

self.addEventListener('message', (evt: MessageEvent<Req>) => {
  const msg = evt.data;
  try {
    switch (msg.type) {
      case 'INIT': {
        config = msg.config;
        slots = msg.slots;
        state = initGameState(msg.numAgents, msg.config, resolveSeed(msg.config));
        send({ type: 'STATE', reqId: msg.reqId, state });
        break;
      }
      case 'RESET': {
        if (!config) throw new Error('Worker not initialized');
        state = initGameState(slots.length, config, resolveSeed(config));
        send({ type: 'STATE', reqId: msg.reqId, state });
        break;
      }
      case 'STEP': {
        if (!state || !config) throw new Error('Worker not initialized');
        if (state.done) {
          send({ type: 'STATE', reqId: msg.reqId, state });
          break;
        }
        const all = collectActions(state, msg.humanActions);
        state = step(state, all, config);
        send({ type: 'STATE', reqId: msg.reqId, state });
        break;
      }
      case 'GET_STATE': {
        if (!state) throw new Error('Worker not initialized');
        send({ type: 'STATE', reqId: msg.reqId, state });
        break;
      }
    }
  } catch (err) {
    send({ type: 'ERROR', reqId: msg.reqId, message: err instanceof Error ? err.message : String(err) });
  }
});
