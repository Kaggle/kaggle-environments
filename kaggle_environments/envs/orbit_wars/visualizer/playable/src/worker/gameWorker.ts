/// <reference lib="webworker" />
import { initGameState, step } from '../engine/interpreter';
import type { ActionsByPlayer, Config, GameState } from '../engine/types';
import { AGENTS } from '../ai';
import type { Observation } from '../ai/types';
import type { Req, Res, SlotConfig } from './protocol';

let state: GameState | null = null;
let config: Config | null = null;
let slots: SlotConfig[] = [];

function buildObservation(s: GameState, player: number): Observation {
  return {
    player,
    planets: s.planets,
    fleets: s.fleets,
    angularVelocity: s.angularVelocity,
    step: s.step,
    numAgents: s.numAgents,
  };
}

function collectAiActions(s: GameState, humanActions: ActionsByPlayer): ActionsByPlayer {
  const all: ActionsByPlayer = { ...humanActions };
  for (let pid = 0; pid < slots.length; pid++) {
    const slot = slots[pid];
    if (slot.kind === 'ai') {
      const agent = AGENTS[slot.agentId];
      if (!agent) {
        all[pid] = [];
        continue;
      }
      try {
        all[pid] = agent.fn(buildObservation(s, pid));
      } catch (err) {
        console.error('AI agent error', slot.agentId, err);
        all[pid] = [];
      }
    } else if (!(pid in all)) {
      all[pid] = [];
    }
  }
  return all;
}

function send(res: Res) {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(res);
}

self.addEventListener('message', (evt: MessageEvent<Req>) => {
  const msg = evt.data;
  try {
    switch (msg.type) {
      case 'INIT': {
        config = msg.config;
        slots = msg.slots;
        state = initGameState(msg.numAgents, msg.config);
        send({ type: 'STATE', reqId: msg.reqId, state });
        break;
      }
      case 'RESET': {
        if (!config) throw new Error('Worker not initialized');
        state = initGameState(slots.length as 2 | 4, config);
        send({ type: 'STATE', reqId: msg.reqId, state });
        break;
      }
      case 'STEP': {
        if (!state || !config) throw new Error('Worker not initialized');
        if (state.done) {
          send({ type: 'STATE', reqId: msg.reqId, state });
          break;
        }
        const all = collectAiActions(state, msg.humanActions);
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
