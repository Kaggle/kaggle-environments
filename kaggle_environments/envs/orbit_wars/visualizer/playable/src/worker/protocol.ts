import type { ActionsByPlayer, Config, GameState } from '../engine/types';

export type SlotConfig = { kind: 'human' } | { kind: 'ai'; agentId: string };

export type Req =
  | { type: 'INIT'; reqId: string; config: Config; numAgents: 2 | 4; slots: SlotConfig[] }
  | { type: 'STEP'; reqId: string; humanActions: ActionsByPlayer }
  | { type: 'RESET'; reqId: string }
  | { type: 'GET_STATE'; reqId: string };

export type Res =
  | { type: 'STATE'; reqId: string; state: GameState }
  | { type: 'ERROR'; reqId: string; message: string };
