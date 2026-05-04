import type { ActionsByPlayer, Config, GameState } from '../engine/types';
import type { Req, Res, SlotConfig } from './protocol';

type Pending = {
  resolve: (state: GameState) => void;
  reject: (err: Error) => void;
};

export class WorkerClient {
  private worker: Worker;
  private pending = new Map<string, Pending>();
  private nextId = 0;

  constructor() {
    this.worker = new Worker(new URL('./gameWorker.ts', import.meta.url), { type: 'module' });
    this.worker.addEventListener('message', (evt: MessageEvent<Res>) => {
      const res = evt.data;
      const p = this.pending.get(res.reqId);
      if (!p) return;
      this.pending.delete(res.reqId);
      if (res.type === 'ERROR') p.reject(new Error(res.message));
      else p.resolve(res.state);
    });
  }

  private nextReqId(): string {
    return String(this.nextId++);
  }

  private dispatch(req: Req): Promise<GameState> {
    return new Promise<GameState>((resolve, reject) => {
      this.pending.set(req.reqId, { resolve, reject });
      this.worker.postMessage(req);
    });
  }

  init(config: Config, numAgents: 2 | 4, slots: SlotConfig[]): Promise<GameState> {
    return this.dispatch({ type: 'INIT', reqId: this.nextReqId(), config, numAgents, slots });
  }

  step(humanActions: ActionsByPlayer): Promise<GameState> {
    return this.dispatch({ type: 'STEP', reqId: this.nextReqId(), humanActions });
  }

  reset(): Promise<GameState> {
    return this.dispatch({ type: 'RESET', reqId: this.nextReqId() });
  }

  getState(): Promise<GameState> {
    return this.dispatch({ type: 'GET_STATE', reqId: this.nextReqId() });
  }

  terminate(): void {
    this.worker.terminate();
  }
}
