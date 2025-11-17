import { BaseGameStep, ReplayData } from './types';

export interface GameAdapter<TSteps = BaseGameStep[]> {
  mount(container: HTMLElement, initialData: ReplayData<TSteps>): void;
  render(step: number, replay: ReplayData<TSteps>, agents: any[], replayerInstance?: any): void;
  unmount(): void;
}
