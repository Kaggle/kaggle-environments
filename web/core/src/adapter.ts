import { ReplayData } from './types';

export interface GameAdapter {
  mount(container: HTMLElement, initialData: ReplayData): void;
  render(step: number, replay: ReplayData, agents: any[], replayerInstance?: any): void;
  unmount(): void;
}
