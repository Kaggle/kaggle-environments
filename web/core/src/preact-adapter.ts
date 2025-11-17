import { h, render, FunctionComponent } from 'preact';
import { GameAdapter } from './adapter';
import { BaseGameStep, ReplayData } from './types';

export interface RendererProps<TSteps = BaseGameStep[]> {
  replay: ReplayData<TSteps>;
  step: number;
  agents: any[];
}

export class PreactAdapter<TSteps = BaseGameStep[]> implements GameAdapter<TSteps> {
  private container: HTMLElement | null = null;
  private renderer: FunctionComponent<RendererProps<TSteps>>;

  constructor(renderer: FunctionComponent<RendererProps<TSteps>>) {
    this.renderer = renderer;
  }

  mount(container: HTMLElement, initialData: ReplayData<TSteps>): void {
    this.container = container;
    this.render(0, initialData, []);
  }

  render(step: number, replay: ReplayData<TSteps>, agents: any[]): void {
    if (!this.container) return;
    render(h(this.renderer, { replay, step, agents }), this.container);
  }

  unmount(): void {
    if (!this.container) return;
    render(null, this.container);
    this.container = null;
  }
}
