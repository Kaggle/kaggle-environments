import { GameAdapter } from './adapter';
import { ReplayData } from './types';
import { render } from 'preact';

// The legacy renderer function signature
type LegacyRenderer = (options: any, container?: HTMLElement) => void;

export class LegacyAdapter implements GameAdapter {
  private container: HTMLElement | null = null;
  private renderer: LegacyRenderer;

  constructor(renderer: LegacyRenderer) {
    this.renderer = renderer;
  }

  mount(container: HTMLElement): void {
    this.container = container;
  }

  render(step: number, replay: ReplayData, agents: any[]): void {
    if (!this.container) return;

    // Clear container before rendering, as legacy renderers often append.
    this.container.innerHTML = '';

    const renderOptions = {
      // For chess/poker
      parent: this.container,
      steps: replay.steps,
      playerNames: replay.info?.TeamNames || agents.map((a) => a.name),

      // For werewolf and others
      replay: replay,
      agents: agents,

      // Common properties
      step: step,
      width: this.container.clientWidth,
      height: this.container.clientHeight,
    };

    // Some legacy renderers take the container as a second argument.
    // Others expect it inside the options object. We provide both.
    this.renderer(renderOptions, this.container);
  }

  unmount(): void {
    if (this.container) {
      render(null, this.container);
    }
    this.container = null;
  }
}
