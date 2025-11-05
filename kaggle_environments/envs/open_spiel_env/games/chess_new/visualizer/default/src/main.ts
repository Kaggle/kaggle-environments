import { Player, GameAdapter, ReplayData, ChessStep } from '@kaggle-environments/core';
import { renderer } from './chess_renderer';
import { render } from 'preact';

class LegacyAdapter implements GameAdapter {
  private container: HTMLElement | null = null;

  mount(container: HTMLElement): void {
    this.container = container;
  }

  render(step: number, replay: ReplayData): void {
    if (!this.container) return;
    this.container.innerHTML = ''; // Clear container before rendering
    renderer({
      parent: this.container,
      steps: replay.steps as ChessStep[],
      step: step,
      // These are probably not used but good to have
      width: this.container.clientWidth,
      height: this.container.clientHeight,
    });
  }

  unmount(): void {
    if (this.container) {
      render(null, this.container);
    }
    this.container = null;
  }
}

const app = document.getElementById('app');
if (app) {
  new Player(app, new LegacyAdapter());
}
