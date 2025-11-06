import { createReplayVisualizer, GameAdapter, ReplayData, ChessStep } from '@kaggle-environments/core';
import { renderer } from './chess_renderer';
import { render } from 'preact';

export interface RendererOptions {
  steps: ChessStep[];
  step: number;
  parent: HTMLElement;
  playerNames: string[];
  width?: number;
  height?: number;
  /* I think this is meant to represent the HTML element
  in which to render the visualizer? I couldn't find a replay
  that includes that property but we'll keep it just in case. */
  viewer?: any;
}

class LegacyAdapter implements GameAdapter {
  private container: HTMLElement | null = null;

  mount(container: HTMLElement): void {
    this.container = container;
  }

  render(step: number, replay: ReplayData): void {
    if (!this.container) return;
    this.container.innerHTML = ''; // Clear container before rendering

    const renderData: RendererOptions = {
      parent: this.container,
      steps: replay.steps as ChessStep[],
      step: step,
      playerNames: replay.info?.TeamNames ?? ['', ''],
      width: this.container.clientWidth,
      height: this.container.clientHeight,
    };

    renderer(renderData);
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
  // Set up an HMR boundary for development
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }

  createReplayVisualizer(app, new LegacyAdapter());
}
