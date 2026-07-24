import {
  createReplayVisualizer,
  GameAdapter,
  processEpisodeData,
  ReplayData,
  BaseGameStep,
} from '@kaggle-environments/core';
import { luxNormalizeReplay } from './luxTransformer';

// Helper to load a script
function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.onload = () => resolve();
    script.onerror = (err) => {
      console.error('Failed to load script', src, err);
      reject(new Error(`Failed to load script: ${src}`));
    };
    document.head.appendChild(script);
  });
}

const SCRIPTS = [
  'https://unpkg.com/lux-viewer-2021@latest/dist/vendors~app~phaser.js',
  'https://unpkg.com/lux-viewer-2021@latest/dist/vendors~app.js',
  'https://unpkg.com/lux-viewer-2021@latest/dist/app.js',
  'https://unpkg.com/lux-viewer-2021@latest/dist/phaser.js',
];

let scriptsPromise: Promise<void[]> | null = null;

function loadViewerScripts(): Promise<void[]> {
  if (scriptsPromise) {
    return scriptsPromise;
  }
  scriptsPromise = Promise.all(SCRIPTS.map(loadScript));
  return scriptsPromise;
}

declare global {
  interface Window {
    kaggle: any;
  }
}

class LuxAdapter implements GameAdapter<BaseGameStep[]> {
  private el: HTMLElement | null = null;

  mount(el: HTMLElement, replay: ReplayData<BaseGameStep[]>) {
    this.el = el;

    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }

    const root = document.createElement('div');
    root.id = 'root';
    el.appendChild(root);

    window.kaggle = { environment: replay };

    loadViewerScripts().catch((error) => {
      el.innerText = 'Error loading Lux AI Viewer scripts.';
      console.error(error);
    });
  }

  unmount(): void {
    if (this.el) {
      while (this.el.firstChild) {
        this.el.removeChild(this.el.firstChild);
      }
    }
  }

  render() {
    // lux-viewer handles its own state.
  }
}

const app = document.getElementById('app');
if (app) {
  createReplayVisualizer(app, new LuxAdapter(), {
    // Normalize LLM-harness dict actions ({submission, thoughts, ...}) down to
    // the bare command list the external lux-viewer-2021 board renderer expects.
    // Without this the viewer throws on `action.map` for LLM replays.
    transformer: (replay) => luxNormalizeReplay(processEpisodeData(replay, 'lux_ai_2021')),
  });
}
