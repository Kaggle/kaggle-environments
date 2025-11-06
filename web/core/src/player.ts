import { GameAdapter } from './adapter';
import { ReplayData } from './types';
import cssString from './style.css?raw';
import { processEpisodeData } from './transformers';

// Inject CSS for a library bundle/build
(() => {
  if (typeof document === 'undefined') return; // Guard for non-browser environments
  const style = document.createElement('style');
  style.textContent = cssString;
  document.head.appendChild(style);
})();

export class ReplayVisualizer {
  private container: HTMLElement;
  private adapter: GameAdapter;
  private replay: ReplayData | null = null;
  private agents: any[] = [];
  private step = 0;
  private playing = false;
  private speed = 500; // ms per step
  private mounted = false;
  private showControls = true;
  private hmrState?: any; // Will hold the persistent state dev HMR

  // --- Element references ---
  private viewer: HTMLElement;
  private controls: HTMLElement;
  private playPauseButton: HTMLButtonElement;
  private playPauseIconPath: SVGPathElement;
  private prevButton: HTMLButtonElement;
  private nextButton: HTMLButtonElement;
  private stepSlider: HTMLInputElement;
  private stepCounter: HTMLSpanElement;

  constructor(container: HTMLElement, adapter: GameAdapter, options: { hmrState?: any } = {}) {
    this.container = container;
    this.adapter = adapter;

    // Store the HMR state if it was passed in
    if (import.meta.env?.DEV && options.hmrState) {
      this.hmrState = options.hmrState;
    }

    const playerDiv = document.createElement('div');
    playerDiv.className = 'player';

    this.viewer = document.createElement('div');
    this.viewer.className = 'viewer';

    this.controls = document.createElement('div');
    this.controls.className = 'controls';

    playerDiv.appendChild(this.viewer);
    playerDiv.appendChild(this.controls);
    this.container.innerHTML = '';
    this.container.appendChild(playerDiv);

    this.playPauseButton = this.createButton('play-pause', this.getIconHTML('play'));
    this.playPauseIconPath = this.playPauseButton.querySelector('path')!;

    this.prevButton = this.createButton('prev', this.getIconHTML('prev'));
    this.nextButton = this.createButton('next', this.getIconHTML('next'));

    this.stepSlider = document.createElement('input');
    this.stepSlider.type = 'range';
    this.stepSlider.min = '0';
    this.stepSlider.value = '0';

    this.stepCounter = document.createElement('span');
    this.stepCounter.className = 'step-counter';

    this.controls.appendChild(this.playPauseButton);
    this.controls.appendChild(this.prevButton);
    this.controls.appendChild(this.stepSlider);
    this.controls.appendChild(this.nextButton);
    this.controls.appendChild(this.stepCounter);

    // Wire up event listeners ONCE
    this.playPauseButton.addEventListener('click', () => (this.playing ? this.pause() : this.play()));
    this.prevButton.addEventListener('click', () => this.setStep(this.step - 1));
    this.nextButton.addEventListener('click', () => this.setStep(this.step + 1));
    this.stepSlider.addEventListener('input', (e) => {
      this.pause();
      this.setStep(parseInt((e.target as HTMLInputElement).value, 10));
    });

    this.loadData();
  }

  // Helper to create buttons with SVG icons
  private createButton(id: string, svgPathHTML: string): HTMLButtonElement {
    const button = document.createElement('button');
    button.id = id;
    button.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" viewBox="0 0 24 24" fill="#FFFFFF">
      ${svgPathHTML}
    </svg>
    `;
    return button;
  }

  // Helper to get SVG path data
  private getIconHTML(icon: 'play' | 'pause' | 'prev' | 'next'): string {
    switch (icon) {
      case 'play':
        return `<path d="M8 5v14l11-7z" /><path d="M0 0h24v24H0z" fill="none" />`;
      case 'pause':
        return `<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /><path d="M0 0h24v24H0z" fill="none" />`;
      case 'prev':
        return `<path d="M6 18V6h2v12H6zm3.5-6L18 6v12l-8.5-6z" />`;
      case 'next':
        return `<path d="M7 18l8.5-6L7 6v12zM15 6v12h2V6h-2z" />`;
    }
  }

  private loadData() {
    // 1. Restore basic HMR state (like UI toggles) if it exists.
    //    We use nullish coalescing (??) to set defaults if state is empty.
    if (this.hmrState) {
      this.showControls = this.hmrState.controls ?? true;
      this.playing = this.hmrState.playing ?? false;
    }

    // 2. (PRIORITY 1) Check for replay data *within* HMR state.
    if (this.hmrState?.replay) {
      const state = this.hmrState;

      this.setData(state.replay, state.agents, { skipRender: true });

      // Restore step *after* setData has set the slider max
      this.setStep(state.step);

      // renderControls() is called by setStep, but we call it
      // again to ensure the play/pause icon is correct.
      this.renderControls();

      if (this.playing) {
        this.tick();
      }
    }

    // 3. (PRIORITY 2) No HMR replay data. Check for VITE_REPLAY_FILE in dev mode.
    //    This block is now reachable even if this.hmrState exists.
    else if (import.meta.env?.DEV) {
      const replayFile = import.meta.env.VITE_REPLAY_FILE;
      if (replayFile) {
        fetch(replayFile)
          .then((res) => res.json())
          .then((data) => {
            this.setData(data, data.info.Agents);
          })
          .catch((err) => {
            console.error(`Error fetching ${replayFile}:`, err);
            this.viewer.innerHTML = `<div>Error loading ${replayFile}</div>`;
            this.renderControls(); // Render controls to show they are disabled
          });
      } else {
        // Dev mode, but no HMR data and no replayFile. Wait for postMessage.
        this.viewer.innerHTML = '<div>Waiting for replay data...</div>';
        this.renderControls(); // Apply restored showControls state
      }
    }

    // 4. (PRIORITY 3) Production build (or not DEV) and no HMR data.
    else {
      this.viewer.innerHTML = '<div>Loading...</div>';
      this.renderControls();
    }

    // 5. Add listener (always)
    window.addEventListener('message', this.handleMessage);
  }

  private handleMessage = (event: MessageEvent) => {
    if (!event.data) return;

    // Helper to update HMR state
    const updateHMRState = (key: string, value: any) => {
      // Check if HMR state object exists before writing to it
      if (this.hmrState) {
        this.hmrState[key] = value;
      }
    };

    if (typeof event.data.controls === 'boolean') {
      this.showControls = event.data.controls;
      updateHMRState('controls', this.showControls); // Save to HMR
      this.renderControls();
    }

    let needsRender = false;

    // Update agents if provided
    if (event.data.agents) {
      this.agents = event.data.agents;
      updateHMRState('agents', this.agents); // Save to HMR
      needsRender = true;
    }

    // Update replay object from 'environment'
    if (event.data.environment) {
      if (!this.replay) {
        this.replay = {
          name: 'unknown',
          version: 'unknown',
          steps: [],
          configuration: {},
          info: {},
        };
      }
      // Use Object.assign to merge new data without overwriting the whole object
      const { steps, ...rest } = event.data.environment;
      Object.assign(this.replay, rest);
      if (Array.isArray(steps)) {
        this.replay.steps = steps;
      }
      needsRender = true;
    }

    // Update steps from 'setSteps'
    if (event.data.setSteps && this.replay) {
      this.replay.steps = event.data.setSteps;
      needsRender = true;
    }

    // Overwrite replay object if a full 'replay' is provided
    if (event.data.replay) {
      this.replay = event.data.replay;
      needsRender = true;
    }

    // After any replay update, save it to HMR state
    if (needsRender && this.replay) {
      updateHMRState('replay', this.replay); // Save to HMR
    }

    // Update the current step
    if (typeof event.data.step === 'number') {
      this.step = event.data.step;
      updateHMRState('step', this.step); // Save to HMR
      needsRender = true;
    }

    // If any data was updated and we have a replay object, call setData.
    if (needsRender && this.replay) {
      this.setData(this.replay, this.agents);
    }
  };

  private setData(replay: ReplayData, agents: any[] = [], options: { skipRender?: boolean } = {}) {
    this.replay = replay;
    this.agents = agents;

    if (!this.mounted) {
      this.adapter.mount(this.viewer, this.replay);
      this.mounted = true;
    }

    // TODO(michaelaaron) - Turn this into something more reasonable.
    if (
      this?.replay?.steps &&
      !(this?.replay?.steps as any)?.[0]?.stepType &&
      this?.replay?.configuration?.openSpielGameName === 'repeated_poker'
    ) {
      this.replay.steps = processEpisodeData(this.replay, 'repeated_poker');
    }

    // --- HMR State Update ---
    if (this.hmrState) {
      this.hmrState.replay = this.replay;
      this.hmrState.agents = this.agents;
    }
    // --- End HMR Logic ---

    // Always update controls and render the current state.
    this.stepSlider.max = (this.replay.steps.length > 0 ? this.replay.steps.length - 1 : 0).toString();
    this.renderControls();

    // Only render/tick if not told to skip (e.g., during HMR restore)
    if (!options.skipRender) {
      this.adapter.render(this.step, this.replay, this.agents);
      this.tick();
    }
  }

  private setStep(step: number) {
    if (!this.replay) return;
    this.step = Math.max(0, Math.min(this.replay.steps.length - 1, step));

    // --- HMR State Update ---
    if (this.hmrState) {
      this.hmrState.step = this.step;
    }
    // --- End HMR Logic ---

    this.adapter.render(this.step, this.replay, this.agents);
    this.renderControls();
  }

  private play() {
    if (this.playing) return;
    this.playing = true;

    // --- HMR State Update ---
    if (this.hmrState) {
      this.hmrState.playing = true;
    }
    // --- End HMR Logic ---

    if (this.replay && this.step === this.replay.steps.length - 1) {
      this.setStep(0);
    }
    this.tick();
    this.renderControls();
  }

  private pause() {
    if (!this.playing) return;
    this.playing = false;

    // --- HMR State Update ---
    if (this.hmrState) {
      this.hmrState.playing = false;
    }
    // --- End HMR Logic ---

    this.renderControls();
  }

  private tick = () => {
    if (!this.playing || !this.replay) return;

    if (this.step >= this.replay.steps.length - 1) {
      this.playing = false;
      this.renderControls();
      return;
    }

    setTimeout(() => {
      this.setStep(this.step + 1);
      this.tick();
    }, this.speed);
  };

  private renderControls() {
    // Step 1: Handle visibility based *only* on showControls
    if (!this.showControls) {
      this.controls.style.display = 'none';
      return;
    }
    this.controls.style.display = 'flex';

    // Step 2: Handle the *state* of the controls,
    // which *does* depend on having a replay.
    if (!this.replay) {
      // No replay data. Disable buttons and show default text.
      this.playPauseButton.disabled = true;
      this.prevButton.disabled = true;
      this.nextButton.disabled = true;
      this.stepSlider.disabled = true;
      this.stepSlider.value = '0';
      this.stepSlider.max = '0';
      this.stepCounter.textContent = '0 / 0';

      // Make sure icon is 'play'
      const newIconHTML = this.getIconHTML('play');
      if (this.playPauseIconPath.outerHTML !== newIconHTML) {
        this.playPauseIconPath.outerHTML = newIconHTML;
        this.playPauseIconPath = this.playPauseButton.querySelector('path')!;
      }
      return;
    }

    // --- We have a replay, so render full state ---
    this.playPauseButton.disabled = false;
    this.stepSlider.disabled = false;

    const maxSteps = this.replay.steps.length - 1;

    // Update only what's necessary
    const newIconHTML = this.getIconHTML(this.playing ? 'pause' : 'play');
    if (this.playPauseIconPath.outerHTML !== newIconHTML) {
      this.playPauseIconPath.outerHTML = newIconHTML;
      this.playPauseIconPath = this.playPauseButton.querySelector('path')!;
    }

    this.prevButton.disabled = this.step === 0;
    this.nextButton.disabled = this.step === maxSteps;

    this.stepSlider.value = this.step.toString();
    // Ensure max is set correctly
    this.stepSlider.max = (maxSteps >= 0 ? maxSteps : 0).toString();
    this.stepCounter.textContent = `${this.step + 1} / ${maxSteps + 1}`;
  }

  /**
   * Public method to clean up all side effects (listeners, loops)
   * when the instance is about to be destroyed.
   * This is called by the factory during an HMR update.
   */
  public cleanup() {
    // 1. Stop any running loops (setTimeout)
    //    This is critical to prevent old ticks from running.
    this.pause();

    // 2. Remove global event listeners
    //    This is the most common source of HMR-related memory leaks.
    window.removeEventListener('message', this.handleMessage);

    // 3. Tell the adapter to unmount
    //    This allows your renderer (Preact) to clean up its DOM nodes.
    if (this.mounted) {
      this.adapter.unmount();
      this.mounted = false;
    }

    // 4. (Optional) Clear the container to show it's gone
    this.container.innerHTML = '';
  }
}
