import { GameAdapter } from './adapter';
import { BaseGameStep, ReplayData, ReplayMode } from './types';
import { getGameStepRenderTime } from './transformers';
import './style.css';

export class ReplayVisualizer<TSteps extends BaseGameStep[] = BaseGameStep[]> {
  private container: HTMLElement;
  private adapter: GameAdapter<TSteps>;
  private replay: ReplayData<TSteps> | null = null;
  private agents: any[] = [];
  private step = 0;
  private playing = false;
  private speedModifier = 1;
  private replayMode: ReplayMode = 'condensed';
  private externallyControlled = false;
  private mounted = false;
  private showControls = true;
  private showLegend = false;
  private hmrState?: any; // Will hold the persistent state dev HMR
  private transformer?: (replay: ReplayData) => ReplayData;

  // --- Element references ---
  private viewer: HTMLElement;
  private controls: HTMLElement;
  private legend: HTMLElement;
  private playPauseButton: HTMLButtonElement;
  private playPauseIconPath: SVGPathElement;
  private prevButton: HTMLButtonElement;
  private nextButton: HTMLButtonElement;
  private stepSlider: HTMLInputElement;
  private stepCounter: HTMLSpanElement;
  private speedSelector: HTMLSelectElement;

  constructor(
    container: HTMLElement,
    adapter: GameAdapter<TSteps>,
    options: { hmrState?: any; transformer?: (replay: ReplayData) => ReplayData } = {}
  ) {
    this.container = container;
    this.adapter = adapter;
    this.transformer = options.transformer;

    // Store the HMR state if it was passed in
    if (import.meta.env?.DEV && options.hmrState) {
      this.hmrState = options.hmrState;
    }

    const playerDiv = document.createElement('div');
    playerDiv.className = 'player';

    this.viewer = document.createElement('div');
    this.viewer.className = 'viewer';

    this.legend = document.createElement('div');
    this.legend.className = 'legend';

    this.controls = document.createElement('div');
    this.controls.className = 'controls';

    playerDiv.appendChild(this.viewer);
    playerDiv.appendChild(this.legend);
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

    // Speed selector dropdown
    this.speedSelector = document.createElement('select');
    this.speedSelector.className = 'speed-selector';
    const speeds = [0.5, 0.75, 1, 1.5, 2];
    speeds.forEach((speed) => {
      const option = document.createElement('option');
      option.value = speed.toString();
      option.textContent = `${speed}x`;
      if (speed === 1) option.selected = true;
      this.speedSelector.appendChild(option);
    });

    this.controls.appendChild(this.playPauseButton);
    this.controls.appendChild(this.prevButton);
    this.controls.appendChild(this.stepSlider);
    this.controls.appendChild(this.nextButton);
    this.controls.appendChild(this.stepCounter);
    this.controls.appendChild(this.speedSelector);

    // Wire up event listeners ONCE
    this.playPauseButton.addEventListener('click', () => (this.playing ? this.pause() : this.play()));
    this.prevButton.addEventListener('click', () => this.setStep(this.step - 1));
    this.nextButton.addEventListener('click', () => this.setStep(this.step + 1));
    this.stepSlider.addEventListener('input', (e) => {
      this.pause();
      this.setStep(parseInt((e.target as HTMLInputElement).value, 10));
    });
    this.speedSelector.addEventListener('change', (e) => {
      const newSpeed = parseFloat((e.target as HTMLSelectElement).value);
      this.setSpeed(newSpeed);
    });

    // Listen for speed changes from game-specific renderers
    window.addEventListener('replayer-speed', this.handleReplayerSpeed);

    // Keyboard navigation
    window.addEventListener('keydown', this.handleKeyDown);

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

  /**
   * Notify the parent frame of state changes for synchronization.
   */
  private notifyParent(state: { step?: number; playing?: boolean; speed?: number }) {
    window.parent.postMessage(state, '*');
  }

  private loadData() {
    // 1. Restore basic HMR state (like UI toggles) if it exists.
    //    We use nullish coalescing (??) to set defaults if state is empty.
    if (this.hmrState) {
      this.showControls = this.hmrState.controls ?? true;
      this.showLegend = this.hmrState.legend ?? false;
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
      this.renderLegend();

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
            // TODO: Move to game-specific config for showControls/showLegend/stepTime/etc.
            if (this.replay?.name === 'halite' || this.replay?.name === 'hungry_geese') {
              this.showLegend = true;
            }
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
        this.renderLegend();
      }
    }

    // 4. (PRIORITY 3) Production build (or not DEV) and no HMR data.
    else {
      this.viewer.innerHTML = '<div>Loading...</div>';
      this.renderControls();
      this.renderLegend();
    }

    // 5. Add listener (always)
    window.addEventListener('message', this.handleMessage);
  }

  private handleKeyDown = (event: KeyboardEvent) => {
    // Ignore if focus is on an input element (e.g., the slider)
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
      return;
    }

    switch (event.key) {
      case 'ArrowLeft':
        event.preventDefault();
        this.setStep(this.step - 1);
        break;
      case 'ArrowRight':
        event.preventDefault();
        this.setStep(this.step + 1);
        break;
      case ' ':
      case 'Enter':
        event.preventDefault();
        if (this.playing) {
          this.pause();
        } else {
          this.play();
        }
        break;
    }
  };

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
      // When controls are hidden, we're externally controlled
      this.externallyControlled = !event.data.controls;
      updateHMRState('controls', this.showControls); // Save to HMR
      this.renderControls();
    }

    // Handle speed changes from parent
    if (typeof event.data.speed === 'number') {
      this.setSpeed(event.data.speed, true); // fromExternal=true to avoid echo
    }

    // Handle replayMode changes from parent
    if (event.data.replayMode) {
      this.replayMode = event.data.replayMode;
      updateHMRState('replayMode', this.replayMode);
    }

    if (typeof event.data.legend === 'boolean') {
      this.showLegend = event.data.legend;
      updateHMRState('legend', this.showLegend);
      this.renderLegend();
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
      const { steps, ...rest } = event.data.environment;
      if (!this.replay) {
        this.replay = {
          name: 'unknown',
          version: 'unknown',
          steps: [] as unknown as TSteps,
          configuration: {},
          info: {},
          ...rest,
        };
      } else {
        Object.assign(this.replay, rest);
      }

      if (this.replay && Array.isArray(steps)) {
        this.replay.steps = steps as TSteps;
      }
      needsRender = true;
    }

    // Update steps from 'setSteps' - this is a special case.
    // The renderer is taking control of the steps, and we just need to update
    // our controls to match, not trigger a destructive re-render.
    if (event.data.setSteps && this.replay) {
      this.replay.steps = event.data.setSteps;
      this.stepSlider.max = (this.replay.steps.length > 0 ? this.replay.steps.length - 1 : 0).toString();
      this.renderControls();
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

    // Update the current step (from external source, so don't notify back)
    if (typeof event.data.step === 'number') {
      this.step = event.data.step;
      updateHMRState('step', this.step); // Save to HMR
      needsRender = true;
    }

    // Handle external play/pause control
    // IMPORTANT: Use setPlayingState instead of play()/pause() to avoid starting
    // the internal tick loop. External controllers manage their own step advancement.
    if (typeof event.data.playing === 'boolean') {
      this.setPlayingState(event.data.playing);
    }

    // If any data was updated and we have a replay object, call setData.
    // Skip tick if:
    // - Internal controls are hidden (showControls === false), OR
    // - External controller is explicitly setting play state, OR
    // - This is just a step update (not initial data) - tick is already running if playing
    // This preserves auto-play behavior for games using internal controls only,
    // while preventing multiple tick loops from starting on step echoes.
    if (needsRender && this.replay) {
      const isStepOnlyUpdate = typeof event.data.step === 'number' && !event.data.environment && !event.data.replay;
      const isExternallyControlled = !this.showControls || typeof event.data.playing === 'boolean';
      this.setData(this.replay, this.agents, { skipTick: isExternallyControlled || isStepOnlyUpdate });
    }
  };

  private setData(
    replay: ReplayData<TSteps>,
    agents: any[] = [],
    options: { skipRender?: boolean; skipTick?: boolean } = {}
  ) {
    // Apply the transformer if one is provided
    const transformedReplay = this.transformer ? this.transformer(replay) : replay;

    this.replay = transformedReplay as ReplayData<TSteps>;
    this.agents = agents;

    if (!this.mounted && this.replay) {
      this.adapter.mount(this.viewer, this.replay);
      this.mounted = true;
    }

    // --- HMR State Update ---
    if (this.hmrState) {
      this.hmrState.replay = this.replay;
      this.hmrState.agents = this.agents;
    }
    // --- End HMR Logic ---

    // Always update controls and render the current state.
    if (this.replay) {
      this.stepSlider.max = (this.replay.steps.length > 0 ? this.replay.steps.length - 1 : 0).toString();
    }
    this.renderControls();
    this.renderLegend();

    // Only render/tick if not told to skip (e.g., during HMR restore)
    if (!options.skipRender && this.replay) {
      this.adapter.render(this.step, this.replay, this.agents, this);
      // Only start tick loop if not externally controlled
      if (!options.skipTick) {
        this.tick();
      }
    }
  }

  private setStep(step: number, notify = true) {
    if (!this.replay) return;
    this.step = Math.max(0, Math.min(this.replay.steps.length - 1, step));

    // --- HMR State Update ---
    if (this.hmrState) {
      this.hmrState.step = this.step;
    }
    // --- End HMR Logic ---

    this.adapter.render(this.step, this.replay, this.agents, this);
    this.renderControls();
    if (notify) {
      this.notifyParent({ step: this.step });
    }
  }

  public setAgents(agents: any[]) {
    this.agents = agents;
    if (this.hmrState) {
      this.hmrState.agents = this.agents;
    }
    this.renderLegend();
  }

  private play() {
    if (this.playing) return;
    this.playing = true;

    // --- HMR State Update ---
    if (this.hmrState) {
      this.hmrState.playing = true;
    }
    // --- End HMR Logic ---

    // When externally controlled, just notify parent - don't reset step or start tick
    if (this.externallyControlled) {
      this.renderControls();
      this.notifyParent({ playing: true });
      return;
    }

    if (this.replay && this.step === this.replay.steps.length - 1) {
      this.setStep(0);
    }
    this.tick();
    this.renderControls();
    this.notifyParent({ playing: true });
  }

  private pause(notify = true) {
    if (!this.playing) return;
    this.playing = false;

    // --- HMR State Update ---
    if (this.hmrState) {
      this.hmrState.playing = false;
    }
    // --- End HMR Logic ---

    this.renderControls();
    if (notify) {
      this.notifyParent({ playing: false });
    }
  }

  /**
   * Sets the playing state directly and updates controls, but does not trigger
   * the timer-based 'tick()' method. Useful for renderers that implement their
   * own playback logic (e.g., audio-driven).
   * @param playing The new playing state.
   */
  public setPlayingState(playing: boolean) {
    if (this.playing === playing) return;
    this.playing = playing;
    this.renderControls();
  }

  /**
   * Sets the playback speed modifier and notifies parent/game-specific renderers.
   * @param speed The speed multiplier (e.g., 0.5, 1, 1.5, 2)
   * @param fromExternal If true, skip notifying parent (to avoid echo loops)
   */
  private setSpeed(speed: number, fromExternal = false) {
    if (this.speedModifier === speed) return;
    this.speedModifier = speed;

    // Update the speed selector UI
    this.speedSelector.value = speed.toString();

    // Dispatch event for game-specific renderers (werewolf, etc.)
    window.dispatchEvent(new CustomEvent('replayer-speed', { detail: { rate: speed, fromReplayer: true } }));

    // Notify parent if this change originated from our UI
    if (!fromExternal) {
      this.notifyParent({ speed });
    }
  }

  /**
   * Handles speed changes from game-specific renderers.
   */
  private handleReplayerSpeed = (event: Event) => {
    const customEvent = event as CustomEvent<{ rate: number; fromReplayer?: boolean }>;
    // Avoid infinite loop: only process if not from us
    if (customEvent.detail.fromReplayer) return;

    const newSpeed = customEvent.detail.rate;
    if (this.speedModifier !== newSpeed) {
      this.speedModifier = newSpeed;
      this.speedSelector.value = newSpeed.toString();
      this.notifyParent({ speed: newSpeed });
    }
  };

  private tick = () => {
    // When externally controlled, parent drives all timing - don't run our own loop
    if (this.externallyControlled) return;
    if (!this.playing || !this.replay) return;

    if (this.step >= this.replay.steps.length - 1) {
      this.playing = false;
      this.renderControls();
      return;
    }

    // Calculate dynamic step duration based on content
    const currentStep = this.replay.steps[this.step];
    const gameName = this.replay.name ?? '';
    const stepDuration = getGameStepRenderTime(currentStep, gameName, this.replayMode, this.speedModifier);

    setTimeout(() => {
      this.setStep(this.step + 1);
      this.tick();
    }, stepDuration);
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

  private renderLegend() {
    if (!this.showLegend || !this.agents || this.agents.length === 0) {
      this.legend.style.display = 'none';
      return;
    }
    this.legend.style.display = 'flex';
    this.legend.innerHTML = ''; // Clear previous content

    // Logic from player.html to group agents
    const groupIntoSets = (arr: any[], num: number) => {
      const sets: any[][] = [];
      arr.forEach((a) => {
        if (sets.length === 0 || sets[sets.length - 1].length === num) {
          sets.push([]);
        }
        sets[sets.length - 1].push(a);
      });
      return sets;
    };

    const sortedAgents = [...this.agents];
    if (typeof sortedAgents[0]?.index === 'number') {
      sortedAgents.sort((a, b) => a.index - b.index);
    }

    const agentPairs = groupIntoSets(sortedAgents, 2);

    agentPairs.forEach((agentList) => {
      const ul = document.createElement('ul');
      agentList.forEach((agent) => {
        const li = document.createElement('li');
        if (agent.id) {
          li.title = `id: ${agent.id}`;
        }
        li.style.color = agent.color || '#FFF';

        if (agent.image) {
          const img = document.createElement('img');
          img.src = agent.image;
          li.appendChild(img);
        }

        const span = document.createElement('span');
        span.textContent = agent.name;
        li.appendChild(span);

        ul.appendChild(li);
      });
      this.legend.appendChild(ul);
    });
  }

  /**
   * Public method to clean up all side effects (listeners, loops)
   * when the instance is about to be destroyed.
   * This is called by the factory during an HMR update.
   */
  public cleanup() {
    // 1. Stop any running loops (setTimeout)
    //    This is critical to prevent old ticks from running.
    //    Skip notification since we're being destroyed.
    this.pause(false);

    // 2. Remove global event listeners
    //    This is the most common source of HMR-related memory leaks.
    window.removeEventListener('message', this.handleMessage);
    window.removeEventListener('keydown', this.handleKeyDown);
    window.removeEventListener('replayer-speed', this.handleReplayerSpeed);

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
