import { GameAdapter } from './adapter';
import { BaseGameStep, ReplayData } from './types';
import './style.css';

/**
 * ReplayVisualizer is a thin shell that:
 * - Creates the container DOM element
 * - Mounts the adapter with initial data
 * - Forwards postMessage data to the adapter
 * - Handles HMR cleanup
 *
 * All UI (controls, playback, ReasoningLogs) is handled by EpisodePlayer
 * inside the adapter (ReplayAdapter).
 */
export class ReplayVisualizer<TSteps extends BaseGameStep[] = BaseGameStep[]> {
  private container: HTMLElement;
  private adapter: GameAdapter<TSteps>;
  private replay: ReplayData<TSteps> | null = null;
  private agents: any[] = [];
  private step = 0;
  private mounted = false;
  private hmrState?: any;
  private transformer?: (replay: ReplayData) => ReplayData;

  // --- Element references ---
  private viewer: HTMLElement;

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

    playerDiv.appendChild(this.viewer);
    this.container.innerHTML = '';
    this.container.appendChild(playerDiv);

    this.loadData();
  }

  private loadData() {
    // 1. (PRIORITY 1) Check for replay data *within* HMR state.
    if (this.hmrState?.replay) {
      const state = this.hmrState;
      this.setData(state.replay, state.agents);
    }

    // 2. (PRIORITY 2) No HMR replay data. Check for VITE_REPLAY_FILE in dev mode.
    else if (import.meta.env.VITE_REPLAY_FILE) {
      const replayFile = import.meta.env.VITE_REPLAY_FILE;
      fetch(replayFile)
        .then((res) => res.json())
        .then((data) => {
          // Create agents from available data if info.Agents doesn't exist
          let agents = data.info?.Agents;
          if (!agents && data.steps?.[0]) {
            // Derive agent count from first step (each player has an entry)
            const playerCount = Array.isArray(data.steps[0]) ? data.steps[0].length : 0;
            const teamNames = data.info?.TeamNames || [];
            agents = Array.from({ length: playerCount }, (_, i) => ({
              index: i,
              name: teamNames[i] || `Player ${i + 1}`,
            }));
          }
          this.setData(data, agents);
        })
        .catch((err) => {
          console.error(`Error fetching ${replayFile}:`, err);
          this.viewer.innerHTML = `<div>Error loading ${replayFile}</div>`;
        });
    } else if (import.meta.env?.DEV) {
      // Dev mode, but no HMR data and no replayFile. Wait for postMessage.
      this.viewer.innerHTML = '<div>Waiting for replay data...</div>';
    }
    // 3. (PRIORITY 3) Production build (or not DEV) and no HMR data.
    else {
      this.viewer.innerHTML = '<div>Loading...</div>';
    }

    // 4. Add listener (always)
    window.addEventListener('message', this.handleMessage);
  }

  private handleMessage = (event: MessageEvent) => {
    if (!event.data) return;

    // Helper to update HMR state
    const updateHMRState = (key: string, value: any) => {
      if (this.hmrState) {
        this.hmrState[key] = value;
      }
    };

    let needsRender = false;

    // Update agents if provided
    if (event.data.agents) {
      this.agents = event.data.agents;
      updateHMRState('agents', this.agents);
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

    // Overwrite replay object if a full 'replay' is provided
    if (event.data.replay) {
      this.replay = event.data.replay;
      needsRender = true;
    }

    // After any replay update, save it to HMR state
    if (needsRender && this.replay) {
      updateHMRState('replay', this.replay);
    }

    // Update the current step (for forwarding to adapter)
    if (typeof event.data.step === 'number') {
      this.step = event.data.step;
      updateHMRState('step', this.step);
      needsRender = true;
    }

    // If any data was updated and we have a replay object, forward to adapter
    if (needsRender && this.replay) {
      this.setData(this.replay, this.agents);
    }
  };

  private setData(replay: ReplayData<TSteps>, agents: any[] = []) {
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

    // Forward to adapter for rendering
    if (this.replay) {
      this.adapter.render(this.step, this.replay, this.agents);
    }
  }

  public setAgents(agents: any[]) {
    this.agents = agents;
    if (this.hmrState) {
      this.hmrState.agents = this.agents;
    }
  }

  /**
   * Public method to clean up all side effects (listeners, loops)
   * when the instance is about to be destroyed.
   * This is called by the factory during an HMR update.
   */
  public cleanup() {
    // Remove global event listeners
    window.removeEventListener('message', this.handleMessage);

    // Tell the adapter to unmount
    if (this.mounted) {
      this.adapter.unmount();
      this.mounted = false;
    }

    // Clear the container
    this.container.innerHTML = '';
  }
}
