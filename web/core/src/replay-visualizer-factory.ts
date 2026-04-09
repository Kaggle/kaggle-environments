import { GameAdapter } from './adapter';
import { BaseGameStep, ReplayData } from './types';
import { ReplayVisualizer } from './player';

/**
 * The shape of the persistent state for a single ReplayVisualizer instance.
 */
interface ReplayVisualizerState {
  replay: any | null;
  agents: any[];
  step: number;
  controls: boolean;
  playing: boolean;
}

// --- HMR State Management ---
let hmrReplayVisualizerStates: Map<string, ReplayVisualizerState>;
let hmrReplayVisualizerInstances: Map<string, ReplayVisualizer>;

if (import.meta.env?.DEV && import.meta.hot) {
  // 2. Initialize or retrieve the persistent state Map
  if (!import.meta.hot?.data?.replayVisualizerStates) {
    import.meta.hot.data.replayVisualizerStates = new Map<string, ReplayVisualizerState>();
  }
  hmrReplayVisualizerStates = import.meta.hot?.data?.replayVisualizerStates;

  // 3. Initialize or retrieve the instance Map (for cleanup)
  if (!import.meta.hot?.data?.replayVisualizerInstances) {
    import.meta.hot.data.replayVisualizerInstances = new Map<string, ReplayVisualizer>();
  }
  hmrReplayVisualizerInstances = import.meta.hot.data.replayVisualizerInstances;

  // Optional: Log on reload
  console.log(
    `[ReplayVisualizerFactory HMR] Reloaded. State Map has ${hmrReplayVisualizerStates.size} entries. Instance Map has ${hmrReplayVisualizerInstances.size} entries.`
  );
}

/**
 * A factory to create a new ReplayVisualizer, automatically handling
 * HMR state persistence and cleanup.
 */
export function createReplayVisualizer<TSteps extends BaseGameStep[] = BaseGameStep[]>(
  container: HTMLElement,
  adapter: GameAdapter<TSteps>,
  options: { transformer?: (replay: ReplayData) => ReplayData } = {}
): ReplayVisualizer<TSteps> {
  // --- Production Build ---
  if (!import.meta.env?.DEV || !import.meta.hot) {
    return new ReplayVisualizer(container, adapter, options);
  }

  // --- Development Build (HMR Logic) ---

  // ** REQUIREMENT CHECK **
  // We must have an ID to use as a stable key.
  if (!container.id) {
    console.error(
      'ReplayVisualizerFactory: The container element provided to createReplayVisualizer() must have an ID for HMR state persistence to work.',
      container
    );
    // Fail gracefully by returning a non-HMR-aware replayVisualizer
    return new ReplayVisualizer(container, adapter, options);
  }

  const key = container.id;

  // 1. Clean up the *old* replayVisualizer instance if one exists for this ID
  const oldInstance = hmrReplayVisualizerInstances.get(key);
  if (oldInstance) {
    console.log(`[ReplayVisualizerFactory] Cleaning up old instance for key: ${key}`);
    oldInstance.cleanup();
  }

  // 2. Get or create the persistent state for this *specific* replayVisualizer ID
  let state = hmrReplayVisualizerStates.get(key);
  if (!state) {
    console.log(`[ReplayVisualizerFactory] Creating new HMR state for key: ${key}`);
    state = {
      replay: null,
      agents: [],
      step: 0,
      controls: true,
      playing: false,
    };
    hmrReplayVisualizerStates.set(key, state);
  } else {
    console.log(`[ReplayVisualizerFactory] Re-using existing HMR state for key: ${key}`);
  }

  // 3. Create the new ReplayVisualizer, passing it its persistent state
  const newReplayVisualizer = new ReplayVisualizer(container, adapter, {
    hmrState: state,
    ...options,
  });

  // 4. Store the *new* instance for cleanup on the *next* HMR reload
  hmrReplayVisualizerInstances.set(key, newReplayVisualizer);

  return newReplayVisualizer;
}
