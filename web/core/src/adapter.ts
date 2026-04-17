import { BaseGameStep, ReplayData } from './types';

/**
 * GameAdapter is the interface for game visualizer adapters.
 *
 * Adapters bridge the gap between the ReplayVisualizer shell and the
 * actual game rendering logic. The primary implementation is ReplayAdapter,
 * which uses EpisodePlayer for all UI and playback management.
 */
export interface GameAdapter<TSteps = BaseGameStep[]> {
  /**
   * Mount the adapter to a container element.
   * Called once when data is first available.
   */
  mount(container: HTMLElement, initialData: ReplayData<TSteps>): void;

  /**
   * Render the game at the given step.
   * Called when replay data or agents are updated.
   * Note: EpisodePlayer manages step state internally, so the step parameter
   * is primarily for initial data loading.
   */
  render(step: number, replay: ReplayData<TSteps>, agents: any[]): void;

  /**
   * Unmount the adapter and clean up resources.
   */
  unmount(): void;
}
