import { GameAdapter } from './adapter';
import { ReplayData } from './types';
import { render } from 'preact';

// The legacy renderer function signature
type LegacyRenderer = (options: any, container?: HTMLElement) => void;

export class LegacyAdapter implements GameAdapter {
  private container: HTMLElement | null = null;
  private renderer: LegacyRenderer;
  private isInitialRender = true;

  constructor(renderer: LegacyRenderer) {
    this.renderer = renderer;
  }

  mount(container: HTMLElement): void {
    this.container = container;
  }

  // replayerInstance passing is a bit of a hack for werewolf - would be nice to eliminate it
  render(step: number, replay: ReplayData, agents: any[], replayerInstance?: any): void {
    if (!this.container) return;

    // Clear container only on the first render pass.
    if (this.isInitialRender) {
      this.container.innerHTML = '';
      this.isInitialRender = false;
    }

    // Generally it would be better to not do this - but werewolf needs some inner frame control
    // that conforms to this interface - marking as unstable for now to see if we can figure out
    // a better long-term solution here
    const unstable_replayerControls = replayerInstance
      ? {
          setStep: (newStep: number) => replayerInstance.setStep(newStep),
          play: (continuing?: boolean) => replayerInstance.play(continuing),
          pause: () => replayerInstance.pause(),
          setPlaying: (playing: boolean) => replayerInstance.setPlayingState(playing),
          // Expose current step and playing state for renderer to read
          step: replayerInstance.step,
          playing: replayerInstance.playing,
          // Expose the actual player object for advanced scenarios if needed
          _replayerInstance: replayerInstance,
        }
      : undefined;

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

      unstable_replayerControls: unstable_replayerControls,
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
