import { GameAdapter } from './adapter';
import { ReplayData } from './types';
import { render } from 'preact';

// The legacy renderer function signature
type LegacyRenderer = (options: any, container?: HTMLElement) => void;

// const parent = window.parent

// function handleSetCurrentStep(step: number) {
//   window.parent.postMessage(
//     {
//       step,
//     },
//     '*'
//   );
// }

// function handleSetPlaying(playing: boolean) {
//   window.parent.postMessage(
//     {
//       playing,
//     },
//     '*'
//   );
// }

interface UnstableReplayerControls {
  setStep: (newStep: number) => void;
  play: (continuing?: boolean) => void;
  pause: () => void;
  setPlaying: (playing: boolean) => void;
  // Expose current step and playing state for renderer to read
  step: number;
  playing: boolean;
  // Expose the actual player object for advanced scenarios if needed
  _replayerInstance: any;
}

export interface RenderOptions {
  // For chess/poker
  parent: HTMLElement;
  steps: any[];
  playerNames: string[];

  // For werewolf and others
  replay: ReplayData;
  agents: any;

  // Common properties
  step?: number;
  width: number;
  height: number;

  unstable_replayerControls?: UnstableReplayerControls;

  // For message passing to an outer frame
  // setCurrentStep: (currentStep: number) => void;
  // setPlaying: (playing: boolean) => void;
}

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
    const unstable_replayerControls: UnstableReplayerControls | undefined = replayerInstance
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

    const renderOptions: RenderOptions = {
      parent: this.container,
      steps: replay.steps,
      playerNames: replay.info?.TeamNames || agents.map((a) => a.name),
      replay: replay,
      agents: agents,
      step: step,
      width: this.container.clientWidth,
      height: this.container.clientHeight,
      unstable_replayerControls: unstable_replayerControls,
      // setCurrentStep: handleSetCurrentStep,
      // setPlaying: handleSetPlaying,
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
