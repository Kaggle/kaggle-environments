import { GameAdapter } from './adapter';
import { BaseGameStep, ReplayData } from './types';
import { render } from 'preact';

// The legacy renderer function signature
export type LegacyRenderer<TSteps = BaseGameStep[]> = (
  options: LegacyRendererOptions<TSteps>,
  container?: HTMLElement
) => void;

export interface LegacyRendererOptions<TSteps = BaseGameStep[]> {
  parent: HTMLElement;
  steps: TSteps;
  playerNames: string[];
  replay: ReplayData<TSteps>;
  agents: any[];
  step: number;
  width: number;
  height: number;
  setCurrentStep: (step: number) => void;
  setPlaying: (playing?: boolean) => void;
  // Generally not recommended: setAgents is a bit of a hack is for older renderers that need to update
  // the visualizer's state (e.g. agents for the legend).
  setAgents?: (agents: any[]) => void;
  unstable_replayerControls?: {
    step: number;
    setStep: (step: number) => void;
    play: (continuing?: boolean) => void;
    pause: () => void;
    setPlaying: (playing: boolean) => void;
    [key: string]: any;
  };
}

export class LegacyAdapter<TSteps = BaseGameStep[]> implements GameAdapter<TSteps> {
  private container: HTMLElement | null = null;
  private renderer: LegacyRenderer<TSteps>;
  private isInitialRender = true;

  constructor(renderer: LegacyRenderer<TSteps>) {
    this.renderer = renderer;
  }

  mount(container: HTMLElement): void {
    this.container = container;
  }

  // replayerInstance passing is a bit of a hack for werewolf - would be nice to eliminate it
  render(step: number, replay: ReplayData<TSteps>, agents: any[], replayerInstance?: any): void {
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
          setStep: (newStep: number) => {
            replayerInstance.setStep(newStep);
            // Also notify parent frame of step change
            window.parent.postMessage({ step: newStep }, '*');
          },
          play: (continuing?: boolean) => replayerInstance.play(continuing),
          pause: () => replayerInstance.pause(),
          setPlaying: (playing: boolean) => {
            replayerInstance.setPlayingState(playing);
            // Also notify parent frame of playing state change
            window.parent.postMessage({ playing }, '*');
          },
          // Expose current step and playing state for renderer to read
          step: replayerInstance.step,
          playing: replayerInstance.playing,
          // Expose the actual player object for advanced scenarios if needed
          _replayerInstance: replayerInstance,
        }
      : undefined;

    const setAgents = (agents: any[]) => {
      if (replayerInstance) {
        replayerInstance.setAgents(agents);
      }
    };

    const renderOptions: LegacyRendererOptions<TSteps> = {
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

      // For message passing to an outer frame
      setCurrentStep: (step: number) =>
        window.parent.postMessage(
          {
            step,
          },
          '*'
        ),
      setPlaying: (playing?: boolean) => {
        window.parent.postMessage(
          {
            playing,
          },
          '*'
        );
      },
      setAgents: setAgents,
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
