import * as React from 'react';
import { useRef, useEffect } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { GameAdapter } from './adapter';
import { BaseGameStep, ReplayData } from './types';
import { EpisodePlayer, GameRendererProps, UiMode } from './components/EpisodePlayer';
import { theme } from './theme';
import { processEpisodeData } from './transformers';

// Re-export UiMode for consumers
export type { UiMode } from './components/EpisodePlayer';

/** Transformer function type for processing replay data */
export type ReplayTransformer<TSteps = BaseGameStep[]> = (replay: ReplayData, gameName: string) => ReplayData<TSteps>;

/**
 * Options passed to legacy renderer functions.
 * This interface is used by LegacyRendererWrapper to call existing game renderers.
 */
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
}

/**
 * Dynamically injects Material Symbols font from Google Fonts CDN.
 * Material Symbols is the newer icon set that includes icons like
 * bottom_panel_close, sensors, expand_all, etc.
 * Only called once, caches the result.
 */
/**
 * Injects CSS to isolate game renderers from CssBaseline's global resets.
 * This resets box-sizing and other properties that CssBaseline sets globally.
 */
let gameIsolationStylesInjected = false;
function injectGameIsolationStyles(): void {
  if (gameIsolationStylesInjected) return;
  if (typeof document === 'undefined') return;

  const style = document.createElement('style');
  style.textContent = `
    .game-renderer-isolation,
    .game-renderer-isolation *,
    .game-renderer-isolation *::before,
    .game-renderer-isolation *::after {
      box-sizing: content-box;
    }
  `;
  document.head.appendChild(style);
  gameIsolationStylesInjected = true;
}

let iconFontInjected = false;
function injectMaterialIconsFont(): void {
  if (iconFontInjected) return;
  if (typeof document === 'undefined') return;

  // Check if already present
  const existingLink = document.querySelector('link[href*="Material+Symbols"]');
  if (existingLink) {
    iconFontInjected = true;
    return;
  }

  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = 'https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0';
  document.head.appendChild(link);

  // MUI's Icon component expects the 'material-icons' class, but Material Symbols
  // uses 'material-symbols-outlined'. Add a style to make them compatible.
  const style = document.createElement('style');
  style.textContent = `.material-icons { font-family: 'Material Symbols Outlined'; }`;
  document.head.appendChild(style);

  iconFontInjected = true;
}

/**
 * Props passed to a custom UI component when using `ui: CustomComponent`.
 * This allows full customization of playback controls and logs display.
 */
export interface PlaybackUiProps {
  /** Close/hide the UI panel */
  closePanel: () => void;
  /** Toggle or set playback state */
  onPlayChange: (playing?: boolean) => void;
  /** Set playback speed */
  onSpeedChange: (speed: number) => void;
  /** Jump to a specific step */
  onStepChange: (step: number) => void;
  /** Whether currently playing */
  playing: boolean;
  /** Current playback speed */
  speed: number;
  /** Total number of steps */
  totalSteps: number;
  /** Current step index */
  currentStep: number;
  /** The processed replay data */
  replay: ReplayData;
  /** Game name for display/timing */
  gameName: string;
}

/**
 * Legacy renderer function signature.
 * This allows existing game renderers to work without modification.
 */
export type RendererFn<TSteps = BaseGameStep[]> = (
  options: LegacyRendererOptions<TSteps>,
  container?: HTMLElement
) => void;

/**
 * Options for ReplayAdapter.
 * Provide EITHER `renderer` (legacy function) OR `GameRenderer` (React component).
 */
export interface ReplayAdapterOptions<TSteps extends BaseGameStep[] = BaseGameStep[]> {
  /** The game name for transformer/timing lookup */
  gameName: string;

  /**
   * DOM-based renderer function.
   * Use this to keep your game visualizer free of React code.
   * The adapter will wrap it internally.
   */
  renderer?: RendererFn<TSteps>;

  /**
   * React component that renders the game.
   * Use this if you want to write your renderer in React.
   */
  GameRenderer?: React.ComponentType<GameRendererProps<TSteps>>;

  /**
   * Custom transformer function to process replay data before rendering.
   * If not provided, uses the default `processEpisodeData` with gameName.
   *
   * This is useful for game-specific data transformations that will
   * eventually live in the game's own folder.
   *
   * @example
   * ```ts
   * transformer: (replay) => myGameTransformer(replay)
   * ```
   */
  transformer?: ReplayTransformer<TSteps>;

  /**
   * UI mode for playback controls and ReasoningLogs:
   * - 'inline': Classic inline controls below the game (no ReasoningLogs)
   * - 'side-panel': Full experience with ReasoningLogs and controls in side panel (default)
   * - 'none': No UI (for externally-driven playback)
   * - Custom React component: Provide your own UI component
   *
   * @default 'side-panel'
   */
  ui?: UiMode | React.ComponentType<PlaybackUiProps>;

  /** Layout mode: 'side-by-side' puts logs to the right, 'stacked' puts logs below */
  layout?: 'side-by-side' | 'stacked';

  /** Initial playback speed (default: 1) */
  initialSpeed?: number;
}

/**
 * Internal React component that wraps a legacy DOM renderer.
 * This allows legacy renderers to work within the React-based EpisodePlayer.
 */
function LegacyRendererWrapper<TSteps extends BaseGameStep[] = BaseGameStep[]>({
  renderer,
  replay,
  step,
  agents,
}: {
  renderer: RendererFn<TSteps>;
  replay: ReplayData<TSteps>;
  step: number;
  agents: any[];
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !replay) return;

    const playerNames = replay.info?.TeamNames || agents.map((a) => a?.name || 'Player');

    const options: LegacyRendererOptions<TSteps> = {
      parent: containerRef.current,
      steps: replay.steps,
      step,
      playerNames,
      replay,
      agents,
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      // These are no-ops since EpisodePlayer manages state
      setCurrentStep: () => {},
      setPlaying: () => {},
    };

    renderer(options, containerRef.current);
  }, [renderer, replay, step, agents]);

  return React.createElement('div', {
    ref: containerRef,
    className: 'game-renderer-isolation',
    style: {
      width: '100%',
      height: '100%',
      flex: 1,
      minHeight: 0,
      overflow: 'hidden',
      position: 'relative',
    },
  });
}

/**
 * ReplayAdapter is the unified adapter for game visualizers.
 *
 * It accepts EITHER a legacy renderer function OR a React component,
 * and provides configurable UI modes via the `ui` option.
 *
 * @example Full experience with side panel (default):
 * ```ts
 * new ReplayAdapter({
 *   gameName: 'chess',
 *   renderer: renderer,
 *   ui: 'side-panel', // default - shows ReasoningLogs and controls
 * })
 * ```
 *
 * @example Classic inline controls (no ReasoningLogs):
 * ```ts
 * new ReplayAdapter({
 *   gameName: 'chess',
 *   renderer: renderer,
 *   ui: 'inline',
 * })
 * ```
 *
 * @example No UI (externally-driven playback):
 * ```ts
 * new ReplayAdapter({
 *   gameName: 'chess',
 *   renderer: renderer,
 *   ui: 'none',
 * })
 * ```
 */
export class ReplayAdapter<TSteps extends BaseGameStep[] = BaseGameStep[]> implements GameAdapter<TSteps> {
  private root: Root | null = null;
  private options: ReplayAdapterOptions<TSteps>;
  private rawReplay: ReplayData | undefined;
  private transformedReplay: ReplayData<TSteps> | undefined;
  private currentAgents: any[] = [];
  private wrappedRenderer: React.ComponentType<GameRendererProps<TSteps>> | null = null;

  constructor(options: ReplayAdapterOptions<TSteps>) {
    if (!options.renderer && !options.GameRenderer) {
      throw new Error('ReplayAdapter requires either `renderer` or `GameRenderer` option');
    }

    this.options = options;

    // Wrap legacy renderer as a React component for EpisodePlayer
    if (options.renderer) {
      const legacyRenderer = options.renderer;
      this.wrappedRenderer = (props: GameRendererProps<TSteps>) =>
        React.createElement(LegacyRendererWrapper as React.ComponentType<any>, {
          renderer: legacyRenderer,
          replay: props.replay,
          step: props.step,
          agents: props.agents,
        });
    }
  }

  /**
   * Apply the transformer to the replay data.
   * Uses custom transformer if provided, otherwise falls back to processEpisodeData.
   */
  private transformReplay(replay: ReplayData): ReplayData<TSteps> {
    const { transformer, gameName } = this.options;

    if (transformer) {
      return transformer(replay, gameName);
    }

    // Default: use processEpisodeData
    return processEpisodeData(replay, gameName) as ReplayData<TSteps>;
  }

  mount(container: HTMLElement, initialData?: ReplayData<TSteps>): void {
    // Transform initial data if provided
    if (initialData) {
      this.rawReplay = initialData;
      this.transformedReplay = this.transformReplay(initialData);
    }

    const ui = this.options.ui ?? 'side-panel';

    // Always use EpisodePlayer - it handles all UI modes
    this.root = createRoot(container);

    // Notify parent of iframe capabilities so it can adjust its UI accordingly.
    // This allows the parent (e.g., EpisodesPanel) to:
    // - Hide its ReasoningLogs panel when iframe handles UI
    // - Stop sending step updates when iframe manages playback
    // - Hide its controls when iframe has its own
    if (window.parent !== window) {
      const capabilities = {
        // Whether iframe handles the UI (controls and/or ReasoningLogs).
        // True for 'inline' and 'side-panel' - parent should hide its UI.
        // False for 'none' - parent controls everything.
        iframeHandlesUi: ui !== 'none',
        // Whether iframe has its own playback controls
        hasControls: ui !== 'none',
        // Whether iframe has ReasoningLogs (only 'side-panel' mode has them)
        // When false, parent should hide its ReasoningLogs regardless of FF
        hasReasoningLogs: ui === 'side-panel',
        // Whether iframe manages its own playback tick loop
        managesPlayback: true,
      };
      console.log('[ReplayAdapter] Sending iframeCapabilities:', capabilities, 'ui mode:', ui);
      window.parent.postMessage({ iframeCapabilities: capabilities }, '*');
    }

    // Inject Material Icons font for MUI Icon components
    injectMaterialIconsFont();

    // Inject game isolation styles to protect legacy renderers from CssBaseline
    injectGameIsolationStyles();

    this.renderEpisodePlayer();
  }

  render(_step: number, replay: ReplayData<TSteps>, agents: any[]): void {
    this.currentAgents = agents;

    // Only re-transform if replay changed
    if (replay !== this.rawReplay) {
      this.rawReplay = replay;
      this.transformedReplay = this.transformReplay(replay);
    }

    // EpisodePlayer manages step internally via usePlaybackState,
    // but we re-render to pass updated replay/agents data
    this.renderEpisodePlayer();
  }

  unmount(): void {
    if (this.root) {
      this.root.unmount();
      this.root = null;
    }
  }

  /**
   * Render using EpisodePlayer which handles all UI modes.
   */
  private renderEpisodePlayer(): void {
    if (!this.root) return;

    const { gameName, GameRenderer, ui = 'side-panel', layout = 'side-by-side', initialSpeed = 1 } = this.options;

    // Use the wrapped legacy renderer or the provided React component
    const RendererComponent = this.wrappedRenderer || GameRenderer;
    if (!RendererComponent) return;

    this.root.render(
      React.createElement(
        ThemeProvider,
        { theme },
        React.createElement(
          React.Fragment,
          null,
          React.createElement(CssBaseline),
          React.createElement(EpisodePlayer, {
            // Pass already-transformed replay - EpisodePlayer should skip its own transformation
            replay: this.transformedReplay,
            agents: this.currentAgents,
            gameName,
            GameRenderer: RendererComponent as React.ComponentType<GameRendererProps<BaseGameStep[]>>,
            ui: typeof ui === 'string' ? ui : 'side-panel', // Pass ui mode (ignore custom components for now)
            layout,
            initialSpeed,
            // Signal that replay is already transformed
            skipTransform: true,
            style: { width: '100%', height: '100%' },
          })
        )
      )
    );
  }
}
