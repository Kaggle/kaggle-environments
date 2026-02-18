import * as React from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { usePlayerController } from '../hooks/usePlayerController';
import { BaseGameStep, InterestingEvent, ReplayData, ReplayMode } from '../types';
import { getInterestingEvents, getGameStepRenderTime, processEpisodeData } from '../transformers';
import { ReasoningLogs } from '../ReasoningLogs';
import { PlaybackControls } from './PlaybackControls';

/**
 * UI mode for playback controls and ReasoningLogs.
 * - 'inline': Classic inline controls below the game (no ReasoningLogs)
 * - 'side-panel': Full experience with ReasoningLogs and controls in side panel
 * - 'none': No UI (for externally-driven playback)
 */
export type UiMode = 'inline' | 'side-panel' | 'none';

export interface EpisodePlayerProps<TSteps extends BaseGameStep[] = BaseGameStep[]> {
  /** The replay data to visualize */
  replay?: ReplayData<TSteps>;
  /** Agent data for display */
  agents?: any[];
  /** The game name (e.g., 'werewolf', 'open_spiel_chess') */
  gameName: string;
  /** The game renderer component */
  GameRenderer: React.ComponentType<GameRendererProps<TSteps>>;
  /**
   * UI mode for controls and ReasoningLogs:
   * - 'inline': Classic inline controls below the game (no ReasoningLogs)
   * - 'side-panel': Full experience with ReasoningLogs and controls in side panel
   * - 'none': No UI (for externally-driven playback)
   * @default 'side-panel'
   */
  ui?: UiMode;
  /** Layout mode for side-panel: 'side-by-side' puts logs to the right, 'stacked' puts logs below */
  layout?: 'side-by-side' | 'stacked';
  /** Initial step to start at */
  initialStep?: number;
  /** Initial playing state */
  initialPlaying?: boolean;
  /** Initial playback speed */
  initialSpeed?: number;
  /** Initial replay mode */
  initialReplayMode?: ReplayMode;
  /** Callback when step changes */
  onStepChange?: (step: number) => void;
  /** Callback when playing state changes */
  onPlayingChange?: (playing: boolean) => void;
  /** Callback when speed changes */
  onSpeedChange?: (speed: number) => void;
  /** Callback when theme changes (received from parent via postMessage) */
  onThemeChange?: (theme: 'dark' | 'light') => void;
  /** Container style */
  style?: React.CSSProperties;
  /** Container class name */
  className?: string;
  /**
   * If true, skip internal transformation (replay is already transformed).
   * Used by ReplayAdapter which handles transformation itself.
   */
  skipTransform?: boolean;
}

export interface GameRendererProps<TSteps extends BaseGameStep[] = BaseGameStep[]> {
  replay: ReplayData<TSteps>;
  step: number;
  agents: any[];
}

const containerStyles: React.CSSProperties = {
  display: 'flex',
  width: '100%',
  height: '100%',
  overflow: 'hidden',
};

const visualizerContainerStyles: React.CSSProperties = {
  flex: 1,
  minWidth: 0,
  minHeight: 0,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
};

const reasoningLogsContainerStyles: React.CSSProperties = {
  width: '330px',
  flexShrink: 0,
  height: '100%',
  overflow: 'hidden',
};

const inlineControlsContainerStyles: React.CSSProperties = {
  width: '100%',
  padding: '8px',
  backgroundColor: '#1a1a1a',
  flexShrink: 0,
  position: 'relative',
  zIndex: 10,
};

export function EpisodePlayer<TSteps extends BaseGameStep[] = BaseGameStep[]>({
  replay: rawReplay,
  agents = [],
  gameName,
  GameRenderer,
  ui = 'side-panel',
  layout = 'side-by-side',
  initialStep = 0,
  initialPlaying = false,
  initialSpeed = 1,
  initialReplayMode = 'condensed',
  onStepChange,
  onPlayingChange,
  onSpeedChange,
  onThemeChange,
  style,
  className,
  skipTransform = false,
}: EpisodePlayerProps<TSteps>) {
  const [replay, setReplay] = useState<ReplayData<TSteps> | undefined>(rawReplay);
  const [currentAgents, setCurrentAgents] = useState<any[]>(agents);
  const [showLogs, setShowLogs] = useState(ui === 'side-panel');
  const containerRef = useRef<HTMLDivElement>(null);

  // Process replay data through transformer (skip if already transformed)
  const processedReplay = useMemo(() => {
    if (!replay) return undefined;
    if (skipTransform) return replay;
    return processEpisodeData(replay, gameName) as ReplayData<TSteps>;
  }, [replay, gameName, skipTransform]);

  const totalSteps = processedReplay?.steps.length ?? 0;

  // Calculate step duration based on game content
  const getStepDuration = useCallback(
    (stepIndex: number, mode: ReplayMode, speed: number) => {
      if (!processedReplay || stepIndex >= processedReplay.steps.length) {
        return 2200 / speed;
      }
      const step = processedReplay.steps[stepIndex];
      return getGameStepRenderTime(step, gameName, mode, speed);
    },
    [processedReplay, gameName]
  );

  const [state, actions, parentData] = usePlayerController({
    totalSteps,
    getStepDuration,
    initial: {
      step: initialStep,
      playing: initialPlaying,
      speed: initialSpeed,
      replayMode: initialReplayMode,
    },
    onChange: (newState, changed) => {
      if (changed === 'step') onStepChange?.(newState.step);
      if (changed === 'playing') onPlayingChange?.(newState.playing);
      if (changed === 'speed') onSpeedChange?.(newState.speed);
    },
  });

  useEffect(() => {
    if (parentData.replay) {
      setReplay(parentData.replay as ReplayData<TSteps>);
    }
  }, [parentData.replay]);

  useEffect(() => {
    if (parentData.agents) {
      setCurrentAgents(parentData.agents);
    }
  }, [parentData.agents]);

  useEffect(() => {
    console.log('[EpisodePlayer] parentData.theme changed:', parentData.theme);
    if (parentData.theme) {
      console.log('[EpisodePlayer] Calling onThemeChange with:', parentData.theme);
      onThemeChange?.(parentData.theme);
    }
  }, [parentData.theme, onThemeChange]);

  // Calculate interesting events for the slider
  const interestingEvents = useMemo<InterestingEvent[]>(() => {
    if (!processedReplay) return [];
    return getInterestingEvents(processedReplay.steps, gameName);
  }, [processedReplay, gameName]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return;
      }

      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault();
          actions.stepBackward();
          break;
        case 'ArrowRight':
          event.preventDefault();
          actions.stepForward();
          break;
        case ' ':
        case 'Enter':
          event.preventDefault();
          actions.toggle();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [actions]);

  useEffect(() => {
    if (rawReplay) {
      setReplay(rawReplay);
    }
  }, [rawReplay]);

  useEffect(() => {
    setCurrentAgents(agents);
  }, [agents]);

  const handlePlayChange = useCallback(
    (playing?: boolean) => {
      if (playing === undefined) {
        actions.toggle();
      } else if (playing) {
        actions.play();
      } else {
        actions.pause();
      }
    },
    [actions]
  );

  const handleClosePanel = useCallback(() => {
    setShowLogs(false);
  }, []);

  if (!processedReplay) {
    return (
      <div
        ref={containerRef}
        className={className}
        style={{ ...containerStyles, ...style, justifyContent: 'center', alignItems: 'center' }}
      >
        <div>Loading...</div>
      </div>
    );
  }

  // For side-panel mode, use row/column layout. For inline, always column (game above controls).
  const flexDirection = ui === 'side-panel' && layout !== 'stacked' ? 'row' : 'column';

  return (
    <div ref={containerRef} className={className} style={{ ...containerStyles, flexDirection, ...style }}>
      <div style={visualizerContainerStyles}>
        <GameRenderer replay={processedReplay} step={state.step} agents={currentAgents} />
      </div>

      {/* Inline mode: PlaybackControls below the game (hidden if parent handles UI) */}
      {ui === 'inline' && !parentData.parentHandlesUi && (
        <div style={inlineControlsContainerStyles}>
          <PlaybackControls
            playing={state.playing}
            currentStep={state.step}
            totalSteps={totalSteps}
            speedModifier={state.speed}
            onPlayChange={handlePlayChange}
            onStepChange={actions.setStep}
          />
        </div>
      )}

      {/* Side-panel mode: ReasoningLogs with controls (hidden if parent handles UI) */}
      {ui === 'side-panel' && showLogs && !parentData.parentHandlesUi && (
        <div style={layout === 'stacked' ? { width: '100%', height: '300px' } : reasoningLogsContainerStyles}>
          <ReasoningLogs
            closePanel={handleClosePanel}
            onPlayChange={handlePlayChange}
            onSpeedChange={actions.setSpeed}
            onStepChange={actions.setStep}
            playing={state.playing}
            replayMode={state.replayMode}
            setReplayMode={actions.setReplayMode}
            speedModifier={state.speed}
            totalSteps={totalSteps}
            steps={processedReplay.steps}
            currentStep={state.step}
            gameName={gameName}
            interestingEvents={interestingEvents}
          />
        </div>
      )}

      {/* 'none' mode: No UI rendered */}
    </div>
  );
}
