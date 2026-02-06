import * as React from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { usePlaybackState } from '../hooks/usePlaybackState';
import { useParentMessaging } from '../hooks/useParentMessaging';
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
  width: '400px',
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
  style,
  className,
  skipTransform = false,
}: EpisodePlayerProps<TSteps>) {
  const [replay, setReplay] = useState<ReplayData<TSteps> | undefined>(rawReplay);
  const [currentAgents, setCurrentAgents] = useState<any[]>(agents);
  const [showLogs, setShowLogs] = useState(ui === 'side-panel');
  const [replayMode, setReplayMode] = useState<ReplayMode>(initialReplayMode);
  // Track if parent is handling UI - if true, hide our UI regardless of `ui` prop
  const [parentHandlesUi, setParentHandlesUi] = useState(false);

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

  // Playback state management
  const playback = usePlaybackState({
    totalSteps,
    gameName,
    initialStep,
    initialPlaying,
    initialSpeed,
    initialReplayMode,
    onStepChange,
    onPlayingChange,
    onSpeedChange,
    getStepDuration,
  });

  // Calculate interesting events for the slider
  const interestingEvents = useMemo<InterestingEvent[]>(() => {
    if (!processedReplay) return [];
    return getInterestingEvents(processedReplay.steps, gameName);
  }, [processedReplay, gameName]);

  // Parent messaging for iframe communication
  const { notifyParent } = useParentMessaging({
    onStepChange: (step) => {
      playback.pause();
      playback.setStep(step);
    },
    onPlayingChange: (playing) => {
      if (playing) {
        playback.play();
      } else {
        playback.pause();
      }
    },
    onSpeedChange: playback.setSpeed,
    onReplayModeChange: setReplayMode,
    onReplayData: (data) => setReplay(data as ReplayData<TSteps>),
    onAgentsData: setCurrentAgents,
    // Handle 'environment' data from parent (alternative to 'replay')
    onEnvironmentData: (env) => {
      setReplay((prev) => {
        const base = prev ?? {
          name: 'unknown',
          version: 'unknown',
          steps: [] as unknown as TSteps,
          configuration: {},
          info: {},
        };
        return { ...base, ...env } as ReplayData<TSteps>;
      });
    },
    onControlsChange: () => {}, // Controls visibility handled via ui prop
    onLegendChange: () => {}, // Legend not currently used
    onParentHandlesUi: setParentHandlesUi,
  });

  // Notify parent of state changes
  useEffect(() => {
    notifyParent({ step: playback.step });
  }, [playback.step, notifyParent]);

  useEffect(() => {
    notifyParent({ playing: playback.playing });
  }, [playback.playing, notifyParent]);

  useEffect(() => {
    notifyParent({ speed: playback.speed });
  }, [playback.speed, notifyParent]);

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
          playback.stepBackward();
          break;
        case 'ArrowRight':
          event.preventDefault();
          playback.stepForward();
          break;
        case ' ':
        case 'Enter':
          event.preventDefault();
          playback.togglePlayPause();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [playback]);

  // Update replay when prop changes
  useEffect(() => {
    if (rawReplay) {
      setReplay(rawReplay);
    }
  }, [rawReplay]);

  // Update agents when prop changes
  useEffect(() => {
    setCurrentAgents(agents);
  }, [agents]);

  const handlePlayChange = useCallback(
    (playing?: boolean) => {
      if (playing === undefined) {
        playback.togglePlayPause();
      } else if (playing) {
        playback.play();
      } else {
        playback.pause();
      }
    },
    [playback]
  );

  const handleClosePanel = useCallback(() => {
    setShowLogs(false);
  }, []);

  const handleReplayModeChange = useCallback(
    (mode: ReplayMode) => {
      setReplayMode(mode);
      playback.setReplayMode(mode);
    },
    [playback]
  );

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
        <GameRenderer replay={processedReplay} step={playback.step} agents={currentAgents} />
      </div>

      {/* Inline mode: PlaybackControls below the game (hidden if parent handles UI) */}
      {ui === 'inline' && !parentHandlesUi && (
        <div style={inlineControlsContainerStyles}>
          <PlaybackControls
            playing={playback.playing}
            currentStep={playback.step}
            totalSteps={totalSteps}
            speedModifier={playback.speed}
            replayMode={replayMode}
            onPlayChange={handlePlayChange}
            onStepChange={playback.setStep}
            onSpeedChange={playback.setSpeed}
            onReplayModeChange={handleReplayModeChange}
            interestingEvents={interestingEvents}
          />
        </div>
      )}

      {/* Side-panel mode: ReasoningLogs with controls (hidden if parent handles UI) */}
      {ui === 'side-panel' && showLogs && !parentHandlesUi && (
        <div style={layout === 'stacked' ? { width: '100%', height: '300px' } : reasoningLogsContainerStyles}>
          <ReasoningLogs
            closePanel={handleClosePanel}
            onPlayChange={handlePlayChange}
            onSpeedChange={playback.setSpeed}
            onStepChange={playback.setStep}
            playing={playback.playing}
            replayMode={replayMode}
            setReplayMode={handleReplayModeChange}
            speedModifier={playback.speed}
            totalSteps={totalSteps}
            steps={processedReplay.steps}
            currentStep={playback.step}
            gameName={gameName}
            interestingEvents={interestingEvents}
          />
        </div>
      )}

      {/* 'none' mode: No UI rendered */}
    </div>
  );
}
