import * as React from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { styled } from '@mui/material/styles';
import { usePlayerController } from '../hooks/usePlayerController/usePlayerController';
import { BaseGameStep, InterestingEvent, ReplayData, ReplayMode } from '../types';
import {
  getInterestingEvents,
  getGameStepRenderTime,
  getGameStepLabel,
  getGameStepDescription,
  processEpisodeData,
} from '../transformers/transformers';
import { ReasoningLogs } from '../ReasoningLogs';
import { PlaybackControls } from './PlaybackControls';
import { Button, css, Icon, useMediaQuery } from '@mui/material';

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
  /** Game-specific step label function for ReasoningLogs. Falls back to default if not provided. */
  getStepLabel?: (step: BaseGameStep) => string;
  /** Game-specific step description function for ReasoningLogs. Falls back to default if not provided. */
  getStepDescription?: (step: BaseGameStep) => string;
  /** Game-specific step render time function. Falls back to default if not provided. */
  getStepRenderTime?: (step: BaseGameStep, replayMode: ReplayMode, speedModifier: number) => number;
  /** Game-specific interesting events function. Falls back to default if not provided. */
  getInterestingEvents?: (steps: BaseGameStep[]) => InterestingEvent[];
  /** Game-specific token render distribution for streaming text. Falls back to default if not provided. */
  getTokenRenderDistribution?: (chunkCount: number) => number[];
  /** Whether to use a compact/dense layout for playback controls */
  dense?: boolean;
}

export interface GameRendererProps<TSteps extends BaseGameStep[] = BaseGameStep[]> {
  replay: ReplayData<TSteps>;
  step: number;
  agents: any[];
  /** Callback to set the current step */
  onSetStep?: (step: number) => void;
  /** Callback to set playing state (true = playing, false = paused) */
  onSetPlaying?: (playing: boolean) => void;
  /** Callback to register playback handlers (for renderers that need to intercept play/pause) */
  onRegisterPlaybackHandlers?: (handlers: { onPlay?: () => boolean | void; onPause?: () => void }) => void;
  /** Callback to announce a message to screen readers via the aria-live region */
  onAnnounce?: (message: string) => void;
}

const PlayerContainer = styled('div')<{ $uiMode?: UiMode; $dense: boolean }>`
  display: flex;
  flex-direction: ${({ $uiMode }) => ($uiMode === 'inline' ? 'column' : 'row')};
  width: 100%;
  height: 100%;
  overflow: hidden;

  ${({ theme }) => `${theme.breakpoints.down('tablet')} and (orientation: portrait)`} {
    flex-direction: column;
    ${(p) => p.$dense && 'max-height: 500px;'}
  }
`;

const VisualizerContainer = styled('div')<{ $dense?: boolean }>`
  flex: 1;
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  ${({ $dense }) =>
    $dense &&
    css`
      max-height: 500px;
    `}
`;

const ReasoningLogsContainer = styled('div')<{ $dense: boolean }>`
  flex: 0 0 25%;
  min-width: 330px;
  height: 100%;
  overflow: hidden;

  ${({ theme }) => `${theme.breakpoints.down('tablet')} and (orientation: portrait)`} {
    flex: none;
    width: 100%;
    height: ${(p) => (p.$dense ? 'min-content' : '40%')};
  }
`;

const InlineControlsContainer = styled('div')`
  width: 100%;
  padding: 8px;
  background-color: ${({ theme }) => theme.palette.background.paper};
  flex-shrink: 0;
  position: relative;
  z-index: 10;
`;

const LoadingContainer = styled('div')`
  display: flex;
  width: 100%;
  height: 100%;
  overflow: hidden;
  justify-content: center;
  align-items: center;
`;

const GameLogButton = styled(Button)`
  margin: 8px 24px 0;
  position: fixed;
  bottom: 24px;
  z-index: 1;

  ${({ theme }) => `${theme.breakpoints.down('tablet')} and (orientation: portrait)`} {
    margin: 12px;
  }
`;

const VisuallyHidden = styled('div')`
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
`;

export function EpisodePlayer<TSteps extends BaseGameStep[] = BaseGameStep[]>({
  replay: rawReplay,
  agents = [],
  gameName,
  GameRenderer,
  ui = 'side-panel',
  initialStep = 0,
  initialSpeed = 1,
  initialReplayMode = 'condensed',
  onStepChange,
  onPlayingChange,
  onSpeedChange,
  style,
  className,
  skipTransform = false,
  getStepLabel,
  getStepDescription,
  getStepRenderTime: getStepRenderTimeProp,
  getInterestingEvents: getInterestingEventsProp,
  getTokenRenderDistribution,
  dense = false,
}: EpisodePlayerProps<TSteps>) {
  const [replay, setReplay] = useState<ReplayData<TSteps> | undefined>(rawReplay);
  const [currentAgents, setCurrentAgents] = useState<any[]>(agents);
  const [showLogs, setShowLogs] = useState(ui === 'side-panel');
  const [liveAnnouncement, setLiveAnnouncement] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  // For landscape orientation, we keep using the desktop layout since there is less need to be vertical friendly
  const useVerticalLayout = useMediaQuery((theme) => `${theme.breakpoints.down('tablet')} and (orientation: portrait)`);

  // Refs for custom playback handlers registered by renderers (e.g., for audio-driven playback)
  const playbackHandlersRef = useRef<{ onPlay?: () => void; onPause?: () => void }>({});

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
      if (getStepRenderTimeProp) {
        return getStepRenderTimeProp(step, mode, speed);
      }
      return getGameStepRenderTime(step, gameName, mode, speed);
    },
    [processedReplay, gameName, getStepRenderTimeProp]
  );

  const [state, actions, parentData] = usePlayerController({
    totalSteps,
    getStepDuration,
    initial: {
      step: initialStep,
      playing: true,
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
    if (parentData.replay && !skipTransform) {
      setReplay(parentData.replay as ReplayData<TSteps>);
    }
  }, [parentData.replay, skipTransform]);

  useEffect(() => {
    if (parentData.agents && !skipTransform) {
      setCurrentAgents(parentData.agents);
    }
  }, [parentData.agents, skipTransform]);

  // Calculate interesting events for the slider
  const interestingEvents = useMemo<InterestingEvent[]>(() => {
    if (!processedReplay) return [];
    if (getInterestingEventsProp) {
      return getInterestingEventsProp(processedReplay.steps);
    }
    return getInterestingEvents(processedReplay.steps, gameName);
  }, [processedReplay, gameName, getInterestingEventsProp]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'BUTTON' ||
        target.tagName === 'A' ||
        target.tagName === 'DIALOG'
      ) {
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

  // Build aria-live announcement when the step changes
  useEffect(() => {
    if (!processedReplay || state.step >= processedReplay.steps.length) {
      return;
    }
    const step = processedReplay.steps[state.step];
    const player = step.players?.find((p) => p.isTurn);
    const label = getStepLabel ? getStepLabel(step) : getGameStepLabel(step, gameName);
    const streaming = state.replayMode !== 'condensed';
    let announcement = player?.name ? `${player.name}: ${label}` : label;
    if (streaming) {
      const description = getStepDescription ? getStepDescription(step) : getGameStepDescription(step, gameName);
      if (description) {
        announcement += '. ' + description;
      }
    }
    setLiveAnnouncement(announcement);
  }, [state.step, state.replayMode, processedReplay, gameName, getStepLabel, getStepDescription]);

  useEffect(() => {
    if (rawReplay) {
      setReplay(rawReplay);
    }
  }, [rawReplay]);

  useEffect(() => {
    setCurrentAgents(agents);
  }, [agents]);

  // Callback for renderers to register custom playback handlers
  const handleRegisterPlaybackHandlers = useCallback((handlers: { onPlay?: () => void; onPause?: () => void }) => {
    playbackHandlersRef.current = handlers;
  }, []);

  const handleAnnounce = useCallback((message: string) => {
    setLiveAnnouncement(message);
  }, []);

  const handlePlayChange = useCallback(
    (playing?: boolean) => {
      const handlers = playbackHandlersRef.current;
      if (playing === undefined) {
        // Toggle: check current state
        if (state.playing) {
          handlers.onPause?.();
          actions.pause();
        } else {
          // Call handler if registered; if it returns true, it handled playback
          const handled = handlers.onPlay?.();
          if (!handled) {
            actions.play();
          }
        }
      } else if (playing) {
        // Call handler if registered; if it returns true, it handled playback
        const handled = handlers.onPlay?.();
        if (!handled) {
          actions.play();
        }
      } else {
        handlers.onPause?.();
        actions.pause();
      }
    },
    [actions, state.playing]
  );

  const handleClosePanel = useCallback(() => {
    setShowLogs(false);
  }, []);

  if (!processedReplay) {
    return (
      <LoadingContainer ref={containerRef} className={className} style={style}>
        <div>Loading...</div>
      </LoadingContainer>
    );
  }

  return (
    <PlayerContainer ref={containerRef} className={className} style={style} $uiMode={ui} $dense={dense}>
      <VisuallyHidden aria-live="polite" role="status">
        {liveAnnouncement}
      </VisuallyHidden>
      <VisualizerContainer $dense={dense}>
        <GameRenderer
          replay={processedReplay}
          step={state.step}
          agents={currentAgents}
          onSetStep={actions.setStepOnly}
          onSetPlaying={actions.setPlayingState}
          onRegisterPlaybackHandlers={handleRegisterPlaybackHandlers}
          onAnnounce={handleAnnounce}
        />
      </VisualizerContainer>

      {/* Inline mode: PlaybackControls below the game (hidden if parent handles UI) */}
      {ui === 'inline' && !parentData.parentHandlesUi && (
        <InlineControlsContainer>
          <PlaybackControls
            playing={state.playing}
            currentStep={state.step}
            totalSteps={totalSteps}
            speedModifier={state.speed}
            onPlayChange={handlePlayChange}
            onStepChange={actions.setStep}
            onSpeedChange={actions.setSpeed}
          />
        </InlineControlsContainer>
      )}

      {/* Side-panel mode: ReasoningLogs with controls (hidden if parent handles UI) */}
      {ui === 'side-panel' &&
        !parentData.parentHandlesUi &&
        (showLogs ? (
          <ReasoningLogsContainer $dense={dense}>
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
              getStepLabel={getStepLabel}
              getStepDescription={getStepDescription}
              getTokenRenderDistribution={getTokenRenderDistribution}
              dense={dense}
            />
          </ReasoningLogsContainer>
        ) : (
          <GameLogButton
            variant="high"
            onClick={() => setShowLogs(true)}
            startIcon={<Icon>{useVerticalLayout ? 'bottom_panel_open' : 'left_panel_open'}</Icon>}
          >
            Game Log
          </GameLogButton>
        ))}

      {/* 'none' mode: No UI rendered */}
    </PlayerContainer>
  );
}
