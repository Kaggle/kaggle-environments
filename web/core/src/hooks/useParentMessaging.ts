import { useEffect, useCallback, useRef } from 'react';
import { ReplayData, ReplayMode } from '../types';

export interface ParentMessage {
  // Control messages
  step?: number;
  playing?: boolean;
  speed?: number;
  replayMode?: ReplayMode;
  controls?: boolean;
  legend?: boolean;

  // UI coordination - parent tells iframe whether to show its UI
  parentHandlesUi?: boolean;

  // Data messages
  replay?: ReplayData;
  environment?: Partial<ReplayData>;
  agents?: any[];
  setSteps?: any[];
}

export interface UseParentMessagingOptions {
  onStepChange?: (step: number) => void;
  onPlayingChange?: (playing: boolean) => void;
  onSpeedChange?: (speed: number) => void;
  onReplayModeChange?: (mode: ReplayMode) => void;
  onControlsChange?: (show: boolean) => void;
  onLegendChange?: (show: boolean) => void;
  onReplayData?: (replay: ReplayData) => void;
  onAgentsData?: (agents: any[]) => void;
  onEnvironmentData?: (environment: Partial<ReplayData>) => void;
  onSetSteps?: (steps: any[]) => void;
  /** Called when parent signals whether it handles UI (ReasoningLogs/controls) */
  onParentHandlesUi?: (parentHandlesUi: boolean) => void;
}

export interface UseParentMessagingReturn {
  notifyParent: (state: Partial<ParentMessage>) => void;
}

export function useParentMessaging(options: UseParentMessagingOptions): UseParentMessagingReturn {
  // Options are accessed through callbacksRef to avoid stale closures

  // Use refs to avoid stale closures in the message handler
  const callbacksRef = useRef(options);
  callbacksRef.current = options;

  const handleMessage = useCallback((event: MessageEvent) => {
    const data = event.data as ParentMessage;
    if (!data) return;

    const callbacks = callbacksRef.current;

    // Handle control messages
    if (typeof data.step === 'number' && callbacks.onStepChange) {
      callbacks.onStepChange(data.step);
    }

    if (typeof data.playing === 'boolean' && callbacks.onPlayingChange) {
      callbacks.onPlayingChange(data.playing);
    }

    if (typeof data.speed === 'number' && callbacks.onSpeedChange) {
      callbacks.onSpeedChange(data.speed);
    }

    if (data.replayMode && callbacks.onReplayModeChange) {
      callbacks.onReplayModeChange(data.replayMode);
    }

    if (typeof data.controls === 'boolean' && callbacks.onControlsChange) {
      callbacks.onControlsChange(data.controls);
    }

    if (typeof data.legend === 'boolean' && callbacks.onLegendChange) {
      callbacks.onLegendChange(data.legend);
    }

    if (typeof data.parentHandlesUi === 'boolean' && callbacks.onParentHandlesUi) {
      console.log('[useParentMessaging] Received parentHandlesUi:', data.parentHandlesUi);
      callbacks.onParentHandlesUi(data.parentHandlesUi);
    }

    // Handle data messages
    if (data.replay && callbacks.onReplayData) {
      callbacks.onReplayData(data.replay);
    }

    if (data.environment && callbacks.onEnvironmentData) {
      callbacks.onEnvironmentData(data.environment);
    }

    if (data.agents && callbacks.onAgentsData) {
      callbacks.onAgentsData(data.agents);
    }

    if (data.setSteps && callbacks.onSetSteps) {
      callbacks.onSetSteps(data.setSteps);
    }
  }, []);

  useEffect(() => {
    window.addEventListener('message', handleMessage);
    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, [handleMessage]);

  const notifyParent = useCallback((state: Partial<ParentMessage>) => {
    // Only post to parent if we're actually in an iframe
    // This prevents echo loops when window.parent === window
    if (window.parent !== window) {
      window.parent.postMessage(state, '*');
    }
  }, []);

  return { notifyParent };
}
