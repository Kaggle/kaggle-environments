import { useState, useCallback, useRef, useEffect } from 'react';
import { ReplayMode } from '../types';
import { getGameStepRenderTime } from '../transformers';

export interface PlaybackState {
  step: number;
  playing: boolean;
  speed: number;
  replayMode: ReplayMode;
  totalSteps: number;
}

export interface PlaybackActions {
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  setStep: (step: number) => void;
  setSpeed: (speed: number) => void;
  setReplayMode: (mode: ReplayMode) => void;
  stepForward: () => void;
  stepBackward: () => void;
  restart: () => void;
}

export interface UsePlaybackStateOptions {
  totalSteps: number;
  gameName: string;
  initialStep?: number;
  initialPlaying?: boolean;
  initialSpeed?: number;
  initialReplayMode?: ReplayMode;
  onStepChange?: (step: number) => void;
  onPlayingChange?: (playing: boolean) => void;
  onSpeedChange?: (speed: number) => void;
  onReplayModeChange?: (mode: ReplayMode) => void;
  getStepDuration?: (step: number, replayMode: ReplayMode, speed: number) => number;
}

export interface UsePlaybackStateReturn extends PlaybackState, PlaybackActions {}

export function usePlaybackState(options: UsePlaybackStateOptions): UsePlaybackStateReturn {
  const {
    totalSteps,
    gameName,
    initialStep = 0,
    initialPlaying = false,
    initialSpeed = 1,
    initialReplayMode = 'condensed',
    onStepChange,
    onPlayingChange,
    onSpeedChange,
    onReplayModeChange,
    getStepDuration,
  } = options;

  const [step, setStepInternal] = useState(initialStep);
  const [playing, setPlayingInternal] = useState(initialPlaying);
  const [speed, setSpeedInternal] = useState(initialSpeed);
  const [replayMode, setReplayModeInternal] = useState<ReplayMode>(initialReplayMode);

  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const setStep = useCallback(
    (newStep: number) => {
      const clampedStep = Math.max(0, Math.min(totalSteps - 1, newStep));
      setStepInternal(clampedStep);
      onStepChange?.(clampedStep);
    },
    [totalSteps, onStepChange]
  );

  const setSpeed = useCallback(
    (newSpeed: number) => {
      setSpeedInternal(newSpeed);
      onSpeedChange?.(newSpeed);
    },
    [onSpeedChange]
  );

  const setReplayMode = useCallback(
    (mode: ReplayMode) => {
      setReplayModeInternal(mode);
      onReplayModeChange?.(mode);
    },
    [onReplayModeChange]
  );

  const pause = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setPlayingInternal(false);
    onPlayingChange?.(false);
  }, [onPlayingChange]);

  const scheduleNextStep = useCallback(
    (currentStep: number, currentSpeed: number, currentReplayMode: ReplayMode) => {
      if (currentStep >= totalSteps - 1) {
        setPlayingInternal(false);
        onPlayingChange?.(false);
        return;
      }

      // Default step duration calculation
      const duration = getStepDuration
        ? getStepDuration(currentStep, currentReplayMode, currentSpeed)
        : getGameStepRenderTime({ step: currentStep, players: [] }, gameName, currentReplayMode, currentSpeed);

      timeoutRef.current = setTimeout(() => {
        const nextStep = currentStep + 1;
        setStepInternal(nextStep);
        onStepChange?.(nextStep);
        scheduleNextStep(nextStep, currentSpeed, currentReplayMode);
      }, duration);
    },
    [totalSteps, gameName, getStepDuration, onStepChange, onPlayingChange]
  );

  const play = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // If at the end, restart from beginning
    let startStep = step;
    if (step >= totalSteps - 1) {
      startStep = 0;
      setStepInternal(0);
      onStepChange?.(0);
    }

    setPlayingInternal(true);
    onPlayingChange?.(true);
    scheduleNextStep(startStep, speed, replayMode);
  }, [step, totalSteps, speed, replayMode, scheduleNextStep, onStepChange, onPlayingChange]);

  const togglePlayPause = useCallback(() => {
    if (playing) {
      pause();
    } else {
      play();
    }
  }, [playing, play, pause]);

  const stepForward = useCallback(() => {
    pause();
    setStep(step + 1);
  }, [pause, setStep, step]);

  const stepBackward = useCallback(() => {
    pause();
    setStep(step - 1);
  }, [pause, setStep, step]);

  const restart = useCallback(() => {
    setStep(0);
    play();
  }, [setStep, play]);

  // Update totalSteps if it changes and step is out of bounds
  useEffect(() => {
    if (step >= totalSteps && totalSteps > 0) {
      setStepInternal(totalSteps - 1);
    }
  }, [totalSteps, step]);

  return {
    // State
    step,
    playing,
    speed,
    replayMode,
    totalSteps,

    // Actions
    play,
    pause,
    togglePlayPause,
    setStep,
    setSpeed,
    setReplayMode,
    stepForward,
    stepBackward,
    restart,
  };
}
