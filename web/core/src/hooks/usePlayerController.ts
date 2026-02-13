import { useReducer, useCallback, useRef, useEffect, useLayoutEffect } from 'react';
import { ReplayData, ReplayMode } from '../types';

export interface PlayerState {
  step: number;
  playing: boolean;
  speed: number;
  replayMode: ReplayMode;
}

export interface ParentData {
  replay?: ReplayData;
  agents?: any[];
  parentHandlesUi: boolean;
}

type Action =
  | { type: 'SET_STEP'; step: number }
  | { type: 'SET_PLAYING'; playing: boolean }
  | { type: 'SET_SPEED'; speed: number }
  | { type: 'SET_REPLAY_MODE'; mode: ReplayMode };

function reducer(state: PlayerState, action: Action): PlayerState {
  switch (action.type) {
    case 'SET_STEP':
      return state.step === action.step ? state : { ...state, step: action.step };
    case 'SET_PLAYING':
      return state.playing === action.playing ? state : { ...state, playing: action.playing };
    case 'SET_SPEED':
      return state.speed === action.speed ? state : { ...state, speed: action.speed };
    case 'SET_REPLAY_MODE':
      return state.replayMode === action.mode ? state : { ...state, replayMode: action.mode };
    default:
      return state;
  }
}

export interface PlayerActions {
  play: () => void;
  pause: () => void;
  toggle: () => void;
  setStep: (step: number) => void;
  setSpeed: (speed: number) => void;
  setReplayMode: (mode: ReplayMode) => void;
  stepForward: () => void;
  stepBackward: () => void;
  restart: () => void;
}

export interface UsePlayerControllerOptions {
  totalSteps: number;
  getStepDuration: (step: number, mode: ReplayMode, speed: number) => number;
  initial?: Partial<PlayerState>;
  onChange?: (state: PlayerState, changed: keyof PlayerState) => void;
}

export function usePlayerController(options: UsePlayerControllerOptions): [PlayerState, PlayerActions, ParentData] {
  const { totalSteps, getStepDuration, initial, onChange } = options;

  const [state, dispatch] = useReducer(reducer, {
    step: initial?.step ?? 0,
    playing: initial?.playing ?? false,
    speed: initial?.speed ?? 1,
    replayMode: initial?.replayMode ?? 'condensed',
  });

  // Parent data received via postMessage
  const [parentData, setParentData] = useReducer(
    (s: ParentData, updates: Partial<ParentData>) => ({ ...s, ...updates }),
    { parentHandlesUi: false }
  );
  const parentDataRef = useRef(parentData);
  const stateRef = useRef(state);
  const onChangeRef = useRef(onChange);
  const getStepDurationRef = useRef(getStepDuration);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useLayoutEffect(() => {
    parentDataRef.current = parentData;
    stateRef.current = state;
    onChangeRef.current = onChange;
    getStepDurationRef.current = getStepDuration;
  });

  // Notify parent (if in iframe)
  const notifyParent = useCallback((updates: Partial<PlayerState>) => {
    if (window.parent !== window) {
      window.parent.postMessage(updates, '*');
    }
  }, []);

  // Wrapped dispatch that calls onChange and notifies parent
  const dispatchWithNotify = useCallback(
    (action: Action) => {
      dispatch(action);

      // Determine what changed and notify
      const prev = stateRef.current;
      const next = reducer(prev, action);

      if (next !== prev) {
        const actionToKey: Record<Action['type'], keyof PlayerState> = {
          SET_STEP: 'step',
          SET_PLAYING: 'playing',
          SET_SPEED: 'speed',
          SET_REPLAY_MODE: 'replayMode',
        };
        onChangeRef.current?.(next, actionToKey[action.type]);

        // Notify parent of changes
        const updates: Partial<PlayerState> = {};
        if (next.step !== prev.step) updates.step = next.step;
        if (next.playing !== prev.playing) updates.playing = next.playing;
        if (next.speed !== prev.speed) updates.speed = next.speed;
        if (next.replayMode !== prev.replayMode) updates.replayMode = next.replayMode;
        if (Object.keys(updates).length > 0) {
          notifyParent(updates);
        }
      }
    },
    [notifyParent]
  );

  // --- Playback scheduling ---

  const clearPlaybackTimeout = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const scheduleNextStepRef =
    useRef<(currentStep: number, currentSpeed: number, currentReplayMode: ReplayMode) => void>();

  const scheduleNextStep = useCallback(
    (currentStep: number, currentSpeed: number, currentReplayMode: ReplayMode) => {
      if (currentStep >= totalSteps - 1) {
        dispatchWithNotify({ type: 'SET_PLAYING', playing: false });
        return;
      }

      const duration = getStepDurationRef.current(currentStep, currentReplayMode, currentSpeed);

      timeoutRef.current = setTimeout(() => {
        const nextStep = currentStep + 1;
        dispatchWithNotify({ type: 'SET_STEP', step: nextStep });
        scheduleNextStepRef.current?.(nextStep, currentSpeed, currentReplayMode);
      }, duration);
    },
    [totalSteps, dispatchWithNotify]
  );

  useLayoutEffect(() => {
    scheduleNextStepRef.current = scheduleNextStep;
  }, [scheduleNextStep]);

  // Cleanup on unmount
  useEffect(() => {
    return () => clearPlaybackTimeout();
  }, [clearPlaybackTimeout]);

  // --- Actions ---

  const pause = useCallback(() => {
    clearPlaybackTimeout();
    dispatchWithNotify({ type: 'SET_PLAYING', playing: false });
  }, [clearPlaybackTimeout, dispatchWithNotify]);

  const play = useCallback(() => {
    clearPlaybackTimeout();

    let startStep = stateRef.current.step;
    if (startStep >= totalSteps - 1) {
      startStep = 0;
      dispatchWithNotify({ type: 'SET_STEP', step: 0 });
    }

    dispatchWithNotify({ type: 'SET_PLAYING', playing: true });
    scheduleNextStep(startStep, stateRef.current.speed, stateRef.current.replayMode);
  }, [totalSteps, clearPlaybackTimeout, dispatchWithNotify, scheduleNextStep]);

  const toggle = useCallback(() => {
    if (stateRef.current.playing) {
      pause();
    } else {
      play();
    }
  }, [play, pause]);

  const setStep = useCallback(
    (step: number) => {
      pause();
      dispatchWithNotify({ type: 'SET_STEP', step });
    },
    [pause, dispatchWithNotify]
  );

  const setSpeed = useCallback(
    (speed: number) => {
      dispatchWithNotify({ type: 'SET_SPEED', speed });
    },
    [dispatchWithNotify]
  );

  const setReplayMode = useCallback(
    (mode: ReplayMode) => {
      dispatchWithNotify({ type: 'SET_REPLAY_MODE', mode });
    },
    [dispatchWithNotify]
  );

  const stepForward = useCallback(() => {
    pause();
    dispatchWithNotify({ type: 'SET_STEP', step: stateRef.current.step + 1 });
  }, [pause, dispatchWithNotify]);

  const stepBackward = useCallback(() => {
    pause();
    dispatchWithNotify({ type: 'SET_STEP', step: stateRef.current.step - 1 });
  }, [pause, dispatchWithNotify]);

  const restart = useCallback(() => {
    dispatchWithNotify({ type: 'SET_STEP', step: 0 });
    play();
  }, [dispatchWithNotify, play]);

  // --- Parent messaging (receive) ---

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const data = event.data;
      if (!data || typeof data !== 'object') return;

      // Handle playback state updates from parent
      if (typeof data.step === 'number') {
        pause();
        dispatchWithNotify({ type: 'SET_STEP', step: data.step });
      }
      if (typeof data.playing === 'boolean') {
        if (data.playing) {
          play();
        } else {
          pause();
        }
      }
      if (typeof data.speed === 'number') {
        dispatchWithNotify({ type: 'SET_SPEED', speed: data.speed });
      }
      if (data.replayMode) {
        dispatchWithNotify({ type: 'SET_REPLAY_MODE', mode: data.replayMode });
      }

      // Handle data from parent
      if (data.replay) {
        setParentData({ replay: data.replay });
      }
      if (data.environment) {
        // Merge environment data into existing replay
        const currentReplay = parentDataRef.current.replay ?? {};
        setParentData({
          replay: { ...currentReplay, ...data.environment } as ReplayData,
        });
      }
      if (data.agents) {
        setParentData({ agents: data.agents });
      }
      if (typeof data.parentHandlesUi === 'boolean') {
        setParentData({ parentHandlesUi: data.parentHandlesUi ?? false });
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [pause, play, dispatchWithNotify]);

  // --- Clamp step if totalSteps changes ---

  useEffect(() => {
    if (state.step >= totalSteps && totalSteps > 0) {
      dispatch({ type: 'SET_STEP', step: totalSteps - 1 });
    }
  }, [totalSteps, state.step]);

  const actions: PlayerActions = {
    play,
    pause,
    toggle,
    setStep,
    setSpeed,
    setReplayMode,
    stepForward,
    stepBackward,
    restart,
  };

  return [state, actions, parentData];
}
