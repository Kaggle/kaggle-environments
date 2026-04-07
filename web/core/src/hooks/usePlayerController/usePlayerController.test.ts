// jsdom v26+ depends on @asamuzakjp/css-color which uses top-level await in ESM,
// but jsdom loads it via require(), causing ERR_REQUIRE_ASYNC_MODULE on Node 22.
// Pinned to jsdom 25 until upstream fixes the CJS/ESM compat issue.
// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePlayerController, UsePlayerControllerOptions } from './usePlayerController';

const defaultOptions: UsePlayerControllerOptions = {
  totalSteps: 5,
  getStepDuration: () => 100,
};

const renderController = (overrides: Partial<UsePlayerControllerOptions> = {}) =>
  renderHook((props: UsePlayerControllerOptions) => usePlayerController(props), {
    initialProps: { ...defaultOptions, ...overrides },
  });

const getState = (result: ReturnType<typeof renderController>['result']) => result.current[0];
const getActions = (result: ReturnType<typeof renderController>['result']) => result.current[1];
const getParentData = (result: ReturnType<typeof renderController>['result']) => result.current[2];

describe('initial state', () => {
  it('uses sensible defaults', () => {
    const { result } = renderController();
    expect(getState(result)).toEqual({ step: 0, playing: false, speed: 1, replayMode: 'condensed' });
  });

  it('accepts initial overrides', () => {
    const { result } = renderController({
      initial: { step: 2, playing: true, speed: 1.5, replayMode: 'zen' },
    });
    const state = getState(result);
    expect(state.step).toBe(2);
    expect(state.speed).toBe(1.5);
    expect(state.replayMode).toBe('zen');
  });

  it('returns default parent data', () => {
    const { result } = renderController();
    expect(getParentData(result)).toEqual({ parentHandlesUi: false });
  });
});

describe('setStep', () => {
  it('sets the step and pauses playback', () => {
    const { result } = renderController({ initial: { playing: true } });
    act(() => getActions(result).setStep(3));
    expect(getState(result).step).toBe(3);
    expect(getState(result).playing).toBe(false);
  });

  it('clamps step to valid range (lower bound)', () => {
    const { result } = renderController();
    act(() => getActions(result).setStep(-5));
    expect(getState(result).step).toBe(0);
  });

  it('clamps step to valid range (upper bound)', () => {
    const { result } = renderController({ totalSteps: 5 });
    act(() => getActions(result).setStep(99));
    expect(getState(result).step).toBe(4);
  });
});

describe('setStepOnly', () => {
  it('sets the step without changing playing state', () => {
    const { result } = renderController({ initial: { playing: true } });
    act(() => getActions(result).setStepOnly(2));
    expect(getState(result).step).toBe(2);
    expect(getState(result).playing).toBe(true);
  });

  it('clamps the step value', () => {
    const { result } = renderController({ totalSteps: 3 });
    act(() => getActions(result).setStepOnly(100));
    expect(getState(result).step).toBe(2);
  });
});

describe('stepForward / stepBackward', () => {
  it('advances one step and pauses', () => {
    const { result } = renderController({ initial: { step: 1 } });
    act(() => getActions(result).stepForward());
    expect(getState(result).step).toBe(2);
    expect(getState(result).playing).toBe(false);
  });

  it('goes back one step and pauses', () => {
    const { result } = renderController({ initial: { step: 2 } });
    act(() => getActions(result).stepBackward());
    expect(getState(result).step).toBe(1);
    expect(getState(result).playing).toBe(false);
  });

  it('does not go below step 0', () => {
    const { result } = renderController({ initial: { step: 0 } });
    act(() => getActions(result).stepBackward());
    expect(getState(result).step).toBe(0);
  });

  it('does not go past last step', () => {
    const { result } = renderController({ totalSteps: 3, initial: { step: 2 } });
    act(() => getActions(result).stepForward());
    expect(getState(result).step).toBe(2);
  });
});

describe('setSpeed', () => {
  it('updates the speed', () => {
    const { result } = renderController();
    act(() => getActions(result).setSpeed(2));
    expect(getState(result).speed).toBe(2);
  });
});

describe('setReplayMode', () => {
  it('updates the replay mode', () => {
    const { result } = renderController();
    act(() => getActions(result).setReplayMode('zen'));
    expect(getState(result).replayMode).toBe('zen');
  });
});

describe('play / pause / toggle', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('play sets playing to true', () => {
    const { result } = renderController();
    act(() => getActions(result).play());
    expect(getState(result).playing).toBe(true);
  });

  it('pause sets playing to false', () => {
    const { result } = renderController({ initial: { playing: true } });
    act(() => getActions(result).pause());
    expect(getState(result).playing).toBe(false);
  });

  it('toggle switches between play and pause', () => {
    const { result } = renderController();
    act(() => getActions(result).toggle());
    expect(getState(result).playing).toBe(true);
    act(() => getActions(result).toggle());
    expect(getState(result).playing).toBe(false);
  });

  it('play restarts from 0 when at the last step', () => {
    const { result } = renderController({ totalSteps: 3, initial: { step: 2 } });
    act(() => getActions(result).play());
    expect(getState(result).step).toBe(0);
    expect(getState(result).playing).toBe(true);
  });
});

describe('restart', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('resets to step 0 and starts playing', () => {
    const { result } = renderController({ initial: { step: 3 } });
    act(() => getActions(result).restart());
    expect(getState(result).step).toBe(0);
    expect(getState(result).playing).toBe(true);
  });
});

describe('setPlayingState', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('sets playing to true without scheduling playback', () => {
    const getStepDuration = vi.fn(() => 100);
    const { result } = renderController({ getStepDuration });

    act(() => getActions(result).setPlayingState(true));
    expect(getState(result).playing).toBe(true);

    act(() => vi.advanceTimersByTime(500));
    expect(getState(result).step).toBe(0);
  });

  it('sets playing to false and clears timeouts', () => {
    const { result } = renderController({ initial: { playing: true } });
    act(() => getActions(result).setPlayingState(false));
    expect(getState(result).playing).toBe(false);
  });
});

describe('playback scheduling', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('advances steps on timer ticks', () => {
    const { result } = renderController({ totalSteps: 4, getStepDuration: () => 100 });
    act(() => getActions(result).play());
    expect(getState(result).step).toBe(0);

    act(() => vi.advanceTimersByTime(100));
    expect(getState(result).step).toBe(1);

    act(() => vi.advanceTimersByTime(100));
    expect(getState(result).step).toBe(2);
  });

  it('stops playing when reaching the last step', () => {
    const { result } = renderController({ totalSteps: 3, getStepDuration: () => 50 });
    act(() => getActions(result).play());

    act(() => vi.advanceTimersByTime(50));
    expect(getState(result).step).toBe(1);

    act(() => vi.advanceTimersByTime(50));
    expect(getState(result).step).toBe(2);
    expect(getState(result).playing).toBe(false);
  });

  it('uses getStepDuration for timing', () => {
    const getStepDuration = vi.fn((step: number) => (step === 0 ? 200 : 50));
    const { result } = renderController({ totalSteps: 3, getStepDuration });

    act(() => getActions(result).play());

    act(() => vi.advanceTimersByTime(100));
    expect(getState(result).step).toBe(0);

    act(() => vi.advanceTimersByTime(100));
    expect(getState(result).step).toBe(1);
    expect(getStepDuration).toHaveBeenCalledWith(0, 'condensed', 1);
  });

  it('pause stops the playback timer', () => {
    const { result } = renderController({ totalSteps: 10, getStepDuration: () => 100 });
    act(() => getActions(result).play());

    act(() => vi.advanceTimersByTime(100));
    expect(getState(result).step).toBe(1);

    act(() => getActions(result).pause());
    act(() => vi.advanceTimersByTime(500));
    expect(getState(result).step).toBe(1);
  });
});

describe('onChange callback', () => {
  it('fires when step changes', () => {
    const onChange = vi.fn();
    const { result } = renderController({ onChange });

    act(() => getActions(result).setStep(2));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ step: 2 }), 'step');
  });

  it('fires when speed changes', () => {
    const onChange = vi.fn();
    const { result } = renderController({ onChange });

    act(() => getActions(result).setSpeed(2));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ speed: 2 }), 'speed');
  });

  it('fires when replayMode changes', () => {
    const onChange = vi.fn();
    const { result } = renderController({ onChange });

    act(() => getActions(result).setReplayMode('zen'));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ replayMode: 'zen' }), 'replayMode');
  });

  it('does not fire when setting the same value', () => {
    const onChange = vi.fn();
    const { result } = renderController({ onChange, initial: { step: 0 } });

    act(() => getActions(result).setStep(0));
    expect(onChange).not.toHaveBeenCalledWith(expect.anything(), 'step');
  });
});

describe('parent messaging (outgoing)', () => {
  const postMessageSpy = vi.fn();
  let originalParent: typeof window.parent;

  beforeEach(() => {
    postMessageSpy.mockClear();
    originalParent = window.parent;
    Object.defineProperty(window, 'parent', {
      value: { postMessage: postMessageSpy },
      writable: true,
      configurable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, 'parent', {
      value: originalParent,
      writable: true,
      configurable: true,
    });
  });

  it('sends step updates to parent frame', () => {
    const { result } = renderController();
    act(() => getActions(result).setStep(2));
    expect(postMessageSpy).toHaveBeenCalledWith(expect.objectContaining({ step: 2 }), '*');
  });

  it('sends speed updates to parent frame', () => {
    const { result } = renderController();
    act(() => getActions(result).setSpeed(1.5));
    expect(postMessageSpy).toHaveBeenCalledWith(expect.objectContaining({ speed: 1.5 }), '*');
  });
});

describe('parent messaging (incoming)', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('handles step message from parent', () => {
    const { result } = renderController({ totalSteps: 10 });
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { step: 3 } }));
    });
    expect(getState(result).step).toBe(3);
  });

  it('handles playing message from parent', () => {
    const { result } = renderController();
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { playing: true } }));
    });
    expect(getState(result).playing).toBe(true);
  });

  it('handles speed message from parent', () => {
    const { result } = renderController();
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { speed: 0.5 } }));
    });
    expect(getState(result).speed).toBe(0.5);
  });

  it('handles replayMode message from parent', () => {
    const { result } = renderController();
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { replayMode: 'zen' } }));
    });
    expect(getState(result).replayMode).toBe('zen');
  });

  it('handles replay data from parent', () => {
    const { result } = renderController();
    const replay = { name: 'test', version: '1', steps: [], configuration: {} };
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));
    });
    expect(getParentData(result).replay).toEqual(replay);
  });

  it('handles agents data from parent', () => {
    const { result } = renderController();
    const agents = [{ name: 'Agent1' }];
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { agents } }));
    });
    expect(getParentData(result).agents).toEqual(agents);
  });

  it('ignores non-object messages', () => {
    const { result } = renderController();
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: 'not-an-object' }));
    });
    expect(getState(result).step).toBe(0);
  });

  it('ignores null messages', () => {
    const { result } = renderController();
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: null }));
    });
    expect(getState(result).step).toBe(0);
  });

  it('clamps incoming step to valid range', () => {
    const { result } = renderController({ totalSteps: 3 });
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { step: 99 } }));
    });
    expect(getState(result).step).toBe(2);
  });

  it('merges environment data into existing replay', () => {
    const { result } = renderController();
    const replay = { name: 'test', version: '1', steps: [], configuration: {} };
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));
    });

    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { environment: { info: { extra: true } } } }));
    });
    expect(getParentData(result).replay).toEqual(expect.objectContaining({ name: 'test', info: { extra: true } }));
  });

  it('handles parentHandlesUi flag from parent', () => {
    const { result } = renderController();
    act(() => {
      window.dispatchEvent(new MessageEvent('message', { data: { parentHandlesUi: true } }));
    });
    expect(getParentData(result).parentHandlesUi).toBe(true);
  });
});

describe('step clamping on totalSteps change', () => {
  it('clamps step when totalSteps shrinks below current step', () => {
    const { result, rerender } = renderController({ totalSteps: 10, initial: { step: 8 } });
    expect(getState(result).step).toBe(8);

    rerender({ ...defaultOptions, totalSteps: 5, initial: { step: 8 } });
    expect(getState(result).step).toBe(4);
  });
});
