import { describe, it, expect } from 'vitest';
import {
  processEpisodeData,
  getGameStepLabel,
  getGameStepDescription,
  getGameStepRenderTime,
  getInterestingEvents,
  getTokenRenderDistribution,
} from './transformers';
import { TIME_PER_CHUNK } from './timing';
import { BaseGameStep, ReplayData } from './types';

const makeStep = (overrides: Partial<BaseGameStep> = {}): BaseGameStep => ({
  step: 1,
  players: [
    { id: 0, name: 'Alice', thumbnail: '', isTurn: true, actionDisplayText: 'plays X', thoughts: 'I think...' },
    { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
  ],
  ...overrides,
});

const makeReplay = (overrides: Partial<ReplayData> = {}): ReplayData => ({
  name: 'test-game',
  version: '1.0',
  steps: [makeStep()],
  configuration: {},
  ...overrides,
});

describe('processEpisodeData', () => {
  it('returns already-transformed data as-is', () => {
    const data = makeReplay({ isTransformed: true });
    const result = processEpisodeData(data, 'test-game');
    expect(result).toBe(data);
  });

  it('returns untransformed data through (no-op default transformer)', () => {
    const data = makeReplay({ isTransformed: false });
    const result = processEpisodeData(data, 'test-game');
    expect(result).toBe(data);
  });

  it('handles replay with no isTransformed flag', () => {
    const data = makeReplay();
    delete data.isTransformed;
    const result = processEpisodeData(data, 'test-game');
    expect(result).toBe(data);
  });
});

describe('getGameStepLabel', () => {
  it('returns actionDisplayText of the active player', () => {
    const step = makeStep();
    expect(getGameStepLabel(step, 'test-game')).toBe('plays X');
  });

  it('returns empty string when no players', () => {
    const step = { step: 1 } as BaseGameStep;
    expect(getGameStepLabel(step, 'test-game')).toBe('');
  });

  it('returns empty string when no player has isTurn', () => {
    const step = makeStep({
      players: [
        { id: 0, name: 'Alice', thumbnail: '', isTurn: false },
        { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
      ],
    });
    expect(getGameStepLabel(step, 'test-game')).toBe('');
  });

  it('returns empty string when active player has no actionDisplayText', () => {
    const step = makeStep({
      players: [{ id: 0, name: 'Alice', thumbnail: '', isTurn: true }],
    });
    expect(getGameStepLabel(step, 'test-game')).toBe('');
  });
});

describe('getGameStepDescription', () => {
  it('returns thoughts of the active player', () => {
    const step = makeStep();
    expect(getGameStepDescription(step, 'test-game')).toBe('I think...');
  });

  it('returns empty string when no players', () => {
    const step = { step: 1 } as BaseGameStep;
    expect(getGameStepDescription(step, 'test-game')).toBe('');
  });

  it('returns empty string when active player has no thoughts', () => {
    const step = makeStep({
      players: [{ id: 0, name: 'Alice', thumbnail: '', isTurn: true }],
    });
    expect(getGameStepDescription(step, 'test-game')).toBe('');
  });
});

describe('getGameStepRenderTime', () => {
  it('delegates to defaultGetStepRenderTime with correct args', () => {
    const step = makeStep();
    // condensed mode, 1x speed → default 2200
    expect(getGameStepRenderTime(step, 'test-game', 'condensed', 1)).toBe(2200);
  });

  it('passes through custom defaultDuration', () => {
    const step = makeStep();
    expect(getGameStepRenderTime(step, 'test-game', 'condensed', 1, 5000)).toBe(5000);
  });

  it('uses thought-based timing in streaming modes', () => {
    const step = makeStep({
      players: [
        { id: 0, name: 'Alice', thumbnail: '', isTurn: true, thoughts: 'word1 word2 word3' },
        { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
      ],
    });
    // 3 words * TIME_PER_CHUNK
    expect(getGameStepRenderTime(step, 'test-game', 'zen', 1)).toBe(3 * TIME_PER_CHUNK);
  });
});

describe('getInterestingEvents', () => {
  it('returns an empty array (default implementation)', () => {
    const steps = [makeStep(), makeStep({ step: 2 })];
    expect(getInterestingEvents(steps, 'test-game')).toEqual([]);
  });
});

describe('getTokenRenderDistribution', () => {
  it('returns uniform distribution (delegates to generateDefaultDelayDistribution)', () => {
    const result = getTokenRenderDistribution(4, 'test-game');
    expect(result).toEqual(Array(4).fill(TIME_PER_CHUNK));
  });

  it('returns empty array for 0 chunks', () => {
    expect(getTokenRenderDistribution(0, 'test-game')).toEqual([]);
  });
});
