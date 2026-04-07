import { describe, it, expect } from 'vitest';
import {
  generateEaseInOutDelayDistribution,
  generateEaseInDelayDistribution,
  generateDefaultDelayDistribution,
  defaultGetStepRenderTime,
  TIME_PER_CHUNK,
} from './timing';
import { BaseGameStep } from '../types';
import { makeStep } from '../test-utils';

const sum = (arr: number[]) => arr.reduce((a, b) => a + b, 0);

describe('generateDefaultDelayDistribution', () => {
  it('returns an array of equal values', () => {
    const result = generateDefaultDelayDistribution(5);
    expect(result).toEqual(Array(5).fill(TIME_PER_CHUNK));
  });

  it('returns empty array for 0 chunks', () => {
    expect(generateDefaultDelayDistribution(0)).toEqual([]);
  });

  it('returns single value for 1 chunk', () => {
    expect(generateDefaultDelayDistribution(1)).toEqual([TIME_PER_CHUNK]);
  });
});

describe('generateEaseInOutDelayDistribution', () => {
  it('returns empty array for 0 chunks', () => {
    expect(generateEaseInOutDelayDistribution(0)).toEqual([]);
  });

  it('returns [totalTime] for 1 chunk', () => {
    expect(generateEaseInOutDelayDistribution(1)).toEqual([TIME_PER_CHUNK]);
  });

  it('preserves total time (sum equals chunkCount * TIME_PER_CHUNK)', () => {
    const count = 10;
    const result = generateEaseInOutDelayDistribution(count);
    expect(result).toHaveLength(count);
    expect(sum(result)).toBeCloseTo(count * TIME_PER_CHUNK);
  });

  it('is symmetric (mirrored pairs sum to the same value)', () => {
    const result = generateEaseInOutDelayDistribution(10);
    const pairSum = result[0] + result[result.length - 1];
    for (let i = 1; i < result.length / 2; i++) {
      expect(result[i] + result[result.length - 1 - i]).toBeCloseTo(pairSum);
    }
  });

  it('peaks in the middle', () => {
    const result = generateEaseInOutDelayDistribution(11);
    const midIndex = Math.floor(result.length / 2);
    // Middle value should be larger than the first value
    expect(result[midIndex]).toBeGreaterThan(result[0]);
  });
});

describe('generateEaseInDelayDistribution', () => {
  it('returns empty array for 0 chunks', () => {
    expect(generateEaseInDelayDistribution(0)).toEqual([]);
  });

  it('returns [totalTime] for 1 chunk', () => {
    expect(generateEaseInDelayDistribution(1)).toEqual([TIME_PER_CHUNK]);
  });

  it('preserves total time', () => {
    const count = 8;
    const result = generateEaseInDelayDistribution(count);
    expect(result).toHaveLength(count);
    expect(sum(result)).toBeCloseTo(count * TIME_PER_CHUNK);
  });

  it('starts slow and ends fast (values increase monotonically)', () => {
    const result = generateEaseInDelayDistribution(10);
    for (let i = 1; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(result[i - 1]);
    }
  });
});

describe('defaultGetStepRenderTime', () => {
  it('returns default duration at 1x speed when no thoughts and condensed mode', () => {
    const step = makeStep();
    expect(defaultGetStepRenderTime(step, 'condensed', 1)).toBe(2200);
  });

  it('respects custom defaultDuration', () => {
    const step = makeStep();
    expect(defaultGetStepRenderTime(step, 'condensed', 1, 3000)).toBe(3000);
  });

  it('scales duration inversely with speed modifier', () => {
    const step = makeStep();
    // 2x speed → half the time
    expect(defaultGetStepRenderTime(step, 'condensed', 2)).toBe(1100);
    // 0.5x speed → double the time
    expect(defaultGetStepRenderTime(step, 'condensed', 0.5)).toBe(4400);
  });

  it('uses thought-based timing in zen mode', () => {
    const step = makeStep({
      players: [
        { id: 0, name: 'Alice', thumbnail: '', isTurn: true, thoughts: 'one two three' },
        { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
      ],
    });
    // 3 words → 3 chunks → 3 * TIME_PER_CHUNK at 1x
    expect(defaultGetStepRenderTime(step, 'zen', 1)).toBe(3 * TIME_PER_CHUNK);
  });

  it('uses thought-based timing in only-stream mode', () => {
    const step = makeStep({
      players: [
        { id: 0, name: 'Alice', thumbnail: '', isTurn: true, thoughts: 'a b c d e' },
        { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
      ],
    });
    // 5 words → 5 chunks → 5 * TIME_PER_CHUNK at 1x
    expect(defaultGetStepRenderTime(step, 'only-stream', 1)).toBe(5 * TIME_PER_CHUNK);
  });

  it('ignores thoughts in condensed mode', () => {
    const step = makeStep({
      players: [
        { id: 0, name: 'Alice', thumbnail: '', isTurn: true, thoughts: 'a b c d e' },
        { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
      ],
    });
    // Should use default duration, not thought-based
    expect(defaultGetStepRenderTime(step, 'condensed', 1)).toBe(2200);
  });

  it('applies speed modifier to thought-based timing', () => {
    const step = makeStep({
      players: [
        { id: 0, name: 'Alice', thumbnail: '', isTurn: true, thoughts: 'one two three four' },
        { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
      ],
    });
    // 4 words * TIME_PER_CHUNK / 2x speed
    expect(defaultGetStepRenderTime(step, 'zen', 2)).toBe((4 * TIME_PER_CHUNK) / 2);
  });

  it('falls back to default duration when no players', () => {
    const step = { step: 1 } as BaseGameStep;
    expect(defaultGetStepRenderTime(step, 'zen', 1)).toBe(2200);
  });

  it('falls back to default duration when current player has no thoughts in streaming mode', () => {
    const step = makeStep();
    expect(defaultGetStepRenderTime(step, 'zen', 1)).toBe(2200);
  });
});
