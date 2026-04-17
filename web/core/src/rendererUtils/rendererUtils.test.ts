import { describe, it, expect } from 'vitest';
import { getStepData } from './rendererUtils';
import { RawReplayData } from '../types';
import { makeEntry, makeRawReplay } from '../test-utils';

describe('getStepData', () => {
  it('returns step data for a valid step index', () => {
    const replay = makeRawReplay({
      steps: [
        [makeEntry({ observation: { board: [1, 2, 3] } }), makeEntry()],
        [makeEntry({ observation: { board: [4, 5, 6] } }), makeEntry()],
      ],
    });
    const result = getStepData(replay, 0);
    expect(result).not.toBeNull();
    expect(result![0].observation).toEqual({ board: [1, 2, 3] });
  });

  it('returns correct data for non-zero step index', () => {
    const replay = makeRawReplay({
      steps: [
        [makeEntry({ observation: { turn: 0 } }), makeEntry()],
        [makeEntry({ observation: { turn: 1 } }), makeEntry()],
        [makeEntry({ observation: { turn: 2 } }), makeEntry()],
      ],
    });
    const result = getStepData(replay, 2);
    expect(result).not.toBeNull();
    expect(result![0].observation).toEqual({ turn: 2 });
  });

  it('returns null for undefined replay', () => {
    expect(getStepData(undefined, 0)).toBeNull();
  });

  it('returns null when replay has no steps property', () => {
    expect(getStepData({} as any, 0)).toBeNull();
  });

  it('returns null when steps is not an array', () => {
    expect(getStepData({ steps: 'not-an-array' } as any, 0)).toBeNull();
  });

  it('returns null for negative step index', () => {
    const replay = makeRawReplay();
    expect(getStepData(replay, -1)).toBeNull();
  });

  it('returns null for step index beyond bounds', () => {
    const replay = makeRawReplay({ steps: [[makeEntry()]] });
    expect(getStepData(replay, 1)).toBeNull();
  });

  it('returns null when step data is empty array', () => {
    const replay = makeRawReplay({ steps: [[]] as any });
    expect(getStepData(replay, 0)).toBeNull();
  });

  it('returns null when step data is not an array', () => {
    const replay = makeRawReplay({ steps: [{ notArray: true }] as any });
    expect(getStepData(replay, 0)).toBeNull();
  });

  it('returns null when first entry has no observation', () => {
    const replay = makeRawReplay({
      steps: [[{ reward: 0, status: 'ACTIVE' }]] as any,
    });
    expect(getStepData(replay, 0)).toBeNull();
  });

  it('returns null when first entry is not an object', () => {
    const replay = makeRawReplay({ steps: [['string-entry']] as any });
    expect(getStepData(replay, 0)).toBeNull();
  });

  it('returns null when first entry is null', () => {
    const replay = makeRawReplay({ steps: [[null, makeEntry()]] as any });
    expect(getStepData(replay, 0)).toBeNull();
  });

  it('works with custom observation types', () => {
    interface SomeGame {
      board: number[][];
      currentPlayer: number;
    }
    const obs: SomeGame = {
      board: [
        [0, 1],
        [1, 0],
      ],
      currentPlayer: 1,
    };
    const replay: RawReplayData<SomeGame> = {
      name: 'game',
      version: '1.0',
      steps: [[{ observation: obs, reward: null, status: 'ACTIVE' }]],
      configuration: {},
    };
    const result = getStepData<SomeGame>(replay, 0);
    expect(result).not.toBeNull();
    expect(result![0].observation.currentPlayer).toBe(1);
  });

  it('handles step at index 0 in single-step replay', () => {
    const replay = makeRawReplay({ steps: [[makeEntry({ observation: { only: true } })]] });
    const result = getStepData(replay, 0);
    expect(result).not.toBeNull();
    expect(result![0].observation).toEqual({ only: true });
  });

  it('handles last valid step index', () => {
    const replay = makeRawReplay({
      steps: [[makeEntry()], [makeEntry()], [makeEntry({ observation: { last: true } })]],
    });
    const result = getStepData(replay, 2);
    expect(result).not.toBeNull();
    expect(result![0].observation).toEqual({ last: true });
  });

  it('returns the full step array (all player entries)', () => {
    const replay = makeRawReplay({
      steps: [
        [makeEntry({ observation: { player: 0 }, reward: 1 }), makeEntry({ observation: { player: 1 }, reward: -1 })],
      ],
    });
    const result = getStepData(replay, 0);
    expect(result).toHaveLength(2);
    expect(result![0].reward).toBe(1);
    expect(result![1].reward).toBe(-1);
  });
});
