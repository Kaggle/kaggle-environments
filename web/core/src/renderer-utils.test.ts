import { describe, it, expect } from 'vitest';
import { getStepData } from './renderer-utils';
import { RawReplayData, RawPlayerEntry } from './types';

const makeEntry = (overrides: Partial<RawPlayerEntry> = {}): RawPlayerEntry => ({
  observation: { board: [0, 0, 0] },
  reward: 0,
  status: 'ACTIVE',
  ...overrides,
});

const makeReplay = (overrides: Partial<RawReplayData> = {}): RawReplayData => ({
  name: 'test-game',
  version: '1.0',
  steps: [[makeEntry(), makeEntry()]],
  configuration: {},
  ...overrides,
});

describe('getStepData', () => {
  it('returns step data for a valid step index', () => {
    const replay = makeReplay({
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
    const replay = makeReplay({
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

  // --- Null returns for invalid inputs ---

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
    const replay = makeReplay();
    expect(getStepData(replay, -1)).toBeNull();
  });

  it('returns null for step index beyond bounds', () => {
    const replay = makeReplay({ steps: [[makeEntry()]] });
    expect(getStepData(replay, 1)).toBeNull();
  });

  it('returns null when step data is empty array', () => {
    const replay = makeReplay({ steps: [[]] as any });
    expect(getStepData(replay, 0)).toBeNull();
  });

  it('returns null when step data is not an array', () => {
    const replay = makeReplay({ steps: [{ notArray: true }] as any });
    expect(getStepData(replay, 0)).toBeNull();
  });

  it('returns null when first entry has no observation', () => {
    const replay = makeReplay({
      steps: [[{ reward: 0, status: 'ACTIVE' }]] as any,
    });
    expect(getStepData(replay, 0)).toBeNull();
  });

  it('returns null when first entry is not an object', () => {
    const replay = makeReplay({ steps: [['string-entry']] as any });
    expect(getStepData(replay, 0)).toBeNull();
  });

  it('returns null when first entry is null', () => {
    const replay = makeReplay({ steps: [[null, makeEntry()]] as any });
    expect(getStepData(replay, 0)).toBeNull();
  });

  // --- Typed observation ---

  it('works with custom observation types', () => {
    interface GoObservation {
      board: number[][];
      currentPlayer: number;
    }
    const obs: GoObservation = {
      board: [
        [0, 1],
        [1, 0],
      ],
      currentPlayer: 1,
    };
    const replay: RawReplayData<GoObservation> = {
      name: 'go',
      version: '1.0',
      steps: [[{ observation: obs, reward: null, status: 'ACTIVE' }]],
      configuration: {},
    };
    const result = getStepData<GoObservation>(replay, 0);
    expect(result).not.toBeNull();
    expect(result![0].observation.currentPlayer).toBe(1);
  });

  // --- Edge cases ---

  it('handles step at index 0 in single-step replay', () => {
    const replay = makeReplay({ steps: [[makeEntry({ observation: { only: true } })]] });
    const result = getStepData(replay, 0);
    expect(result).not.toBeNull();
    expect(result![0].observation).toEqual({ only: true });
  });

  it('handles last valid step index', () => {
    const replay = makeReplay({
      steps: [[makeEntry()], [makeEntry()], [makeEntry({ observation: { last: true } })]],
    });
    const result = getStepData(replay, 2);
    expect(result).not.toBeNull();
    expect(result![0].observation).toEqual({ last: true });
  });

  it('returns the full step array (all player entries)', () => {
    const replay = makeReplay({
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
