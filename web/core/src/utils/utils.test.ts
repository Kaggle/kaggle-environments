import { describe, it, expect } from 'vitest';
import { getPlayer, applyAgentNamesToReplay } from './utils';
import { makeStep, makeReplay } from '../test-utils';

const alice = { id: 0, name: 'Alice', thumbnail: '', isTurn: true };
const bob = { id: 1, name: 'Bob', thumbnail: '', isTurn: false };

describe('getPlayer', () => {
  it('returns the player whose turn it is', () => {
    const step = makeStep({ players: [alice, bob] });
    expect(getPlayer(step)).toEqual(alice);
  });

  it('returns the second player when they have the turn', () => {
    const step = makeStep({
      players: [
        { ...alice, isTurn: false },
        { ...bob, isTurn: true },
      ],
    });
    expect(getPlayer(step)).toEqual({ ...bob, isTurn: true });
  });

  it('returns undefined when no player has isTurn', () => {
    const step = makeStep({
      players: [
        { ...alice, isTurn: false },
        { ...bob, isTurn: false },
      ],
    });
    expect(getPlayer(step)).toBeUndefined();
  });

  it('returns undefined when players array is empty', () => {
    const step = makeStep({ players: [] });
    expect(getPlayer(step)).toBeUndefined();
  });

  it('returns undefined when players is missing', () => {
    const step = { step: 1 } as any;
    expect(getPlayer(step)).toBeUndefined();
  });

  it('returns the first matching player when multiple have isTurn', () => {
    const step = makeStep({
      players: [
        { ...alice, isTurn: true },
        { ...bob, isTurn: true },
      ],
    });
    expect(getPlayer(step)).toEqual(alice);
  });
});

describe('applyAgentNamesToReplay', () => {
  it('overrides info.TeamNames with agents[].name', () => {
    const replay = makeReplay({ info: { TeamNames: ['Old A', 'Old B'] } });
    const result = applyAgentNamesToReplay(replay, [{ name: 'New A' }, { name: 'New B' }]);
    expect(result.info?.TeamNames).toEqual(['New A', 'New B']);
  });

  it('returns the same reference when no override is needed', () => {
    const replay = makeReplay({ info: { TeamNames: ['A', 'B'] } });
    const result = applyAgentNamesToReplay(replay, [{ name: 'A' }, { name: 'B' }]);
    expect(result).toBe(replay);
  });

  it('returns the same reference when agents is empty', () => {
    const replay = makeReplay({ info: { TeamNames: ['A', 'B'] } });
    expect(applyAgentNamesToReplay(replay, [])).toBe(replay);
    expect(applyAgentNamesToReplay(replay, undefined)).toBe(replay);
    expect(applyAgentNamesToReplay(replay, null)).toBe(replay);
  });

  it('keeps existing TeamName when agent.name is missing or empty', () => {
    const replay = makeReplay({ info: { TeamNames: ['Keep A', 'Keep B'] } });
    const result = applyAgentNamesToReplay(replay, [{ name: '' }, {}]);
    expect(result.info?.TeamNames).toEqual(['Keep A', 'Keep B']);
    expect(result).toBe(replay);
  });

  it('seeds TeamNames when replay has no info.TeamNames', () => {
    const replay = makeReplay({ info: {} });
    const result = applyAgentNamesToReplay(replay, [{ name: 'A' }, { name: 'B' }]);
    expect(result.info?.TeamNames).toEqual(['A', 'B']);
  });

  it('seeds TeamNames when replay has no info at all', () => {
    const replay = makeReplay();
    delete (replay as any).info;
    const result = applyAgentNamesToReplay(replay, [{ name: 'A' }, { name: 'B' }]);
    expect(result.info?.TeamNames).toEqual(['A', 'B']);
  });

  it('respects agent.index for sparse / out-of-order agents', () => {
    const replay = makeReplay({ info: { TeamNames: ['x', 'y', 'z'] } });
    const result = applyAgentNamesToReplay(replay, [{ name: 'Z', index: 2 }, { name: 'X', index: 0 }]);
    expect(result.info?.TeamNames).toEqual(['X', 'y', 'Z']);
  });

  it('preserves other info fields', () => {
    const replay = makeReplay({ info: { TeamNames: ['A'], LiveStats: { wins: 5 } } });
    const result = applyAgentNamesToReplay(replay, [{ name: 'A2' }]);
    expect(result.info?.LiveStats).toEqual({ wins: 5 });
    expect(result.info?.TeamNames).toEqual(['A2']);
  });
});
