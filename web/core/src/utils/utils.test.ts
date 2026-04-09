import { describe, it, expect } from 'vitest';
import { getPlayer } from './utils';
import { makeStep } from '../test-utils';

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
