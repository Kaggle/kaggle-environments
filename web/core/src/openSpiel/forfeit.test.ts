import { describe, it, expect } from 'vitest';
import { detectForfeit, buildForfeitReason, FORFEIT_STATUSES, FORFEIT_REASONS } from './forfeit';
import { OpenSpielRawPlayer } from './types';

const activePlayer = (opts: Partial<OpenSpielRawPlayer> = {}): OpenSpielRawPlayer => ({
  action: { submission: 0, actionString: 'move' },
  observation: {},
  reward: 0,
  status: 'DONE',
  ...opts,
});

describe('detectForfeit', () => {
  it('returns null for invalid input', () => {
    expect(detectForfeit(null)).toBeNull();
    expect(detectForfeit(undefined)).toBeNull();
    expect(detectForfeit([])).toBeNull();
    expect(detectForfeit([activePlayer()])).toBeNull();
  });

  it('returns null for a normal step', () => {
    expect(detectForfeit([activePlayer(), activePlayer({ action: { submission: -1 } })])).toBeNull();
  });

  for (const key of ['TIMEOUT', 'ERROR', 'INVALID']) {
    it(`detects top-level status ${key}`, () => {
      const result = detectForfeit([activePlayer({ status: key }), activePlayer()]);
      expect(result).toEqual({ index: 0, reasonKey: key });
    });
  }

  it('detects illegalMoveForfeit path (submission=-1 + action.status, both DONE)', () => {
    const result = detectForfeit([
      activePlayer(),
      activePlayer({ action: { submission: -1, status: 'failed to parse' } }),
    ]);
    expect(result).toEqual({ index: 1, reasonKey: 'INVALID' });
  });

  it('returns null when both players carry a forfeit top-level status', () => {
    expect(detectForfeit([activePlayer({ status: 'TIMEOUT' }), activePlayer({ status: 'ERROR' })])).toBeNull();
  });

  it('does not fire on action.submission=-1 without an action.status', () => {
    expect(detectForfeit([activePlayer(), activePlayer({ action: { submission: -1 } })])).toBeNull();
  });

  it('prefers top-level status over action-status signal', () => {
    const result = detectForfeit([
      activePlayer({ status: 'TIMEOUT' }),
      activePlayer({ action: { submission: -1, status: 'illegal' } }),
    ]);
    expect(result).toEqual({ index: 0, reasonKey: 'TIMEOUT' });
  });
});

describe('buildForfeitReason', () => {
  it('names loser and winner with the standard reason phrase', () => {
    expect(buildForfeitReason({ index: 0, reasonKey: 'TIMEOUT' }, ['Alice', 'Bob'])).toBe(
      'Alice ran out of time. Bob wins by default.'
    );
    expect(buildForfeitReason({ index: 1, reasonKey: 'INVALID' }, ['Alice', 'Bob'])).toBe(
      'Bob submitted an illegal move. Alice wins by default.'
    );
  });

  it('falls back to Player N labels when team names missing', () => {
    expect(buildForfeitReason({ index: 1, reasonKey: 'ERROR' }, [])).toBe(
      'Player 2 failed to produce valid input. Player 1 wins by default.'
    );
  });

  it('falls back to a generic verb for unknown reason keys', () => {
    expect(buildForfeitReason({ index: 0, reasonKey: 'MYSTERY' }, ['A', 'B'])).toBe('A forfeited. B wins by default.');
  });
});

describe('exported constants', () => {
  it('FORFEIT_STATUSES contains the three canonical keys', () => {
    expect(FORFEIT_STATUSES.has('TIMEOUT')).toBe(true);
    expect(FORFEIT_STATUSES.has('ERROR')).toBe(true);
    expect(FORFEIT_STATUSES.has('INVALID')).toBe(true);
    expect(FORFEIT_STATUSES.has('DONE')).toBe(false);
  });

  it('FORFEIT_REASONS maps each status key', () => {
    expect(FORFEIT_REASONS.TIMEOUT).toBeTruthy();
    expect(FORFEIT_REASONS.INVALID).toBeTruthy();
    expect(FORFEIT_REASONS.ERROR).toBeTruthy();
  });
});
