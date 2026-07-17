import { describe, it, expect } from 'vitest';
import { parseThoughts, deriveWinnerFromRewards } from './transformer';
import { OpenSpielRawPlayer } from './types';

describe('parseThoughts', () => {
  it('returns empty string when no action', () => {
    expect(parseThoughts(undefined)).toBe('');
    expect(parseThoughts({})).toBe('');
  });

  it('prefers action.thoughts over generate_returns', () => {
    expect(
      parseThoughts({
        thoughts: 'from harness',
        generate_returns: [JSON.stringify({ main_response_and_thoughts: 'from raw' })],
      })
    ).toBe('from harness');
  });

  it('falls back to generate_returns[0].main_response_and_thoughts', () => {
    expect(
      parseThoughts({
        generate_returns: [JSON.stringify({ main_response_and_thoughts: 'from raw' })],
      })
    ).toBe('from raw');
  });

  it('returns empty string on invalid JSON fallback', () => {
    expect(parseThoughts({ generate_returns: ['not json'] })).toBe('');
  });
});

describe('deriveWinnerFromRewards', () => {
  const p = (reward: number): OpenSpielRawPlayer => ({ reward, observation: {} });

  it('returns null for degenerate step', () => {
    expect(deriveWinnerFromRewards([], ['A', 'B'])).toBeNull();
    expect(deriveWinnerFromRewards([p(1)], ['A', 'B'])).toBeNull();
  });

  it('returns Draw on equal rewards', () => {
    expect(deriveWinnerFromRewards([p(0), p(0)], ['A', 'B'])).toBe('Draw');
    expect(deriveWinnerFromRewards([p(1), p(1)], ['A', 'B'])).toBe('Draw');
  });

  it('picks the higher-reward player', () => {
    expect(deriveWinnerFromRewards([p(1), p(-1)], ['Alice', 'Bob'])).toBe('Alice wins!');
    expect(deriveWinnerFromRewards([p(-1), p(1)], ['Alice', 'Bob'])).toBe('Bob wins!');
  });

  it('honors the custom suffix', () => {
    expect(deriveWinnerFromRewards([p(1), p(-1)], ['Alice', 'Bob'], 'takes it.')).toBe('Alice takes it.');
  });
});
