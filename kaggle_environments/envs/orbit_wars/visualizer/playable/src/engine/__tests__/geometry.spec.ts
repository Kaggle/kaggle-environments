import { describe, it, expect } from 'vitest';
import { distance, pointToSegmentDistance } from '../geometry';

describe('geometry', () => {
  it('distance returns euclidean distance', () => {
    expect(distance([0, 0], [3, 4])).toBe(5);
    expect(distance([1, 1], [1, 1])).toBe(0);
  });

  it('pointToSegmentDistance handles degenerate segment', () => {
    expect(pointToSegmentDistance([1, 1], [0, 0], [0, 0])).toBeCloseTo(Math.SQRT2);
  });

  it('pointToSegmentDistance projects onto interior', () => {
    expect(pointToSegmentDistance([5, 5], [0, 0], [10, 0])).toBe(5);
  });

  it('pointToSegmentDistance clamps to endpoints', () => {
    expect(pointToSegmentDistance([-3, 4], [0, 0], [10, 0])).toBe(5);
    expect(pointToSegmentDistance([15, -3], [0, 0], [10, 0])).toBeCloseTo(Math.hypot(5, 3));
  });
});
