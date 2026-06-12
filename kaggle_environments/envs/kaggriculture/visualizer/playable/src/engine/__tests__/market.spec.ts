import { describe, expect, it } from 'vitest';
import { marketPrice } from '../market';

describe('marketPrice', () => {
  const cases: Array<[any, number, number]> = [
    ['WHEAT', 10000, 25],
    ['WHEAT', 9000, 57],
    ['WHEAT', 11000, 19],
    ['WHEAT', 5000, 96],
    ['STRAWBERRY', 10500, 1],
    ['STRAWBERRY', 9500, 308],
    ['STRAWBERRY', 1_000_000_000, 1],
    ['MELON', 10100, 225],
    ['MELON', 9900, 290],
    ['FERTILIZER', 10000, 100],
    ['FERTILIZER', 9500, 200],
  ];
  for (const [item, inv, expected] of cases) {
    it(`${item} @ ${inv} -> ${expected}`, () => {
      expect(marketPrice(item, inv)).toBe(expected);
    });
  }
});
