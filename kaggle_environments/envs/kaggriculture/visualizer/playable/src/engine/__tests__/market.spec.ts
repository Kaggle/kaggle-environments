import { describe, expect, it } from 'vitest';
import { marketPrice, shape } from '../market';

describe('marketPrice', () => {
  const cases: Array<[any, number, number]> = [
    ['WHEAT', 10000, 25],
    ['WHEAT', 9000, 57],
    ['WHEAT', 11000, 19],
    ['WHEAT', 5000, 96],
    ['STRAWBERRY', 10500, 1],
    ['STRAWBERRY', 9500, 308],
    ['STRAWBERRY', 1_000_000_000, 1],
    ['MELON', 10100, 150],
    ['MELON', 9900, 290],
    ['FERTILIZER', 10000, 100],
    ['FERTILIZER', 9500, 200],
    // CARROT uses the hinge curve: calm below I0-T (=9550), spiking past it.
    ['CARROT', 10000, 35],
    ['CARROT', 9775, 53],
    ['CARROT', 9550, 70],
    ['CARROT', 9400, 113],
    ['CARROT', 9100, 385],
    // TOMATO is hinge too, but with below_target left at linear's 0.40, so
    // everything down to I0-T (=9800) is unchanged from the old linear curve.
    ['TOMATO', 10000, 60],
    ['TOMATO', 9900, 72],
    ['TOMATO', 9800, 84],
    ['TOMATO', 9700, 144],
    ['TOMATO', 9500, 552],
    // EGG likewise: hinge with linear's old below_target of 0.40, knee at 9668.
    ['EGG', 10000, 50],
    ['EGG', 9834, 60],
    ['EGG', 9668, 70],
    ['EGG', 9502, 120],
    ['EGG', 9170, 460],
  ];
  for (const [item, inv, expected] of cases) {
    it(`${item} @ ${inv} -> ${expected}`, () => {
      expect(marketPrice(item, inv)).toBe(expected);
    });
  }
});

describe('hinge shape', () => {
  it('is calibrated so f(T) === 1', () => {
    expect(shape('hinge', 450, 450)).toBeCloseTo(1);
  });

  it('degenerates to linear without a usable T', () => {
    for (const x of [0, 1, 50, 1000]) {
      expect(shape('hinge', x, undefined)).toBe(x);
      expect(shape('hinge', x, 0)).toBe(x);
    }
  });

  it('is linear below the knee and superlinear above it', () => {
    const T = 450;
    expect(shape('hinge', T / 2, T)).toBeCloseTo(0.5);
    expect(shape('hinge', 2 * T, T)).toBeGreaterThan(4 * shape('hinge', T, T));
  });
});
