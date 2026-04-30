import { describe, it, expect } from 'vitest';
import { PyRandom, hashStringToSeed } from '../rng';

describe('PyRandom', () => {
  it('is reproducible for the same seed', () => {
    const a = new PyRandom(42);
    const b = new PyRandom(42);
    for (let i = 0; i < 100; i++) {
      expect(a.random()).toBe(b.random());
    }
  });

  it('randint stays in inclusive range', () => {
    const r = new PyRandom(1);
    for (let i = 0; i < 1000; i++) {
      const v = r.randint(5, 10);
      expect(v).toBeGreaterThanOrEqual(5);
      expect(v).toBeLessThanOrEqual(10);
    }
  });

  it('uniform stays in [a,b]', () => {
    const r = new PyRandom(2);
    for (let i = 0; i < 1000; i++) {
      const v = r.uniform(0.5, 1.25);
      expect(v).toBeGreaterThanOrEqual(0.5);
      expect(v).toBeLessThan(1.25);
    }
  });

  it('hashStringToSeed is stable', () => {
    expect(hashStringToSeed('foo')).toBe(hashStringToSeed('foo'));
    expect(hashStringToSeed('foo')).not.toBe(hashStringToSeed('bar'));
  });
});
