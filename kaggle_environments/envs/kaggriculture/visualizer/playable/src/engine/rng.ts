/**
 * Mersenne Twister (MT19937) PRNG with a Python-Random-style API. Same shape
 * as the orbit_wars playable engine's `rng.ts`, with an added `choice` so we
 * can mirror Python's `rng.choice(sorted(remaining))` for end-of-day shop
 * unlocks.
 *
 * Deterministic in TS but NOT bit-identical to CPython's `random.Random` —
 * we don't reproduce Python's seeding pipeline. Stochastic events
 * (weed spawn, shop unlock) will therefore diverge from Python rollouts even
 * for the same seed. Stage 3 will address this where it matters for replay
 * tests.
 */

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

export class PyRandom {
  private mt: Uint32Array;
  private index: number;

  constructor(seed: number) {
    this.mt = new Uint32Array(N);
    this.index = N + 1;
    this.seedMt(seed >>> 0);
  }

  private seedMt(seed: number): void {
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = this.mt[i - 1];
      const x = Math.imul(1812433253, prev ^ (prev >>> 30)) + i;
      this.mt[i] = x >>> 0;
    }
    this.index = N;
  }

  private generate(): void {
    for (let i = 0; i < N; i++) {
      const y = ((this.mt[i] & UPPER_MASK) | (this.mt[(i + 1) % N] & LOWER_MASK)) >>> 0;
      const next = (this.mt[(i + M) % N] ^ (y >>> 1)) >>> 0;
      this.mt[i] = (y & 1) === 0 ? next : (next ^ MATRIX_A) >>> 0;
    }
    this.index = 0;
  }

  nextUint32(): number {
    if (this.index >= N) this.generate();
    let y = this.mt[this.index++];
    y ^= y >>> 11;
    y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
    y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** [0, 1) with 53 bits of randomness, matching Python's random.random(). */
  random(): number {
    const a = this.nextUint32() >>> 5; // 27 bits
    const b = this.nextUint32() >>> 6; // 26 bits
    return (a * 67108864 + b) / 9007199254740992;
  }

  /** Inclusive integer in [a, b]. */
  randint(a: number, b: number): number {
    return a + Math.floor(this.random() * (b - a + 1));
  }

  uniform(a: number, b: number): number {
    return a + (b - a) * this.random();
  }

  /** Mirrors Python's `random.Random.choice` — uniform pick from a non-empty list. */
  choice<T>(items: readonly T[]): T {
    if (items.length === 0) throw new Error('rng.choice on empty array');
    return items[Math.floor(this.random() * items.length)];
  }
}

/**
 * Mirrors the per-day RNG seeding used by `_end_of_day`:
 *   random.Random((seed * 1_000_003) ^ day)
 *
 * Done with BigInt because (seed * 1_000_003) can exceed Number's safe
 * integer range for large seeds. We truncate to 32 bits before handing to
 * PyRandom, since the MT seed is treated as a u32.
 */
export function endOfDaySeed(seed: number, day: number): number {
  const product = BigInt(seed) * 1_000_003n;
  const xored = product ^ BigInt(day);
  return Number(xored & 0xffffffffn);
}
