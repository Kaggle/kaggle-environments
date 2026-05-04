/**
 * Mersenne Twister (MT19937) PRNG with a Python-Random-style API.
 *
 * Internally deterministic — the same numeric seed always produces the same
 * sequence. Not bit-identical to CPython's `random.Random` (we don't reproduce
 * Python's seeding pipeline), but stable across browsers.
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
      // 1812433253 * (prev ^ (prev >>> 30)) + i, mod 2^32
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

  /** Next raw 32-bit unsigned int. */
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

  /** Float in [a, b]. */
  uniform(a: number, b: number): number {
    return a + (b - a) * this.random();
  }
}

/**
 * Cheap 32-bit string hash (cyrb53 -> truncated). Used to derive a numeric
 * seed for per-step comet RNGs without needing to reproduce Python's
 * SHA-512-based string seeding.
 */
export function hashStringToSeed(str: string): number {
  let h1 = 0xdeadbeef ^ 0;
  let h2 = 0x41c6ce57 ^ 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (h1 ^ h2) >>> 0;
}
