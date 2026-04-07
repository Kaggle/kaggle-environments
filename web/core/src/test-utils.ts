import { BaseGameStep, RawPlayerEntry, RawReplayData, ReplayData } from './types';

export const makeStep = (overrides: Partial<BaseGameStep> = {}): BaseGameStep => ({
  step: 1,
  players: [
    { id: 0, name: 'Alice', thumbnail: '', isTurn: true },
    { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
  ],
  ...overrides,
});

export const makeReplay = (overrides: Partial<ReplayData> = {}): ReplayData => ({
  name: 'test-game',
  version: '1.0',
  steps: [makeStep()],
  configuration: {},
  ...overrides,
});

export const makeEntry = (overrides: Partial<RawPlayerEntry> = {}): RawPlayerEntry => ({
  observation: { board: [0, 0, 0] },
  reward: 0,
  status: 'ACTIVE',
  ...overrides,
});

export const makeRawReplay = (overrides: Partial<RawReplayData> = {}): RawReplayData => ({
  name: 'test-game',
  version: '1.0',
  steps: [[makeEntry(), makeEntry()]],
  configuration: {},
  ...overrides,
});
