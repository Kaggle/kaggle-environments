// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { createReplayVisualizer } from './replay-visualizer-factory';
import { GameAdapter } from '../adapter';
import { ReplayVisualizer } from '../player/player';
import { ReplayData } from '../types';
import { makeReplay } from '../test-utils';

const createMockAdapter = (): GameAdapter => ({
  mount: vi.fn(),
  render: vi.fn(),
  unmount: vi.fn(),
});

const createContainer = (id = 'test-container'): HTMLElement => {
  const el = document.createElement('div');
  el.id = id;
  document.body.appendChild(el);
  return el;
};

describe('createReplayVisualizer', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('returns a ReplayVisualizer instance', () => {
    const container = createContainer();
    const adapter = createMockAdapter();
    const rv = createReplayVisualizer(container, adapter);
    expect(rv).toBeInstanceOf(ReplayVisualizer);
  });

  it('creates DOM structure inside the container', () => {
    const container = createContainer();
    const adapter = createMockAdapter();
    createReplayVisualizer(container, adapter);
    expect(container.querySelector('.player')).toBeInstanceOf(HTMLElement);
    expect(container.querySelector('.viewer')).toBeInstanceOf(HTMLElement);
  });

  it('passes the transformer option through to ReplayVisualizer', () => {
    const container = createContainer();
    const adapter = createMockAdapter();
    const transformer = vi.fn((replay: ReplayData) => ({
      ...replay,
      name: 'transformed',
    }));

    createReplayVisualizer(container, adapter, { transformer });

    const replay = makeReplay({ name: 'original' });
    window.dispatchEvent(new MessageEvent('message', { data: { replay } }));

    expect(transformer).toHaveBeenCalledWith(replay);
    expect(adapter.mount).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      expect.objectContaining({ name: 'transformed' })
    );
  });

  it('created instance responds to postMessage', () => {
    const container = createContainer();
    const adapter = createMockAdapter();
    createReplayVisualizer(container, adapter);

    const replay = makeReplay();
    window.dispatchEvent(new MessageEvent('message', { data: { replay } }));

    expect(adapter.mount).toHaveBeenCalledTimes(1);
    expect(adapter.render).toHaveBeenCalledTimes(1);
  });

  it('created instance can be cleaned up', () => {
    const container = createContainer();
    const adapter = createMockAdapter();
    const rv = createReplayVisualizer(container, adapter);

    const replay = makeReplay();
    window.dispatchEvent(new MessageEvent('message', { data: { replay } }));

    rv.cleanup();
    expect(adapter.unmount).toHaveBeenCalledTimes(1);
    expect(container.innerHTML).toBe('');
  });

  it('works without transformer option', () => {
    const container = createContainer();
    const adapter = createMockAdapter();
    createReplayVisualizer(container, adapter);

    const replay = makeReplay({ name: 'no-transform' });
    window.dispatchEvent(new MessageEvent('message', { data: { replay } }));

    expect(adapter.mount).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      expect.objectContaining({ name: 'no-transform' })
    );
  });
});
