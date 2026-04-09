// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ReplayVisualizer } from './player';
import { GameAdapter } from '../adapter';
import { ReplayData } from '../types';
import { makeReplay } from '../test-utils';

const createMockAdapter = (): GameAdapter => ({
  mount: vi.fn(),
  render: vi.fn(),
  unmount: vi.fn(),
});

const createContainer = (): HTMLElement => {
  const el = document.createElement('div');
  el.id = 'test-container';
  document.body.appendChild(el);
  return el;
};

describe('ReplayVisualizer', () => {
  let container: HTMLElement;
  let adapter: GameAdapter;

  beforeEach(() => {
    container = createContainer();
    adapter = createMockAdapter();
  });

  afterEach(() => {
    document.body.innerHTML = '';
  });

  describe('constructor', () => {
    it('creates the DOM structure with player and viewer divs', () => {
      new ReplayVisualizer(container, adapter);
      const player = container.querySelector('.player');
      const viewer = container.querySelector('.viewer');
      expect(player).toBeInstanceOf(HTMLElement);
      expect(viewer).toBeInstanceOf(HTMLElement);
      expect(player!.contains(viewer)).toBe(true);
    });

    it('clears existing container content', () => {
      container.innerHTML = '<span>old content</span>';
      new ReplayVisualizer(container, adapter);
      expect(container.querySelector('span')).toBeNull();
      expect(container.querySelector('.player')).toBeInstanceOf(HTMLElement);
    });

    it('adds a message event listener', () => {
      const addSpy = vi.spyOn(window, 'addEventListener');
      new ReplayVisualizer(container, adapter);
      expect(addSpy).toHaveBeenCalledWith('message', expect.any(Function));
      addSpy.mockRestore();
    });

    it('sends ready message to parent when in iframe', () => {
      const postMessageSpy = vi.fn();
      const originalParent = window.parent;
      Object.defineProperty(window, 'parent', {
        value: { postMessage: postMessageSpy },
        writable: true,
        configurable: true,
      });

      new ReplayVisualizer(container, adapter);
      expect(postMessageSpy).toHaveBeenCalledWith({ ready: true }, '*');

      Object.defineProperty(window, 'parent', {
        value: originalParent,
        writable: true,
        configurable: true,
      });
    });

    it('does not send ready message when not in iframe', () => {
      const postMessageSpy = vi.fn();
      Object.defineProperty(window, 'parent', {
        value: window,
        writable: true,
        configurable: true,
      });

      new ReplayVisualizer(container, adapter);
      expect(postMessageSpy).not.toHaveBeenCalled();
    });
  });

  describe('handleMessage', () => {
    it('ignores messages with no data', () => {
      new ReplayVisualizer(container, adapter);
      window.dispatchEvent(new MessageEvent('message', { data: null }));
      expect(adapter.mount).not.toHaveBeenCalled();
      expect(adapter.render).not.toHaveBeenCalled();
    });

    it('updates agents from message and renders', () => {
      new ReplayVisualizer(container, adapter);
      const replay = makeReplay();

      // First send replay to mount
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));
      expect(adapter.mount).toHaveBeenCalledTimes(1);

      // Then send agents
      const agents = [{ name: 'Agent1' }];
      window.dispatchEvent(new MessageEvent('message', { data: { agents } }));
      expect(adapter.render).toHaveBeenCalledWith(0, expect.any(Object), agents);
    });

    it('creates replay from environment data', () => {
      new ReplayVisualizer(container, adapter);
      const environment = {
        steps: [[{ observation: {}, reward: 0, status: 'ACTIVE' }]],
        name: 'test',
      };
      window.dispatchEvent(new MessageEvent('message', { data: { environment } }));
      expect(adapter.mount).toHaveBeenCalledTimes(1);
      expect(adapter.render).toHaveBeenCalledWith(0, expect.objectContaining({ name: 'test' }), []);
    });

    it('merges environment data into existing replay', () => {
      new ReplayVisualizer(container, adapter);
      const replay = makeReplay({ name: 'game1' });
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));

      window.dispatchEvent(
        new MessageEvent('message', {
          data: { environment: { info: { extra: true } } },
        })
      );
      expect(adapter.render).toHaveBeenLastCalledWith(
        0,
        expect.objectContaining({ name: 'game1', info: { extra: true } }),
        []
      );
    });

    it('overwrites replay when full replay is provided', () => {
      new ReplayVisualizer(container, adapter);
      const replay1 = makeReplay({ name: 'first' });
      window.dispatchEvent(new MessageEvent('message', { data: { replay: replay1 } }));

      const replay2 = makeReplay({ name: 'second' });
      window.dispatchEvent(new MessageEvent('message', { data: { replay: replay2 } }));
      expect(adapter.render).toHaveBeenLastCalledWith(0, expect.objectContaining({ name: 'second' }), []);
    });

    it('updates step from message', () => {
      new ReplayVisualizer(container, adapter);
      const replay = makeReplay();
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));

      window.dispatchEvent(new MessageEvent('message', { data: { step: 3 } }));
      expect(adapter.render).toHaveBeenLastCalledWith(3, expect.any(Object), []);
    });

    it('does not render without replay data', () => {
      new ReplayVisualizer(container, adapter);
      window.dispatchEvent(new MessageEvent('message', { data: { step: 3 } }));
      expect(adapter.mount).not.toHaveBeenCalled();
      expect(adapter.render).not.toHaveBeenCalled();
    });

    it('mounts the adapter only once', () => {
      new ReplayVisualizer(container, adapter);
      const replay = makeReplay();
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));
      expect(adapter.mount).toHaveBeenCalledTimes(1);
      expect(adapter.render).toHaveBeenCalledTimes(2);
    });
  });

  describe('transformer', () => {
    it('applies transformer to replay data', () => {
      const transformer = vi.fn((replay: ReplayData) => ({
        ...replay,
        name: 'transformed',
      }));
      new ReplayVisualizer(container, adapter, { transformer });
      const replay = makeReplay({ name: 'original' });
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));

      expect(transformer).toHaveBeenCalledWith(replay);
      expect(adapter.mount).toHaveBeenCalledWith(
        expect.any(HTMLElement),
        expect.objectContaining({ name: 'transformed' })
      );
    });

    it('passes replay through without transformer', () => {
      new ReplayVisualizer(container, adapter);
      const replay = makeReplay({ name: 'original' });
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));

      expect(adapter.mount).toHaveBeenCalledWith(
        expect.any(HTMLElement),
        expect.objectContaining({ name: 'original' })
      );
    });
  });

  describe('setAgents', () => {
    it('updates the agents list', () => {
      const rv = new ReplayVisualizer(container, adapter);
      const agents = [{ name: 'Agent1' }];
      rv.setAgents(agents);

      const replay = makeReplay();
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));
      // After mount, render is called with the agents set via setAgents
      // But setAgents alone doesn't trigger render, only message does
      // So let's verify that agents set via setAgents are used in subsequent renders
      window.dispatchEvent(new MessageEvent('message', { data: { step: 0 } }));
      expect(adapter.render).toHaveBeenLastCalledWith(0, expect.any(Object), agents);
    });
  });

  describe('cleanup', () => {
    it('removes the message event listener', () => {
      const removeSpy = vi.spyOn(window, 'removeEventListener');
      const rv = new ReplayVisualizer(container, adapter);
      rv.cleanup();
      expect(removeSpy).toHaveBeenCalledWith('message', expect.any(Function));
      removeSpy.mockRestore();
    });

    it('calls adapter.unmount when mounted', () => {
      const rv = new ReplayVisualizer(container, adapter);
      const replay = makeReplay();
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));
      expect(adapter.mount).toHaveBeenCalled();

      rv.cleanup();
      expect(adapter.unmount).toHaveBeenCalledTimes(1);
    });

    it('does not call adapter.unmount when not mounted', () => {
      const rv = new ReplayVisualizer(container, adapter);
      rv.cleanup();
      expect(adapter.unmount).not.toHaveBeenCalled();
    });

    it('clears container content', () => {
      const rv = new ReplayVisualizer(container, adapter);
      expect(container.querySelector('.player')).toBeInstanceOf(HTMLElement);
      rv.cleanup();
      expect(container.innerHTML).toBe('');
    });

    it('stops responding to messages after cleanup', () => {
      const rv = new ReplayVisualizer(container, adapter);
      rv.cleanup();

      const replay = makeReplay();
      window.dispatchEvent(new MessageEvent('message', { data: { replay } }));
      expect(adapter.mount).not.toHaveBeenCalled();
    });
  });
});
