// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ReplayAdapter, ReplayAdapterOptions } from './replay-adapter';
import { makeReplay } from '../test-utils';
import { ReplayData } from '../types';

vi.mock('react-dom/client', () => {
  const renderFn = vi.fn();
  const unmountFn = vi.fn();
  return {
    createRoot: vi.fn(() => ({
      render: renderFn,
      unmount: unmountFn,
    })),
    __renderFn: renderFn,
    __unmountFn: unmountFn,
  };
});

vi.mock('../components/EpisodePlayer', () => ({
  EpisodePlayer: () => null,
}));

vi.mock('../theme', () => ({
  theme: { palette: { mode: 'dark' } },
  lightTheme: { palette: { mode: 'light' } },
}));

vi.mock('../transformers/transformers', () => ({
  processEpisodeData: vi.fn((replay: any) => ({ ...replay, isTransformed: true })),
}));

const mockRenderer = vi.fn();

const createContainer = (): HTMLElement => {
  const el = document.createElement('div');
  el.id = 'adapter-container';
  document.body.appendChild(el);
  return el;
};

describe('ReplayAdapter', () => {
  let container: HTMLElement;
  let adapters: ReplayAdapter[] = [];

  // Registers the adapter for cleanup in afterEach to prevent pollution across tests.
  const createAdapter = (opts: ReplayAdapterOptions) => {
    const a = new ReplayAdapter(opts);
    adapters.push(a);
    return a;
  };

  beforeEach(() => {
    container = createContainer();
    adapters = [];
  });

  afterEach(() => {
    adapters.forEach((a) => a.unmount());
    document.body.innerHTML = '';
    vi.clearAllMocks();
  });

  describe('constructor', () => {
    it('throws when neither renderer nor GameRenderer is provided', () => {
      expect(() => new ReplayAdapter({ gameName: 'test' } as any)).toThrow(
        'ReplayAdapter requires either `renderer` or `GameRenderer` option'
      );
    });

    it('accepts a renderer function', () => {
      expect(createAdapter({ gameName: 'test', renderer: mockRenderer })).toBeDefined();
    });

    it('accepts a GameRenderer component', () => {
      const GameRenderer = () => null;
      expect(createAdapter({ gameName: 'test', GameRenderer })).toBeDefined();
    });

    it('adds a message listener for theme changes', () => {
      const addSpy = vi.spyOn(window, 'addEventListener');
      createAdapter({ gameName: 'test', renderer: mockRenderer });
      expect(addSpy).toHaveBeenCalledWith('message', expect.any(Function));
      addSpy.mockRestore();
    });
  });

  describe('mount', () => {
    it('transforms initial data when provided', async () => {
      const { processEpisodeData } = await import('../transformers/transformers');
      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      const replay = makeReplay({ name: 'to-transform' });
      adapter.mount(container, replay);
      expect(processEpisodeData).toHaveBeenCalledWith(replay, 'test');
    });

    it('sends iframeCapabilities to parent when in iframe', () => {
      const postMessageSpy = vi.fn();
      const originalParent = window.parent;
      Object.defineProperty(window, 'parent', {
        value: { postMessage: postMessageSpy },
        writable: true,
        configurable: true,
      });

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);

      expect(postMessageSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          iframeCapabilities: expect.objectContaining({
            iframeHandlesUi: true,
            hasControls: true,
            hasReasoningLogs: true,
            managesPlayback: true,
          }),
        }),
        '*'
      );

      Object.defineProperty(window, 'parent', {
        value: originalParent,
        writable: true,
        configurable: true,
      });
    });

    it('reports no reasoning logs for inline UI mode', () => {
      const postMessageSpy = vi.fn();
      const originalParent = window.parent;
      Object.defineProperty(window, 'parent', {
        value: { postMessage: postMessageSpy },
        writable: true,
        configurable: true,
      });

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer, ui: 'inline' });
      adapter.mount(container);

      expect(postMessageSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          iframeCapabilities: expect.objectContaining({
            hasReasoningLogs: false,
            iframeHandlesUi: true,
          }),
        }),
        '*'
      );

      Object.defineProperty(window, 'parent', {
        value: originalParent,
        writable: true,
        configurable: true,
      });
    });
  });

  describe('render', () => {
    it('transforms replay data and re-renders', async () => {
      const { processEpisodeData } = await import('../transformers/transformers');
      const { __renderFn: renderFn } = (await import('react-dom/client')) as any;

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);
      renderFn.mockClear();

      const replay = makeReplay({ name: 'new-data' });
      adapter.render(0, replay, [{ name: 'Agent1' }]);

      expect(processEpisodeData).toHaveBeenCalledWith(replay, 'test');
      expect(renderFn).toHaveBeenCalled();
    });

    it('does not re-transform if replay reference is the same', async () => {
      const { processEpisodeData } = await import('../transformers/transformers');

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      const replay = makeReplay();
      adapter.mount(container, replay);
      (processEpisodeData as any).mockClear();

      adapter.render(1, replay, []);
      expect(processEpisodeData).not.toHaveBeenCalled();
    });

    it('uses custom transformer when provided', async () => {
      const { processEpisodeData } = await import('../transformers/transformers');

      const customTransformer = vi.fn((replay: ReplayData, gameName: string) => ({
        ...replay,
        name: 'custom-' + gameName,
      }));
      const adapter = createAdapter({
        gameName: 'chess',
        renderer: mockRenderer,
        transformer: customTransformer,
      });

      const replay = makeReplay({ name: 'original' });
      adapter.mount(container, replay);

      expect(customTransformer).toHaveBeenCalledWith(replay, 'chess');
      expect(processEpisodeData).not.toHaveBeenCalled();
    });
  });

  describe('unmount', () => {
    it('removes the theme message listener', () => {
      const removeSpy = vi.spyOn(window, 'removeEventListener');
      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);

      adapter.unmount();
      expect(removeSpy).toHaveBeenCalledWith('message', expect.any(Function));
      removeSpy.mockRestore();
    });

    it('unmounts the React root', async () => {
      const { __unmountFn: unmountFn } = (await import('react-dom/client')) as any;

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);

      adapter.unmount();
      expect(unmountFn).toHaveBeenCalled();
    });

    it('is safe to call without mount', () => {
      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      expect(() => adapter.unmount()).not.toThrow();
    });

    it('nullifies root after unmount', async () => {
      const { __renderFn: renderFn } = (await import('react-dom/client')) as any;

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);
      adapter.unmount();

      renderFn.mockClear();
      adapter.render(0, makeReplay(), []);
      expect(renderFn).not.toHaveBeenCalled();
    });
  });

  describe('theme switching', () => {
    it('updates theme to light on postMessage', async () => {
      const { __renderFn: renderFn } = (await import('react-dom/client')) as any;

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);
      renderFn.mockClear();

      window.dispatchEvent(new MessageEvent('message', { data: { theme: 'light' } }));
      expect(renderFn).toHaveBeenCalled();
    });

    it('updates theme to dark on postMessage', async () => {
      const { __renderFn: renderFn } = (await import('react-dom/client')) as any;

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);

      window.dispatchEvent(new MessageEvent('message', { data: { theme: 'light' } }));
      renderFn.mockClear();

      window.dispatchEvent(new MessageEvent('message', { data: { theme: 'dark' } }));
      expect(renderFn).toHaveBeenCalled();
    });

    it('does not re-render when theme is already the same', async () => {
      const { __renderFn: renderFn } = (await import('react-dom/client')) as any;

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);
      renderFn.mockClear();

      window.dispatchEvent(new MessageEvent('message', { data: { theme: 'dark' } }));
      expect(renderFn).not.toHaveBeenCalled();
    });

    it('does not re-render before mount', async () => {
      const { __renderFn: renderFn } = (await import('react-dom/client')) as any;
      renderFn.mockClear();

      createAdapter({ gameName: 'test', renderer: mockRenderer });

      window.dispatchEvent(new MessageEvent('message', { data: { theme: 'light' } }));
      expect(renderFn).not.toHaveBeenCalled();
    });

    it('ignores non-object messages', async () => {
      const { __renderFn: renderFn } = (await import('react-dom/client')) as any;

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);
      renderFn.mockClear();

      window.dispatchEvent(new MessageEvent('message', { data: null }));
      window.dispatchEvent(new MessageEvent('message', { data: 'string' }));
      expect(renderFn).not.toHaveBeenCalled();
    });
  });

  describe('dense mode', () => {
    it('re-renders on dense change via postMessage', async () => {
      const { __renderFn: renderFn } = (await import('react-dom/client')) as any;

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);
      renderFn.mockClear();

      window.dispatchEvent(new MessageEvent('message', { data: { dense: true } }));
      expect(renderFn).toHaveBeenCalled();
    });

    it('does not re-render when dense is already the same', async () => {
      const { __renderFn: renderFn } = (await import('react-dom/client')) as any;

      const adapter = createAdapter({ gameName: 'test', renderer: mockRenderer });
      adapter.mount(container);
      renderFn.mockClear();

      window.dispatchEvent(new MessageEvent('message', { data: { dense: false } }));
      expect(renderFn).not.toHaveBeenCalled();
    });
  });
});
