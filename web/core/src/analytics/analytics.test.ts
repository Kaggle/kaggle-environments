import { describe, it, expect, vi, beforeEach } from 'vitest';
import { postAnalyticsEvent } from './analytics';

describe('postAnalyticsEvent', () => {
  const postMessageSpy = vi.fn();

  beforeEach(() => {
    postMessageSpy.mockClear();
  });

  it('posts a string event to the parent frame', () => {
    vi.stubGlobal('window', { parent: { postMessage: postMessageSpy } });

    postAnalyticsEvent('play_clicked');
    expect(postMessageSpy).toHaveBeenCalledWith({ analyticsEvent: 'play_clicked' }, '*');
  });

  it('posts an object event to the parent frame', () => {
    vi.stubGlobal('window', { parent: { postMessage: postMessageSpy } });

    const event = { game: 'chess', action: 'step_changed', step: 5 };
    postAnalyticsEvent(event);
    expect(postMessageSpy).toHaveBeenCalledWith({ analyticsEvent: event }, '*');
  });

  it('does nothing when window is undefined', () => {
    vi.stubGlobal('window', undefined);

    expect(() => postAnalyticsEvent('test')).not.toThrow();
    expect(postMessageSpy).not.toHaveBeenCalled();
  });

  it('does nothing when not in an iframe (parent === window)', () => {
    const self = { postMessage: postMessageSpy } as any;
    self.parent = self;
    vi.stubGlobal('window', self);

    postAnalyticsEvent('test');
    expect(postMessageSpy).not.toHaveBeenCalled();
  });
});
