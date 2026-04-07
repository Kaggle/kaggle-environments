import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  getGameAssetsBaseUrl,
  getEpisodeAssetsBaseUrl,
  getGameAssetUrl,
  getEpisodeAssetUrl,
  rebaseAssetPaths,
  fetchGameAsset,
  fetchEpisodeAsset,
  fetchAndRebaseGameAssetMap,
  fetchAndRebaseEpisodeAssetMap,
} from './episodeAssetUtils';

describe('getGameAssetsBaseUrl', () => {
  it('uses provided baseUrl', () => {
    expect(getGameAssetsBaseUrl({ gameName: 'chess', baseUrl: 'https://cdn.example.com' })).toBe(
      'https://cdn.example.com/episode-assets/chess'
    );
  });

  it('uses window.location.origin when no baseUrl provided', () => {
    const originalWindow = globalThis.window;
    globalThis.window = { location: { origin: 'https://kaggle.com' } } as any;
    expect(getGameAssetsBaseUrl({ gameName: 'werewolf' })).toBe('https://kaggle.com/episode-assets/werewolf');
    globalThis.window = originalWindow;
  });

  it('uses empty string when no baseUrl and no window', () => {
    const originalWindow = globalThis.window;
    delete (globalThis as any).window;
    expect(getGameAssetsBaseUrl({ gameName: 'go' })).toBe('/episode-assets/go');
    globalThis.window = originalWindow;
  });
});

describe('getEpisodeAssetsBaseUrl', () => {
  it('appends episodes/{episodeId} to game base URL', () => {
    expect(
      getEpisodeAssetsBaseUrl({ gameName: 'werewolf', episodeId: '12345', baseUrl: 'https://cdn.example.com' })
    ).toBe('https://cdn.example.com/episode-assets/werewolf/episodes/12345');
  });
});

describe('getGameAssetUrl', () => {
  it('builds full URL for a game asset', () => {
    expect(getGameAssetUrl({ gameName: 'werewolf', baseUrl: 'https://cdn.example.com' }, 'config.json')).toBe(
      'https://cdn.example.com/episode-assets/werewolf/config.json'
    );
  });

  it('strips leading slash from asset path', () => {
    expect(getGameAssetUrl({ gameName: 'werewolf', baseUrl: 'https://cdn.example.com' }, '/config.json')).toBe(
      'https://cdn.example.com/episode-assets/werewolf/config.json'
    );
  });

  it('handles nested asset paths', () => {
    expect(getGameAssetUrl({ gameName: 'werewolf', baseUrl: 'https://cdn.example.com' }, 'textures/bg.png')).toBe(
      'https://cdn.example.com/episode-assets/werewolf/textures/bg.png'
    );
  });
});

describe('getEpisodeAssetUrl', () => {
  it('builds full URL for an episode asset', () => {
    expect(
      getEpisodeAssetUrl(
        { gameName: 'werewolf', episodeId: '999', baseUrl: 'https://cdn.example.com' },
        'audio_map.json'
      )
    ).toBe('https://cdn.example.com/episode-assets/werewolf/episodes/999/audio_map.json');
  });

  it('strips leading slash from asset path', () => {
    expect(
      getEpisodeAssetUrl(
        { gameName: 'werewolf', episodeId: '999', baseUrl: 'https://cdn.example.com' },
        '/audio_map.json'
      )
    ).toBe('https://cdn.example.com/episode-assets/werewolf/episodes/999/audio_map.json');
  });
});

describe('rebaseAssetPaths', () => {
  const resolvedUrl = 'https://cdn.example.com/episode-assets/werewolf/episodes/123/audio_map.json';

  it('rebases relative paths to absolute URLs', () => {
    const data = { intro: 'audio/intro.wav', outro: 'audio/outro.wav' };
    const result = rebaseAssetPaths(data, resolvedUrl);
    expect(result).toEqual({
      intro: 'https://cdn.example.com/episode-assets/werewolf/episodes/123/audio/intro.wav',
      outro: 'https://cdn.example.com/episode-assets/werewolf/episodes/123/audio/outro.wav',
    });
  });

  it('leaves absolute http URLs unchanged', () => {
    const data = { clip: 'https://other.com/sound.mp3' };
    const result = rebaseAssetPaths(data, resolvedUrl);
    expect(result.clip).toBe('https://other.com/sound.mp3');
  });

  it('leaves root-relative paths (starting with /) unchanged', () => {
    const data = { clip: '/static/sound.mp3' };
    const result = rebaseAssetPaths(data, resolvedUrl);
    expect(result.clip).toBe('/static/sound.mp3');
  });

  it('ignores non-string values', () => {
    const data = { count: 42, enabled: true, nested: { a: 1 } } as any;
    const result = rebaseAssetPaths(data, resolvedUrl);
    expect(result.count).toBe(42);
    expect(result.enabled).toBe(true);
    expect(result.nested).toEqual({ a: 1 });
  });

  it('does not mutate the original object', () => {
    const data = { intro: 'audio/intro.wav' };
    const result = rebaseAssetPaths(data, resolvedUrl);
    expect(result).not.toBe(data);
    expect(data.intro).toBe('audio/intro.wav');
  });

  it('handles empty object', () => {
    expect(rebaseAssetPaths({}, resolvedUrl)).toEqual({});
  });
});

const mockFetch = vi.fn();

beforeEach(() => {
  globalThis.fetch = mockFetch;
});

afterEach(() => {
  mockFetch.mockReset();
});

describe('fetchGameAsset', () => {
  it('returns parsed JSON data and resolved URL on success', async () => {
    const payload = { key: 'value' };
    mockFetch.mockResolvedValue({
      ok: true,
      url: 'https://cdn.example.com/episode-assets/chess/config.json',
      json: () => Promise.resolve(payload),
    });

    const result = await fetchGameAsset({ gameName: 'chess', baseUrl: 'https://cdn.example.com' }, 'config.json');
    expect(result.data).toEqual(payload);
    expect(result.resolvedUrl).toBe('https://cdn.example.com/episode-assets/chess/config.json');
    expect(result.error).toBeNull();
  });

  it('returns error on non-ok response', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
    });

    const result = await fetchGameAsset({ gameName: 'chess', baseUrl: 'https://cdn.example.com' }, 'missing.json');
    expect(result.data).toBeNull();
    expect(result.resolvedUrl).toBeNull();
    expect(result.error).toBeInstanceOf(Error);
    expect(result.error!.message).toContain('404');
  });

  it('returns error on network failure', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'));

    const result = await fetchGameAsset({ gameName: 'chess', baseUrl: 'https://cdn.example.com' }, 'config.json');
    expect(result.data).toBeNull();
    expect(result.error!.message).toBe('Network error');
  });

  it('uses original URL when response.url is a blob URL', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      url: 'blob:https://example.com/abc123',
      json: () => Promise.resolve({}),
    });

    const result = await fetchGameAsset({ gameName: 'chess', baseUrl: 'https://cdn.example.com' }, 'config.json');
    expect(result.resolvedUrl).toBe('https://cdn.example.com/episode-assets/chess/config.json');
  });
});

describe('fetchEpisodeAsset', () => {
  it('fetches from the episode-specific URL', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      url: 'https://cdn.example.com/episode-assets/chess/episodes/42/data.json',
      json: () => Promise.resolve({ ok: true }),
    });

    const result = await fetchEpisodeAsset(
      { gameName: 'chess', episodeId: '42', baseUrl: 'https://cdn.example.com' },
      'data.json'
    );
    expect(result.data).toEqual({ ok: true });
    expect(mockFetch).toHaveBeenCalledWith('https://cdn.example.com/episode-assets/chess/episodes/42/data.json');
  });
});

describe('fetchAndRebaseGameAssetMap', () => {
  it('fetches and rebases relative paths in the result', async () => {
    const payload = { intro: 'audio/intro.wav', logo: 'https://other.com/logo.png' };
    mockFetch.mockResolvedValue({
      ok: true,
      url: 'https://cdn.example.com/episode-assets/chess/asset_map.json',
      json: () => Promise.resolve(payload),
    });

    const result = await fetchAndRebaseGameAssetMap(
      { gameName: 'chess', baseUrl: 'https://cdn.example.com' },
      'asset_map.json'
    );
    expect(result.data).toEqual({
      intro: 'https://cdn.example.com/episode-assets/chess/audio/intro.wav',
      logo: 'https://other.com/logo.png',
    });
  });

  it('returns error result without rebasing on fetch failure', async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 500, statusText: 'Server Error' });

    const result = await fetchAndRebaseGameAssetMap(
      { gameName: 'chess', baseUrl: 'https://cdn.example.com' },
      'asset_map.json'
    );
    expect(result.data).toBeNull();
    expect(result.error).toBeInstanceOf(Error);
  });
});

describe('fetchAndRebaseEpisodeAssetMap', () => {
  it('fetches and rebases relative paths in the result', async () => {
    const payload = { clip: 'sounds/clip.mp3' };
    mockFetch.mockResolvedValue({
      ok: true,
      url: 'https://cdn.example.com/episode-assets/chess/episodes/7/map.json',
      json: () => Promise.resolve(payload),
    });

    const result = await fetchAndRebaseEpisodeAssetMap(
      { gameName: 'chess', episodeId: '7', baseUrl: 'https://cdn.example.com' },
      'map.json'
    );
    expect(result.data).toEqual({
      clip: 'https://cdn.example.com/episode-assets/chess/episodes/7/sounds/clip.mp3',
    });
  });

  it('returns error result without rebasing on fetch failure', async () => {
    mockFetch.mockRejectedValue(new Error('timeout'));

    const result = await fetchAndRebaseEpisodeAssetMap(
      { gameName: 'chess', episodeId: '7', baseUrl: 'https://cdn.example.com' },
      'map.json'
    );
    expect(result.data).toBeNull();
    expect(result.error!.message).toBe('timeout');
  });
});
