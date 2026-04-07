/**
 * Asset Utilities - Global utility for non-bundled asset routing
 *
 * This utility helps visualizers fetch assets (audio, images, etc.)
 * that are stored separately from the bundled visualizer files.
 *
 * Two types of assets are supported:
 * - Game-specific:    /episode-assets/{gameName}/...
 * - Episode-specific: /episode-assets/{gameName}/episodes/{episodeId}/...
 *
 * This path is separate from the visualizer bundle path to avoid conflicts with
 * the rsync deployment which deletes files not in the build directory.
 */

export interface GameAssetsConfig {
  /** The game name (e.g., 'werewolf', 'chess') */
  gameName: string;
  /** Optional custom base URL for assets (defaults to origin) */
  baseUrl?: string;
}

export interface EpisodeAssetsConfig extends GameAssetsConfig {
  /** The episode ID */
  episodeId: string;
}

export interface AssetFetchResult<T> {
  data: T | null;
  resolvedUrl: string | null;
  error: Error | null;
}

/**
 * Default base path for assets in GCS
 */
const ASSETS_BASE_PATH = '/episode-assets';

/**
 * Builds the base URL for game-specific assets.
 *
 * @param config - Configuration for the game assets
 * @returns The base URL for the game's assets
 *
 * @example
 * ```ts
 * const baseUrl = getGameAssetsBaseUrl({ gameName: 'werewolf' });
 * // Returns: 'https://example.com/episode-assets/werewolf'
 * ```
 */
export function getGameAssetsBaseUrl(config: GameAssetsConfig): string {
  const origin = config.baseUrl || (typeof window !== 'undefined' ? window.location.origin : '');
  return `${origin}${ASSETS_BASE_PATH}/${config.gameName}`;
}

/**
 * Builds the base URL for episode-specific assets.
 *
 * @param config - Configuration for the episode assets
 * @returns The base URL for the episode's assets
 *
 * @example
 * ```ts
 * const baseUrl = getEpisodeAssetsBaseUrl({
 *   gameName: 'werewolf',
 *   episodeId: '12345'
 * });
 * // Returns: 'https://example.com/episode-assets/werewolf/episodes/12345'
 * ```
 */
export function getEpisodeAssetsBaseUrl(config: EpisodeAssetsConfig): string {
  const gameBaseUrl = getGameAssetsBaseUrl(config);
  return `${gameBaseUrl}/episodes/${config.episodeId}`;
}

/**
 * Builds a full URL for a specific asset within a game's assets directory.
 *
 * @param config - Configuration for the game assets
 * @param assetPath - Relative path to the asset (e.g., 'config.json', 'textures/bg.png')
 * @returns The full URL to the asset
 *
 * @example
 * ```ts
 * const url = getGameAssetUrl(
 *   { gameName: 'werewolf' },
 *   'config.json'
 * );
 * // Returns: 'https://example.com/episode-assets/werewolf/config.json'
 * ```
 */
export function getGameAssetUrl(config: GameAssetsConfig, assetPath: string): string {
  const baseUrl = getGameAssetsBaseUrl(config);
  const normalizedPath = assetPath.startsWith('/') ? assetPath.slice(1) : assetPath;
  return `${baseUrl}/${normalizedPath}`;
}

/**
 * Builds a full URL for a specific asset within an episode's assets directory.
 *
 * @param config - Configuration for the episode assets
 * @param assetPath - Relative path to the asset (e.g., 'audio_map.json', 'audio/clip.wav')
 * @returns The full URL to the asset
 *
 * @example
 * ```ts
 * const url = getEpisodeAssetUrl(
 *   { gameName: 'werewolf', episodeId: '12345' },
 *   'audio_map.json'
 * );
 * // Returns: 'https://example.com/episode-assets/werewolf/episodes/12345/audio_map.json'
 * ```
 */
export function getEpisodeAssetUrl(config: EpisodeAssetsConfig, assetPath: string): string {
  const baseUrl = getEpisodeAssetsBaseUrl(config);
  const normalizedPath = assetPath.startsWith('/') ? assetPath.slice(1) : assetPath;
  return `${baseUrl}/${normalizedPath}`;
}

/**
 * Internal helper to fetch a JSON asset from a URL.
 */
async function fetchAssetFromUrl<T>(url: string): Promise<AssetFetchResult<T>> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    // Use response.url if available (handles redirects), otherwise use the original URL
    const resolvedUrl = response.url && !response.url.includes('blob:') ? response.url : url;

    return { data, resolvedUrl, error: null };
  } catch (error) {
    return { data: null, resolvedUrl: null, error: error as Error };
  }
}

/**
 * Fetches a JSON asset from the game assets directory.
 * Returns both the data and the resolved URL for further path resolution.
 *
 * @param config - Configuration for the game assets
 * @param assetPath - Relative path to the JSON asset
 * @returns Promise with the fetched data, resolved URL, and any error
 *
 * @example
 * ```ts
 * const result = await fetchGameAsset<GameConfig>(
 *   { gameName: 'werewolf' },
 *   'config.json'
 * );
 * if (result.data) {
 *   console.log('Loaded game config:', result.data);
 * }
 * ```
 */
export async function fetchGameAsset<T>(config: GameAssetsConfig, assetPath: string): Promise<AssetFetchResult<T>> {
  const url = getGameAssetUrl(config, assetPath);
  return fetchAssetFromUrl<T>(url);
}

/**
 * Fetches a JSON asset from the episode assets directory.
 * Returns both the data and the resolved URL for further path resolution.
 *
 * @param config - Configuration for the episode assets
 * @param assetPath - Relative path to the JSON asset
 * @returns Promise with the fetched data, resolved URL, and any error
 *
 * @example
 * ```ts
 * const result = await fetchEpisodeAsset<AudioMap>(
 *   { gameName: 'werewolf', episodeId: '12345' },
 *   'audio_map.json'
 * );
 * if (result.data) {
 *   console.log('Loaded audio map:', result.data);
 * }
 * ```
 */
export async function fetchEpisodeAsset<T>(
  config: EpisodeAssetsConfig,
  assetPath: string
): Promise<AssetFetchResult<T>> {
  const url = getEpisodeAssetUrl(config, assetPath);
  return fetchAssetFromUrl<T>(url);
}

/**
 * Given a resolved asset URL (like audio_map.json), rebases relative paths
 * within the data to be absolute URLs relative to the asset's directory.
 *
 * This is useful when an asset map contains relative paths that need to
 * be resolved against the map's location.
 *
 * @param data - Object containing string values that may be relative paths
 * @param resolvedUrl - The resolved URL of the asset (for determining base directory)
 * @returns The same object with paths rebased to absolute URLs
 *
 * @example
 * ```ts
 * const audioMap = { intro: 'audio/intro.wav', outro: 'audio/outro.wav' };
 * const rebased = rebaseAssetPaths(audioMap, 'https://example.com/episode-assets/werewolf/episodes/123/audio_map.json');
 * // Returns: { intro: 'https://example.com/episode-assets/werewolf/episodes/123/audio/intro.wav', ... }
 * ```
 */
export function rebaseAssetPaths<T extends Record<string, unknown>>(data: T, resolvedUrl: string): T {
  const baseDir = resolvedUrl.substring(0, resolvedUrl.lastIndexOf('/') + 1);

  const result = { ...data };
  for (const key in result) {
    const value = result[key];
    if (typeof value === 'string' && !value.startsWith('http') && !value.startsWith('/')) {
      (result as Record<string, unknown>)[key] = baseDir + value;
    }
  }

  return result;
}

/**
 * Convenience function to fetch and rebase a game asset map in one call.
 *
 * @param config - Configuration for the game assets
 * @param assetPath - Relative path to the asset map JSON
 * @returns Promise with the fetched and rebased data
 */
export async function fetchAndRebaseGameAssetMap<T extends Record<string, unknown>>(
  config: GameAssetsConfig,
  assetPath: string
): Promise<AssetFetchResult<T>> {
  const result = await fetchGameAsset<T>(config, assetPath);

  if (result.data && result.resolvedUrl) {
    return {
      ...result,
      data: rebaseAssetPaths(result.data, result.resolvedUrl),
    };
  }

  return result;
}

/**
 * Convenience function to fetch and rebase an episode asset map in one call.
 *
 * @param config - Configuration for the episode assets
 * @param assetPath - Relative path to the asset map JSON
 * @returns Promise with the fetched and rebased data
 */
export async function fetchAndRebaseEpisodeAssetMap<T extends Record<string, unknown>>(
  config: EpisodeAssetsConfig,
  assetPath: string
): Promise<AssetFetchResult<T>> {
  const result = await fetchEpisodeAsset<T>(config, assetPath);

  if (result.data && result.resolvedUrl) {
    return {
      ...result,
      data: rebaseAssetPaths(result.data, result.resolvedUrl),
    };
  }

  return result;
}
