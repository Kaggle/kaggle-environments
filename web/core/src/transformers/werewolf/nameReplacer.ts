/**
 * Shared utility for replacing character names with display names (model names).
 * Used by both the transformer (plain text) and the visualizer (HTML).
 */

export interface PlayerConfig {
  name: string;
  display_name?: string;
  thumbnail?: string;
  [key: string]: any;
}

export type OutputFormat = 'text' | 'html';

/**
 * Disambiguates display names by appending a counter to duplicates.
 * E.g., if two players have display_name "GPT-4", they become "GPT-4 (1)" and "GPT-4 (2)".
 * Mutates the config objects in the map.
 *
 * @param configMap - Map from player ID to player config
 */
export function disambiguateDisplayNames(configMap: Map<string, PlayerConfig>): void {
  const nameCounts = new Map<string, number>();

  // First pass: count all display names
  for (const config of configMap.values()) {
    const name = config.display_name || config.name || '';
    nameCounts.set(name, (nameCounts.get(name) || 0) + 1);
  }

  const currentCounts = new Map<string, number>();

  // Second pass: append suffix if count > 1
  for (const config of configMap.values()) {
    const name = config.display_name || config.name || '';
    if ((nameCounts.get(name) || 0) > 1) {
      const count = (currentCounts.get(name) || 0) + 1;
      currentCounts.set(name, count);
      config.display_name = `${name} (${count})`;
    }
  }
}

/**
 * Creates an HTML capsule for a player (used by visualizer)
 */
export function createPlayerCapsule(player: PlayerConfig): string {
  if (!player) return '';
  const nameToShow = player.display_name || player.name;
  const thumbnailSrc = player.thumbnail || '';
  return `<span class="player-capsule" title="${player.name}">
    <img src="${thumbnailSrc}" class="capsule-avatar" alt="${player.name}" onerror="handleThumbnailError(this)">
    <span class="capsule-name">${nameToShow}</span>
  </span>`;
}

/**
 * Creates a memoized function to replace player names in text.
 *
 * @param playerMap - Map from character name to player config
 * @param format - 'text' for plain text replacement, 'html' for HTML capsules
 * @returns A function that takes text and returns it with names replaced
 */
export function createNameReplacer(
  playerMap: Map<string, PlayerConfig>,
  format: OutputFormat = 'text'
): (text: string) => string {
  // Cache for already processed text strings (memoization)
  const textCache = new Map<string, string>();

  // Pre-compute sorted replacements (longest names first to avoid partial matches)
  const sortedPlayerReplacements = [...playerMap.keys()]
    .sort((a, b) => b.length - a.length)
    .map((characterName) => {
      const player = playerMap.get(characterName);
      if (!player) return null;

      const displayName = player.display_name || characterName;

      // Skip if display name is same as character name
      if (format === 'text' && displayName === characterName) return null;

      // Replacement text depends on format
      const replacement = format === 'html'
        ? createPlayerCapsule(player)
        : displayName;

      // Robust regex that handles edge cases better than simple \b word boundaries
      // - (^|[^\w.-]) - Start of string OR preceded by non-word/dot/hyphen character
      // - (\.?) - Captures optional trailing period (preserved in replacement)
      // - (?![\w-]) - Not followed by word char or hyphen
      const escapedName = characterName.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
      // Update: Handle optional "Player" prefix (e.g. "PlayerKai") by consuming it (non-capturing group)
      const regex = new RegExp(
        `(^|[^\\w.-])(?:Player\\s*)?(${escapedName})(\\.?)(?![\\w-])`,
        'g'
      );

      return { replacement, regex };
    })
    .filter((r): r is NonNullable<typeof r> => r !== null);

  return function replaceNames(text: string): string {
    if (!text) return '';

    if (textCache.has(text)) {
      return textCache.get(text)!;
    }

    let newText = text;
    for (const { replacement, regex } of sortedPlayerReplacements) {
      // $1 = prefix (captured), replacement = the display name/capsule, $3 = optional period
      newText = newText.replace(regex, `$1${replacement}$3`);
    }

    textCache.set(text, newText);
    return newText;
  };
}

