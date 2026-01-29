import { disambiguateDisplayNames as disambiguateMap } from '@kaggle-environments/core';

import nameConfig from './nameConfig.json';

/**
 * Simplifies display names based on a predefined map or pattern in nameConfig.json.
 * @param {Array<object>} players - List of player objects.
 */
export function simplifyDisplayNames(players) {
    const { simplification_map = {}, simplification_rules = [] } = nameConfig;

    players.forEach(p => {
        let name = p.display_name || p.name;
        if (!name) return;

        const originalName = name;

        // 1. Direct map lookup
        let mapped = false;
        if (simplification_map[name]) {
            name = simplification_map[name];
            mapped = true;
        } else {
            // 2. Regex rules
            simplification_rules.forEach(rule => {
                if (rule.pattern) {
                    try {
                        const regex = new RegExp(rule.pattern, 'g');
                        const newName = name.replace(regex, rule.replacement || '');
                        if (newName !== name) {
                            name = newName;
                            // regex change shouldn't necessarily block title casing
                        }
                    } catch (e) {
                        console.warn(`Invalid regex pattern: ${rule.pattern}`, e);
                    }
                }
            });
        }

        if (name !== originalName) {
            p.display_name = name.trim();
        }

        // 3. Fallback: Title case for dash-separated names if still has dashes
        // Only do this if NOT mapped (preserve map casing/punctuation) and if it looks like a raw ID
        // Fix: Be more restrictive - only mutate if it looks like a clean lowercase slug 
        // (no dots, no numbers, no caps already present)
        if (!mapped && p.display_name && p.display_name.includes('-') && !p.display_name.includes(' ')) {
            const isRawSlug = /^[a-z0-9-]+$/.test(p.display_name) && !/\d/.test(p.display_name); // Simplified check: no numbers in soft names?
            // Actually, let's just check for dots. Dots almost always imply a version or specific technical name.
            const hasDots = p.display_name.includes('.');

            if (!hasDots) {
                p.display_name = p.display_name.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
            }
        }
    });
}

/**
 * Disambiguates display names by appending a counter to duplicates.
 * Wrapper around core function that accepts an array (for visualizer compatibility).
 * @param {Array<object>} players - List of player objects with name/display_name properties.
 */
export function disambiguateDisplayNames(players) {
    // Convert array to map format expected by core function
    const playerMap = new Map();
    players.forEach((p, i) => {
        playerMap.set(p.name || `player_${i}`, p);
    });

    // Use core function - changes are applied in-place to the player objects
    disambiguateMap(playerMap);
}

/**
 * Augments the player map with additional keys for name variations.
 * Specifically, checks nameConfig for any keys that map to the player's simplified display name.
 * Also adds variations of the slug ID (Title Cased, Spaced, etc).
 * @param {Map<string, object>} playerMap - The map to populate.
 * @param {Array<object>} players - The list of players.
 */
export function augmentPlayerMapWithVariations(playerMap, players) {
    const { simplification_map = {} } = nameConfig;

    // 1. Build Reverse Map (Simplified -> [Originals])
    const reverseMap = new Map();
    Object.entries(simplification_map).forEach(([original, simplified]) => {
        const s = simplified.trim();
        if (!reverseMap.has(s)) reverseMap.set(s, []);
        reverseMap.get(s).push(original);
    });

    players.forEach(p => {
        if (!p) return;
        let displayName = p.display_name; // e.g. "Grok 4.1 Fast (2)"
        if (!displayName) return;

        // Strip disambiguation suffix for reverse lookup
        // e.g. "Grok 4.1 Fast (2)" -> "Grok 4.1 Fast"
        const baseDisplayName = displayName.replace(/\s*\(\d+\)$/, '');

        // A. Add reverse simplification matches
        // e.g. "Grok 4.1 Fast" (base) -> add "Grok 4.1 Fast Reasoning"
        if (reverseMap.has(baseDisplayName)) {
            reverseMap.get(baseDisplayName).forEach(original => {
                if (!playerMap.has(original)) {
                    playerMap.set(original, p);
                }
            });
        }

        // B. Add variations of name/ID
        if (p.name && typeof p.name === 'string') {
            const raw = p.name;
            const parts = raw.split(/[-_]/);
            if (parts.length > 1) {
                // 1. Title Cased version: "grok-4.1-fast" -> "Grok 4.1 Fast"
                const titleCased = parts.map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                if (!playerMap.has(titleCased)) playerMap.set(titleCased, p);

                // 2. Dasherized version of the Title Cased name: "Grok 4.1 Fast" -> "Grok-4.1-Fast"
                // This covers cases where engine logs Title-Cased-With-Dashes
                const dasherized = titleCased.replace(/\s+/g, '-');
                if (!playerMap.has(dasherized)) playerMap.set(dasherized, p);
            }
        }

        // C. Add variation for disambiguated names: "Name (2)" -> "Name 2"
        // Some engine logs might output "Name 2" instead of "Name (2)"
        if (displayName && displayName.match(/\(\d+\)$/)) {
            const noParens = displayName.replace(/\((\d+)\)$/, '$1');
            if (!playerMap.has(noParens)) {
                playerMap.set(noParens, p);
            }
        }

        // D. Special Case: Ensure GPT-5.2 is always mapped if it's in the display name
        if (displayName.includes('GPT-5.2') && !playerMap.has('GPT-5.2')) {
            playerMap.set('GPT-5.2', p);
        }
    });
}
