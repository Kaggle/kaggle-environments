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
        if (simplification_map[name]) {
            name = simplification_map[name];
        } else {
            // 2. Regex rules (only apply if map didn't match, or apply on result?)
            // Following add_audio.py logic: apply rules.
            simplification_rules.forEach(rule => {
                if (rule.pattern) {
                    try {
                        const regex = new RegExp(rule.pattern, 'g');
                        name = name.replace(regex, rule.replacement || '');
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
        if (p.display_name && p.display_name.includes('-') && !p.display_name.includes(' ')) {
            p.display_name = p.display_name.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
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
