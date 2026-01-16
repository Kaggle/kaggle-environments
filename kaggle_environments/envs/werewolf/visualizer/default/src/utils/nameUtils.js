import { disambiguateDisplayNames as disambiguateMap } from '@kaggle-environments/core';

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
