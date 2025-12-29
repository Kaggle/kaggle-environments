/**
 * Disambiguates display names by appending a counter to duplicates.
 * @param {Array<object>} players - List of player objects.
 */
export function disambiguateDisplayNames(players) {
    const nameCounts = new Map();
    
    // First pass: count all display names
    players.forEach(p => {
        const name = p.display_name || p.name;
        nameCounts.set(name, (nameCounts.get(name) || 0) + 1);
    });

    const currentCounts = new Map();

    // Second pass: append suffix if count > 1
    players.forEach(p => {
        const name = p.display_name || p.name;
        if (nameCounts.get(name) > 1) {
            const count = (currentCounts.get(name) || 0) + 1;
            currentCounts.set(name, count);
            p.display_name = `${name} (${count})`;
        }
    });
}
