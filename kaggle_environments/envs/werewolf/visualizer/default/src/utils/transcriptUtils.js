import transcriptConfig from './transcriptConfig.json';

/**
 * Applies transcript overrides to consistent moderator messages.
 * @param {string} text - The original transcript text.
 * @returns {string} The overridden text if a match is found, otherwise the original.
 */
export function applyTranscriptOverrides(text) {
    if (!text) return text;
    const { moderator_overrides = {} } = transcriptConfig;

    // Normalize the input text for robust matching
    const normalize = (s) => (s || '').trim().replace(/\s+([.?!])/g, '$1');
    let result = normalize(text);

    // Sort keys by length descending to prevent partial match collisions
    const sortedKeys = Object.keys(moderator_overrides).sort((a, b) => b.length - a.length);

    for (const key of sortedKeys) {
        const normalizedKey = normalize(key);
        const replacementValue = moderator_overrides[key];

        if (normalizedKey && result.includes(normalizedKey)) {
            // Global replacement of the normalized fragment within the normalized text
            result = result.split(normalizedKey).join(replacementValue);
        }
    }

    return result;
}
