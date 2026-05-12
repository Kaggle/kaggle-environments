/**
 * Maps substrings in team names (e.g. "Claude Opus 4.6") to logo keys.
 * Kaggle replays use model marketing names as TeamNames, so we match on
 * brand keywords to resolve the correct company logo. Falls back to `unknown`
 * logo if no keywords are found.
 */
const KEYWORD_TO_LOGO: Record<string, string> = {
  claude: 'anthropic',
  deepseek: 'deepseek',
  gemini: 'gemini',
  grok: 'grok',
  'gpt-': 'openai',
  o3: 'openai',
  o4: 'openai',
};

export function getAgentBrand(teamName: string): string | null {
  const lower = teamName.toLowerCase();
  const key = Object.keys(KEYWORD_TO_LOGO).find((k) => lower.includes(k));
  return key ? KEYWORD_TO_LOGO[key] : null;
}
