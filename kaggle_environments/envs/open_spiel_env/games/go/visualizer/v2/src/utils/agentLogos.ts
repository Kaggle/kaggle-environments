import anthropicBlack from '../assets/agent-logos/anthropic-black.png';
import anthropicWhite from '../assets/agent-logos/anthropic-white.png';
import deepseekBlack from '../assets/agent-logos/deepseek-black.png';
import deepseekWhite from '../assets/agent-logos/deepseek-white.png';
import geminiBlack from '../assets/agent-logos/gemini-black.png';
import geminiWhite from '../assets/agent-logos/gemini-white.png';
import grokBlack from '../assets/agent-logos/grok-black.png';
import grokWhite from '../assets/agent-logos/grok-white.png';
import openaiBlack from '../assets/agent-logos/openai-black.png';
import openaiWhite from '../assets/agent-logos/openai-white.png';
import unknownWhite from '../assets/agent-logos/unknown-white.png';
import unknownBlack from '../assets/agent-logos/unknown-black.png';

const LOGOS: Record<string, { black: string; white: string }> = {
  anthropic: { black: anthropicBlack, white: anthropicWhite },
  deepseek: { black: deepseekBlack, white: deepseekWhite },
  gemini: { black: geminiBlack, white: geminiWhite },
  grok: { black: grokBlack, white: grokWhite },
  openai: { black: openaiBlack, white: openaiWhite },
  unknown: { black: unknownBlack, white: unknownWhite },
};

/**
 * Maps substrings in team names (e.g. "Claude Opus 4.6") to logo keys.
 * Kaggle replays use model marketing names as TeamNames, so we match on
 * brand keywords to resolve the correct company logo. Falls back to `unknown`
 * logo if no keywords are found.
 */
const KEYWORD_TO_LOGO: [string, string][] = [
  ['claude', 'anthropic'],
  ['deepseek', 'deepseek'],
  ['gemini', 'gemini'],
  ['grok', 'grok'],
  ['gpt-', 'openai'],
  ['o3', 'openai'],
  ['o4', 'openai'],
];

export function getLogoSrc(teamName: string, color: 'black' | 'white'): string {
  const lower = teamName.toLowerCase();
  for (const [keyword, logo] of KEYWORD_TO_LOGO) {
    if (lower.includes(keyword)) {
      return LOGOS[logo][color];
    }
  }
  return LOGOS.unknown[color];
}
