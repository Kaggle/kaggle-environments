import anthropicBlack from '../assets/agent-logos/anthropic-black.webp';
import anthropicWhite from '../assets/agent-logos/anthropic-white.webp';
import deepseekBlack from '../assets/agent-logos/deepseek-black.webp';
import deepseekWhite from '../assets/agent-logos/deepseek-white.webp';
import geminiBlack from '../assets/agent-logos/gemini-black.webp';
import geminiWhite from '../assets/agent-logos/gemini-white.webp';
import grokBlack from '../assets/agent-logos/grok-black.webp';
import grokWhite from '../assets/agent-logos/grok-white.webp';
import openaiBlack from '../assets/agent-logos/openai-black.webp';
import openaiWhite from '../assets/agent-logos/openai-white.webp';
import unknownWhite from '../assets/agent-logos/unknown-white.webp';
import unknownBlack from '../assets/agent-logos/unknown-black.webp';

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

export interface AgentLogo {
  src: string;
  isUnknown: boolean;
}

export function getAgentLogo(teamName: string, color: 'black' | 'white'): AgentLogo {
  const lower = teamName.toLowerCase();
  for (const [keyword, logo] of KEYWORD_TO_LOGO) {
    if (lower.includes(keyword)) {
      return { src: LOGOS[logo][color], isUnknown: false };
    }
  }
  return { src: LOGOS.unknown[color], isUnknown: true };
}
