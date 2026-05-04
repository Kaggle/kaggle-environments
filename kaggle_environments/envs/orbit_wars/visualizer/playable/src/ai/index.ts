import { agent4Agent } from './agent4';
import { next7Agent } from './next7';
import { randomAgent } from './random';
import { starterAgent } from './starter';
import type { AgentFn } from './types';

export const AGENTS: Record<string, { label: string; fn: AgentFn }> = {
  random: { label: 'Random', fn: randomAgent },
  starter: { label: 'Starter', fn: starterAgent },
  easy: { label: 'Easy', fn: next7Agent },
  medium: { label: 'Medium', fn: agent4Agent },
};

export type AgentId = keyof typeof AGENTS & string;
export const DEFAULT_AGENT_ID: AgentId = 'medium';
