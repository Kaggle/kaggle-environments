import { randomAgent } from './random';
import { starterAgent } from './starter';
import type { AgentFn } from './types';

export type { AgentFn, Observation } from './types';
export { randomAgent } from './random';
export { starterAgent } from './starter';

export const AGENTS: Record<string, { label: string; fn: AgentFn }> = {
  random: { label: 'Random', fn: randomAgent },
  starter: { label: 'Starter', fn: starterAgent },
};

export type AgentId = keyof typeof AGENTS & string;
export const DEFAULT_AGENT_ID: AgentId = 'starter';
