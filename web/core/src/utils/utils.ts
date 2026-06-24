import { BaseGameStep, BaseGamePlayer, ReplayData } from '../types';

/**
 * Returns the player whose turn it is for the given step.
 * Returns undefined for system steps (no player has isTurn) so the UI shows "System".
 */
export function getPlayer(step: BaseGameStep): BaseGamePlayer | undefined {
  const players = step.players;
  if (!players || players.length === 0) {
    return undefined;
  }

  return players.find((p) => p.isTurn);
}

interface AgentLike {
  name?: string;
  index?: number;
}

/**
 * Overrides replay.info.TeamNames[i] with agents[i].name when the host supplies a name.
 *
 * The host (Kaggle.com EpisodesPanel) posts a fresh `agents` list whose name reflects
 * current DB state (BenchmarkModelVersion.DisplayName, falling back to team_name). The
 * replay's baked-in info.TeamNames is frozen at game-creation time and goes stale when
 * a team is later linked to a benchmark model or renamed — the host-supplied label is
 * the source of truth. Returns the original replay reference when no override applies
 * so downstream reference-equality checks keep working.
 */
export function applyAgentNamesToReplay<T extends ReplayData<any>>(
  replay: T,
  agents: (AgentLike | undefined | null)[] | undefined | null
): T {
  if (!replay || !agents || agents.length === 0) return replay;

  const existing: string[] = Array.isArray(replay.info?.TeamNames) ? (replay.info!.TeamNames as string[]) : [];
  const merged: string[] = [...existing];
  let changed = false;

  agents.forEach((agent, i) => {
    const idx = typeof agent?.index === 'number' ? agent.index : i;
    const name = agent?.name;
    if (typeof name === 'string' && name.length > 0 && merged[idx] !== name) {
      merged[idx] = name;
      changed = true;
    }
  });

  if (!changed) return replay;

  return {
    ...replay,
    info: { ...(replay.info ?? {}), TeamNames: merged },
  };
}
