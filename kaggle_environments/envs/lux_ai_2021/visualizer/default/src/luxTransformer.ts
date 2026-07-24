import { ReplayData } from '@kaggle-environments/core';

/**
 * Normalize per-team actions down to a bare list of Lux command strings so the
 * external `lux-viewer-2021` board renderer can consume them.
 *
 * Background: the env's `action` spec was widened (env commit "Persist LLM
 * thoughts in Lux AI 2021 replays") so LLM-harness agents now emit a dict
 * wrapper -- `{ submission, thoughts, actionString, call_details,
 * generate_returns }` -- rather than a bare `string[]`. That wrapper is what
 * carries the thoughts into the replay. The external viewer, however, parses
 * each step with `action.map(cmd => ...)`, which throws `TypeError:
 * action.map is not a function` on the dict shape, breaking exactly the LLM
 * replays we care about.
 *
 * This mirrors the Python `_command_list` projection in
 * `kaggle_environments/envs/lux_ai_2021/lux_ai_2021.py`: legacy bot agents send
 * a bare list (left untouched), harness agents send `{ submission: [...] }`
 * (projected to the list), and anything else becomes an empty command list. The
 * dict's other fields (thoughts, etc.) are simply dropped for the viewer -- the
 * external renderer has no thoughts UI, and the raw thoughts remain in the
 * persisted replay for other consumers.
 */

function commandList(action: unknown): string[] {
  if (Array.isArray(action)) {
    return action as string[];
  }
  if (action && typeof action === 'object') {
    const submission = (action as { submission?: unknown }).submission;
    return Array.isArray(submission) ? (submission as string[]) : [];
  }
  return [];
}

export function luxNormalizeReplay(replay: ReplayData): ReplayData {
  if (!replay || !Array.isArray(replay.steps)) {
    return replay;
  }

  const steps = (replay.steps as any[]).map((step) => {
    if (!Array.isArray(step)) {
      return step;
    }
    return step.map((teamEntry) => {
      if (!teamEntry || typeof teamEntry !== 'object') {
        return teamEntry;
      }
      return { ...teamEntry, action: commandList(teamEntry.action) };
    });
  });

  return { ...replay, steps } as ReplayData;
}
