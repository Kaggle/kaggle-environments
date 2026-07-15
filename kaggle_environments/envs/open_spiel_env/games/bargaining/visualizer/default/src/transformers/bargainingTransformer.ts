import { BargainingObs, BargainingPlayer, BargainingReplay, BargainingStep } from './bargainingReplayTypes';

function parseObs(raw?: string): BargainingObs | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as BargainingObs;
  } catch {
    return null;
  }
}

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
function parseThoughts(action?: { thoughts?: string | null; generate_returns?: string[] | null }): string {
  if (action?.thoughts) return action.thoughts;
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) return parsed.main_response_and_thoughts;
    } catch {
      // fall through
    }
  }
  return '';
}

// Submission is -1 (or null) on setup/inactive turns; treat anything else as a real move.
const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const bargainingTransformer = (environment: any): BargainingStep[] => {
  const replay = environment as BargainingReplay;
  const agents = replay.info?.TeamNames ?? [];

  return replay.steps.map((step, index): BargainingStep => {
    const obs0 = parseObs(step?.[0]?.observation?.observationString);
    const obs1 = parseObs(step?.[1]?.observation?.observationString);
    const obs = obs0 ?? obs1;

    const players: BargainingPlayer[] = step.map((p, pi): BargainingPlayer => {
      const sub = p.action?.submission;
      return {
        id: pi,
        name: agents[pi] || `Player ${pi + 1}`,
        thumbnail: '',
        isTurn: isRealMove(sub),
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward,
      };
    });

    return {
      step: index,
      players,
      observations: [obs0, obs1],
      obs,
      isTerminal: !!obs?.is_terminal,
    };
  });
};
