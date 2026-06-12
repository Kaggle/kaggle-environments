import {
  BargainingObs,
  BargainingPlayer,
  BargainingReplay,
  BargainingStep,
} from './bargainingReplayTypes';

function parseObs(raw?: string): BargainingObs | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as BargainingObs;
  } catch {
    return null;
  }
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
        thoughts: p.action?.thoughts ?? '',
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
