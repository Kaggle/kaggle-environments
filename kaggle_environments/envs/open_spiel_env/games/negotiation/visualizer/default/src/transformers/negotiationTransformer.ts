// Negotiation replay transformer.
//
// Populates `players[]` so the sidebar can show each mover's thoughts and
// their proposal/utterance action. Negotiation is a private-information
// game — each seat's observation shows different utilities, so `boardState`
// keeps both players' pre-parsed observations side by side.

interface NegotiationAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  generate_returns?: string[] | null;
}

interface NegotiationReplayPlayer {
  action?: NegotiationAction;
  reward: number;
  observation: { observationString?: string };
  status?: string;
}

interface NegotiationPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
}

export interface NegotiationObs {
  current_player: number;
  viewing_player: number;
  turn_type: 'proposal' | 'utterance' | null;
  max_steps: number;
  item_pool: number[];
  my_utilities: number[];
  proposals: Array<{ player: number; items?: number[]; accept: boolean }>;
  utterances: Array<{ player: number; symbols: number[] }>;
  most_recent_proposal: number[] | null;
  most_recent_utterance: number[] | null;
  agreement_reached: boolean;
  is_terminal: boolean;
  winner: number | 'draw' | null;
  rewards: number[] | null;
  params: {
    num_items: number;
    num_symbols: number;
    utterance_dim: number;
    enable_utterances: boolean;
  };
}

export interface NegotiationStep {
  step: number;
  players: NegotiationPlayer[];
  // Per-seat observations — both are needed because utilities are private.
  boardState: {
    perPlayer: [NegotiationObs | null, NegotiationObs | null];
  } | null;
}

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
function parseThoughts(action?: NegotiationAction): string {
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

function parseObs(raw?: string): NegotiationObs | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as NegotiationObs;
  } catch {
    return null;
  }
}

const isRealMove = (submission: unknown): boolean =>
  submission !== undefined && submission !== null && submission !== -1;

export const negotiationTransformer = (environment: any): NegotiationStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? [];
  const rawSteps: NegotiationReplayPlayer[][] = environment?.steps ?? [];

  return rawSteps.map((step, index): NegotiationStep => {
    const players: NegotiationPlayer[] = step.map(
      (p, i): NegotiationPlayer => ({
        id: i,
        name: teamNames[i] || `Player ${i + 1}`,
        thumbnail: '',
        isTurn: isRealMove(p.action?.submission),
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
      })
    );

    const obs0 = parseObs(step?.[0]?.observation?.observationString);
    const obs1 = parseObs(step?.[1]?.observation?.observationString);

    return {
      step: index,
      players,
      boardState: obs0 || obs1 ? { perPlayer: [obs0, obs1] } : null,
    };
  });
};
