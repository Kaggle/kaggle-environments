// Gin Rummy replay transformer.
//
// Builds the per-step `players` array the side-panel UI needs so the right-hand
// Game Log can render each agent's action label and thoughts. The renderer
// itself still consumes the raw step data via `mergedObservation()`.

interface GinRummyAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface GinRummyReplayPlayer {
  action?: GinRummyAction;
  reward: number;
  observation?: { observationString?: string; isTerminal?: boolean };
  status?: string;
}

interface GinRummyPlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
}

export interface GinRummyStep {
  step: number;
  players: GinRummyPlayer[];
  isTerminal: boolean;
  rawStep: GinRummyReplayPlayer[];
}

const SUIT_GLYPH: Record<string, string> = { s: '\u2660', c: '\u2663', d: '\u2666', h: '\u2665' };

function actionLabel(submission: number | undefined): string {
  if (submission === undefined || submission < 0) return '';
  if (submission === 52) return 'Draw upcard';
  if (submission === 53) return 'Draw stock';
  if (submission === 54) return 'Pass';
  if (submission === 55) return 'Knock';
  if (submission >= 56) return `Meld (action ${submission})`;
  // Single-card action 0-51: derive the card from OpenSpiel's canonical
  // ordering (rank-major within suit s, c, d, h).
  const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'];
  const suits = ['s', 'c', 'd', 'h'];
  const suit = suits[Math.floor(submission / 13)];
  const rank = ranks[submission % 13];
  return `${rank}${SUIT_GLYPH[suit] ?? suit}`;
}

function parseThoughts(action?: GinRummyAction): string {
  // Prefer the Go-harness ``main_response_and_thoughts`` payload when
  // generate_returns is populated; fall through to ``action.thoughts``.
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      // fall through
    }
  }
  return action?.thoughts ?? '';
}

export const ginRummyTransformer = (environment: any): GinRummyStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: GinRummyReplayPlayer[][] = environment?.steps ?? [];
  const out: GinRummyStep[] = [];

  rawSteps.forEach((step, index) => {
    const players: GinRummyPlayer[] = step.map((p, i): GinRummyPlayer => {
      const submission = p.action?.submission;
      const isTurn = submission !== undefined && submission !== -1;
      return {
        id: i,
        name: teamNames[i] ?? (i === 0 ? 'Player 1' : 'Player 2'),
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? actionLabel(submission),
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
      };
    });

    const isTerminal = !!step[0]?.observation?.isTerminal;
    out.push({
      step: index,
      players,
      isTerminal,
      rawStep: step,
    });
  });

  return out;
};
