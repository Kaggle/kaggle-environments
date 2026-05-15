import { BaseGameStep } from '@kaggle-environments/core';

type Notation = { term: string; title: string; text: string };

const LONG_THINKING_MINUTES = 5;

const THOUGHT_ANNOTATIONS: { term: string; label: string | null; description: string }[] = [
  { term: 'rethink', label: null, description: 'rethinks their decision.' },
  { term: "boden's mate", label: "Boden's mate", description: 'Checkmated by two criss-crossed bishops.' },
  { term: 'castling', label: 'Castling', description: 'Strategically moving a king and rook at the same time.' },
  {
    term: 'discovered attack',
    label: 'Discovered Attack',
    description: 'Attacking after a piece is moved out of the way.',
  },
  { term: 'gambit', label: 'Gambit', description: 'Sacrificing a piece for early game advantage.' },
  { term: 'interpose', label: 'Interpose', description: 'Moving a piece between an attacking piece and its target.' },
  { term: 'mating attack', label: 'Mating Attack', description: 'A move that aims to achieve checkmate.' },
  { term: 'royal fork', label: 'Royal Fork', description: 'A fork that threatens the king and queen.' },
  {
    term: 'stalemate',
    label: 'Stalemate',
    description: 'A draw when the active player has no legal moves and their king is not in check.',
  },
  { term: 'zugzwang', label: 'Zugzwang', description: 'Forced to move when any movement weakens your position.' },
];

function formatThoughtMatch(entry: (typeof THOUGHT_ANNOTATIONS)[number], agent: string): Notation {
  return {
    term: entry.term,
    title: entry.label ? `${agent} mentions ${entry.label}:` : '',
    text: entry.label ? entry.description : `${agent} ${entry.description}`,
  };
}

function getThinkingDurationMinutes(generateReturns: string[] | null | undefined): number | null {
  const json = generateReturns?.[0];
  if (!json) return null;
  try {
    return JSON.parse(json).duration_success_only_secs / 60;
  } catch {
    return null;
  }
}

function resolveAnnotation(
  player: { name: string; thoughts?: string; generateReturns?: string[] | null }
): Notation | null {
  const agent = player.name;

  const thoughts = player.thoughts?.toLowerCase();
  if (thoughts) {
    const match = THOUGHT_ANNOTATIONS.find((a) => thoughts.includes(a.term));
    if (match) return formatThoughtMatch(match, agent);
  }

  const duration = getThinkingDurationMinutes(player.generateReturns);
  if (duration != null && duration > LONG_THINKING_MINUTES) {
    return { term: 'duration', title: '', text: `${agent} thought for over ${LONG_THINKING_MINUTES} minutes.` };
  }

  return null;
}

export function getStepDescription(step: BaseGameStep) {
  const player = step.players.find(p => p.isTurn);
  if (!player?.thoughts) return '';

  const annotation = resolveAnnotation(player);
  if (!annotation) return player.thoughts;

  // Note: the thoughts are in markdown, formatted with react-markdown.
  // react-markdown implements only the Commonmark formatting, so no 
  // extended or Github Flavoured markdown works, also no HTML works.
  // So this is pretty limited.
  // Then, even some of these styles don't work (eg. italics) because although 
  // <em> tags are added to the content by the parser, the css is a bit of a 
  // tangle and it's displayed without italicisation.
  const re = new RegExp(annotation.term, 'gi');
  const thoughts = player.thoughts.replace(re, `*${annotation.term}*`);

  return thoughts;
}
