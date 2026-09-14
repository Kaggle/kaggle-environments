// Shared parsing for the Hanabi proxy's JSON observations.
//
// Hanabi is imperfect-information *in a very specific way*: a player sees
// every hand except their own. So no single observation contains the whole
// truth, and a spectator view has to merge them -- player p's real cards come
// from any observation whose observer is not p. Hint knowledge (what each
// holder has been told) is public and identical in every observation, so it
// can be taken from whichever observation we happen to read.
//
// Both the renderer and the transformer need this merge, plus the action-string
// decoding, so it lives here rather than being duplicated.

import { OpenSpielRawPlayer } from '@kaggle-environments/core';

export interface HanabiCardFace {
  color: string;
  rank: number;
}

export interface HanabiCard {
  /** The real card, or null when the reading observer is its holder. */
  card: HanabiCardFace | null;
  /** What the holder has been told out loud. */
  hinted_color: string | null;
  hinted_rank: number | null;
  /** What the holder can still deduce, including negative inference. */
  plausible_colors: string[];
  plausible_ranks: number[];
}

export interface HanabiHand {
  player: number;
  is_observer: boolean;
  cards: HanabiCard[];
}

export interface HanabiObservation {
  num_players: number;
  colors: number;
  ranks: number;
  hand_size: number;
  observer: number;
  current_player: number;
  life_tokens: number;
  max_life_tokens: number;
  info_tokens: number;
  max_info_tokens: number;
  fireworks: Record<string, number>;
  /** Banked score -- zero once the last life is lost, matching the engine. */
  score: number;
  /** Stack heights regardless of lives; equals `score` until a bomb-out. */
  fireworks_total: number;
  max_score: number;
  hands: HanabiHand[];
  deck_size: number;
  discards: HanabiCardFace[];
  is_terminal: boolean;
  winner: null;
  outcome: string;
  returns: number[];
  legal_actions: { action: number; label: string }[];
  move_number: number;
}

// OpenSpiel's color letters, in firework-display order.
export const COLOR_ORDER = ['R', 'Y', 'G', 'W', 'B'];

export const COLOR_NAMES: Record<string, string> = {
  R: 'red',
  Y: 'yellow',
  G: 'green',
  W: 'white',
  B: 'blue',
};

// Ink colors for each suit. White is drawn as a mid gray so it stays legible
// against the white card stock.
export const COLOR_INK: Record<string, string> = {
  R: '#c8384f',
  Y: '#c99700',
  G: '#3f8f4f',
  W: '#8b8b8b',
  B: '#2b6cb0',
};

export const COLOR_TINT: Record<string, string> = {
  R: '#fbe3e7',
  Y: '#fbf1cf',
  G: '#e0f2e2',
  W: '#efefef',
  B: '#dfeaf8',
};

export function colorName(letter: string): string {
  return COLOR_NAMES[letter] ?? letter;
}

export function parseObservation(step: OpenSpielRawPlayer[] | undefined, playerIdx: number): HanabiObservation | null {
  const raw = step?.[playerIdx]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as HanabiObservation;
  } catch {
    return null;
  }
}

/**
 * A spectator view: the public state plus every player's real hand.
 *
 * Returns null before the deal, when no observation has been emitted yet.
 */
export function mergedObservation(step: OpenSpielRawPlayer[] | undefined): HanabiObservation | null {
  if (!Array.isArray(step)) return null;
  const views: HanabiObservation[] = [];
  for (let i = 0; i < step.length; i++) {
    const obs = parseObservation(step, i);
    if (obs) views.push(obs);
  }
  if (!views.length) return null;

  const merged: HanabiObservation = JSON.parse(JSON.stringify(views[0]));
  for (const hand of merged.hands) {
    if (!hand.is_observer) continue;
    // These cards are hidden from views[0]'s observer; find a view that sees them.
    const revealing = views.find((v) => v.observer !== hand.player);
    const revealed = revealing?.hands.find((h) => h.player === hand.player);
    if (!revealed) continue;
    hand.cards.forEach((card, i) => {
      card.card = revealed.cards[i]?.card ?? null;
    });
  }
  return merged;
}

export type HanabiMove =
  | { type: 'play'; slot: number }
  | { type: 'discard'; slot: number }
  | { type: 'hint'; target: number; color: string }
  | { type: 'hint'; target: number; rank: number };

const PLAY_RE = /^\(Play (\d+)\)$/;
const DISCARD_RE = /^\(Discard (\d+)\)$/;
const HINT_COLOR_RE = /^\(Reveal player \+(\d+) color (\S+)\)$/;
const HINT_RANK_RE = /^\(Reveal player \+(\d+) rank (\d+)\)$/;

/**
 * Decode OpenSpiel's action string into a structured move.
 *
 * Hint actions name their target *relative* to the actor ("player +1"), so the
 * actor index and player count are needed to resolve an absolute seat.
 */
export function parseMove(
  actionString: string | null | undefined,
  actor: number,
  numPlayers: number
): HanabiMove | null {
  if (!actionString) return null;
  let m = PLAY_RE.exec(actionString);
  if (m) return { type: 'play', slot: Number(m[1]) };
  m = DISCARD_RE.exec(actionString);
  if (m) return { type: 'discard', slot: Number(m[1]) };
  m = HINT_COLOR_RE.exec(actionString);
  if (m) return { type: 'hint', target: (actor + Number(m[1])) % numPlayers, color: m[2] };
  m = HINT_RANK_RE.exec(actionString);
  if (m) return { type: 'hint', target: (actor + Number(m[1])) % numPlayers, rank: Number(m[2]) };
  return null;
}

/** Index of the player who acted on this step, or -1 for setup / chance steps. */
export function actingPlayer(step: OpenSpielRawPlayer[] | undefined): number {
  if (!Array.isArray(step)) return -1;
  return step.findIndex((p) => {
    const submission = p.action?.submission;
    return submission !== undefined && submission !== null && submission >= 0;
  });
}

export function cardText(card: HanabiCardFace): string {
  return `${card.color}${card.rank}`;
}
