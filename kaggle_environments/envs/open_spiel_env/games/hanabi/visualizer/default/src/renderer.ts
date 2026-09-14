// Hanabi renderer.
//
// The whole game is the gap between what a card *is* and what its holder
// *knows*, so every card is drawn twice over: the face (visible to the
// spectator, hidden from its owner) and, beneath it, the knowledge strip --
// what the holder has been told and what they can still deduce. A card whose
// owner knows nothing shows a full plausible set; a fully-hinted card shows a
// single colored pip.
//
// Everything else on screen is shared team state: the five fireworks, the life
// and info tokens, the deck, and the discard pile.

import { escapeHtml, type RendererOptions } from '@kaggle-environments/core';

import {
  COLOR_INK,
  COLOR_ORDER,
  COLOR_TINT,
  HanabiCard,
  HanabiObservation,
  actingPlayer,
  cardText,
  mergedObservation,
  parseMove,
} from './observation';
import type { HanabiStep } from './transformers/hanabiTransformer';

// Seat colors. Hanabi supports 2-5 players, so this has an entry per seat
// rather than the usual two; it wraps defensively if a variant ever exceeds it.
const SEAT_COLORS = ['#1f77b4', '#d62728', '#2f8f4f', '#8a5cc4', '#b8860b'];

function playerColor(idx: number): string {
  return SEAT_COLORS[idx % SEAT_COLORS.length];
}

function getPlayerName(replay: any, idx: number): string {
  return replay?.info?.TeamNames?.[idx] ?? replay?.agents?.[idx]?.name ?? `Player ${idx + 1}`;
}

/**
 * One name per seat. The seat count comes from the observation rather than
 * from `TeamNames`, which can be missing or short in a hand-built replay.
 */
function getPlayerNames(replay: any, numPlayers: number): string[] {
  return Array.from({ length: numPlayers }, (_, i) => getPlayerName(replay, i));
}

/** Colors present in this game, in display order, plus any unexpected extras. */
function colorsOf(observation: HanabiObservation): string[] {
  const present = Object.keys(observation.fireworks);
  const ordered = COLOR_ORDER.filter((c) => present.includes(c));
  return [...ordered, ...present.filter((c) => !ordered.includes(c))];
}

/** What changed on this step, so the renderer can point at it. */
interface MoveHighlight {
  /** Slots in the target's hand touched by a hint. */
  hintTarget: number | null;
  hintSlots: Set<number>;
  /** Firework that just advanced. */
  playedColor: string | null;
  /** The discard pile grew (by a discard or a misplay). */
  discardAdded: boolean;
  /** Player who drew a replacement, and the slot it landed in. */
  drew: { player: number; slot: number } | null;
}

const NO_HIGHLIGHT: MoveHighlight = {
  hintTarget: null,
  hintSlots: new Set(),
  playedColor: null,
  discardAdded: false,
  drew: null,
};

function computeHighlight(
  actor: number,
  actionString: string | null | undefined,
  before: HanabiObservation | null,
  after: HanabiObservation | null
): MoveHighlight {
  if (actor < 0 || !before || !after) return NO_HIGHLIGHT;
  const move = parseMove(actionString, actor, before.num_players);
  if (!move) return NO_HIGHLIGHT;

  if (move.type === 'hint') {
    // A hint shifts nothing, so the touched slots are the same before and after.
    const hand = after.hands.find((h) => h.player === move.target);
    const slots = new Set<number>();
    (hand?.cards ?? []).forEach((card, i) => {
      const hit = 'color' in move ? card.card?.color === move.color : card.card?.rank === move.rank;
      if (hit) slots.add(i);
    });
    return { ...NO_HIGHLIGHT, hintSlots: slots, hintTarget: move.target };
  }

  // Playing or discarding removes a card, the rest shift left, and the
  // replacement (if the deck still has one) lands in the last slot.
  const handAfter = after.hands.find((h) => h.player === actor)?.cards.length ?? 0;
  const handBefore = before.hands.find((h) => h.player === actor)?.cards.length ?? 0;
  const played = before.hands.find((h) => h.player === actor)?.cards[move.slot]?.card ?? null;
  return {
    hintTarget: null,
    hintSlots: new Set(),
    // Stack heights, not banked score: the score drops to 0 on a bomb-out and
    // would read as "no firework advanced" on an unrelated later comparison.
    playedColor: after.fireworks_total > before.fireworks_total ? (played?.color ?? null) : null,
    discardAdded: after.discards.length > before.discards.length,
    // The hand only stays full if the deck had a card left to replace with.
    drew: handAfter === handBefore && handAfter > 0 ? { player: actor, slot: handAfter - 1 } : null,
  };
}

function delta(before: number | undefined, after: number): string {
  if (before === undefined || before === after) return '';
  const diff = after - before;
  const sign = diff > 0 ? '+' : '';
  return `<span class="delta ${diff > 0 ? 'up' : 'down'}">${sign}${diff}</span>`;
}

function buildTokens(observation: HanabiObservation, previous: HanabiObservation | null): string {
  const lives = Array.from(
    { length: observation.max_life_tokens },
    (_, i) => `<span class="token life ${i < observation.life_tokens ? '' : 'spent'}">&#9829;</span>`
  ).join('');
  const info = Array.from(
    { length: observation.max_info_tokens },
    (_, i) => `<span class="token info ${i < observation.info_tokens ? '' : 'spent'}">&#9679;</span>`
  ).join('');
  return `
    <span class="gauge sketched-border">
      <span class="gauge-label">Lives</span>${lives}${delta(previous?.life_tokens, observation.life_tokens)}
    </span>
    <span class="gauge sketched-border">
      <span class="gauge-label">Info</span>${info}${delta(previous?.info_tokens, observation.info_tokens)}
    </span>
    <span class="gauge sketched-border">
      <span class="gauge-label">Deck</span><span class="gauge-value">${observation.deck_size}</span>
    </span>
    <span class="gauge sketched-border score">
      <span class="gauge-label">Score</span>
      <span class="gauge-value">${observation.score} / ${observation.max_score}</span>
      ${delta(previous?.score, observation.score)}
    </span>
  `;
}

function buildFireworks(observation: HanabiObservation, highlight: MoveHighlight): string {
  const piles = colorsOf(observation)
    .map((color) => {
      const height = observation.fireworks[color] ?? 0;
      const ink = COLOR_INK[color] ?? '#444343';
      const rungs = Array.from({ length: observation.ranks }, (_, i) => {
        const rank = i + 1;
        const lit = rank <= height;
        return `<span class="rung ${lit ? 'lit' : ''}" style="${lit ? `background:${ink};` : ''}">${rank}</span>`;
      })
        .reverse()
        .join('');
      const advanced = highlight.playedColor === color ? ' advanced' : '';
      return `
        <div class="firework${advanced}" style="--ink:${ink};">
          <div class="rungs">${rungs}</div>
          <div class="firework-label" style="color:${ink};">${escapeHtml(color)}</div>
        </div>`;
    })
    .join('');
  return `<div class="fireworks">${piles}</div>`;
}

/** The knowledge strip: what the holder has been told, and what is left open. */
function buildKnowledge(card: HanabiCard, observation: HanabiObservation): string {
  const colors = colorsOf(observation);
  const pips = colors
    .map((color) => {
      const open = card.plausible_colors.includes(color);
      const told = card.hinted_color === color;
      const ink = COLOR_INK[color] ?? '#444343';
      return `<span class="pip ${open ? 'open' : ''} ${told ? 'told' : ''}" style="--ink:${ink};"></span>`;
    })
    .join('');
  const ranks = Array.from({ length: observation.ranks }, (_, i) => {
    const rank = i + 1;
    const open = card.plausible_ranks.includes(rank);
    const told = card.hinted_rank === rank;
    return `<span class="rank-mark ${open ? 'open' : ''} ${told ? 'told' : ''}">${rank}</span>`;
  }).join('');
  return `<div class="knowledge"><div class="pips">${pips}</div><div class="rank-marks">${ranks}</div></div>`;
}

function buildCard(card: HanabiCard, observation: HanabiObservation, classes: string[]): string {
  const face = card.card;
  const ink = face ? (COLOR_INK[face.color] ?? '#444343') : '#8b8b8b';
  const tint = face ? (COLOR_TINT[face.color] ?? '#ffffff') : '#ffffff';
  const faceHtml = face
    ? `<span class="card-rank" style="color:${ink};">${face.rank}</span>
       <span class="card-color" style="color:${ink};">${escapeHtml(face.color)}</span>`
    : `<span class="card-rank unknown">?</span>`;
  return `
    <div class="hb-card ${classes.join(' ')}" style="--ink:${ink};background-color:${tint};">
      <div class="card-face">${faceHtml}</div>
      ${buildKnowledge(card, observation)}
    </div>`;
}

function buildHand(
  observation: HanabiObservation,
  player: number,
  name: string,
  isActive: boolean,
  highlight: MoveHighlight
): string {
  const hand = observation.hands.find((h) => h.player === player);
  const cards = (hand?.cards ?? [])
    .map((card, slot) => {
      const classes: string[] = [];
      if (highlight.hintTarget === player && highlight.hintSlots.has(slot)) classes.push('hinted');
      if (highlight.drew?.player === player && highlight.drew.slot === slot) classes.push('drawn');
      return buildCard(card, observation, classes);
    })
    .join('');
  const body = cards || '<div class="hb-empty">no cards</div>';
  return `
    <div class="hand-panel sketched-border ${isActive ? 'active' : ''}">
      <div class="hand-header" style="color:${playerColor(player)};">
        ${escapeHtml(name)}${isActive ? ' <span class="turn-arrow">&#9654;</span>' : ''}
      </div>
      <div class="hb-hand">${body}</div>
    </div>`;
}

function buildDiscards(observation: HanabiObservation, highlight: MoveHighlight): string {
  if (!observation.discards.length) {
    return `<div class="discards"><span class="discards-label">Discards</span><span class="hb-empty">none</span></div>`;
  }
  const chips = observation.discards
    .map((card, i) => {
      const ink = COLOR_INK[card.color] ?? '#444343';
      const fresh = highlight.discardAdded && i === observation.discards.length - 1 ? ' fresh' : '';
      return `<span class="discard-chip${fresh}" style="color:${ink};border-color:${ink};">${escapeHtml(cardText(card))}</span>`;
    })
    .join('');
  return `<div class="discards"><span class="discards-label">Discards</span>${chips}</div>`;
}

function buildStatus(stepData: HanabiStep | undefined, playerNames: string[], activeIdx: number): string {
  const parts: string[] = [];
  if (stepData?.moveSummary) {
    parts.push(`<span class="annotation">${escapeHtml(stepData.moveSummary)}</span>`);
  }
  if (stepData?.isTerminal) {
    const result = stepData.forfeitReason ? 'Game over' : (stepData.winner ?? 'Game over');
    parts.push(`<p class="result">${escapeHtml(result)}</p>`);
    if (stepData.forfeitReason) {
      parts.push(`<span class="annotation forfeit-reason">${escapeHtml(stepData.forfeitReason)}</span>`);
    }
  } else if (activeIdx >= 0) {
    const name = playerNames[activeIdx] ?? `Player ${activeIdx + 1}`;
    parts.push(
      `<span>Turn: <span style="color:${playerColor(activeIdx)};font-weight:700;">${escapeHtml(name)}</span></span>`
    );
  }
  return parts.join(' ');
}

export function renderer(options: RendererOptions<HanabiStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as HanabiStep[];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="board"></div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const board = parent.querySelector('.board') as HTMLDivElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;
  if (!steps.length) {
    statusContainer.textContent = 'No replay data.';
    return;
  }

  const stepData = steps[step];
  const currentStep = stepData?.rawStep;
  const observation = mergedObservation(currentStep);
  const previous = mergedObservation(steps[step - 1]?.rawStep);

  if (!observation) {
    // The first step precedes the deal, so there is no state to draw yet. The
    // seat count is not known from an observation, so fall back to the step.
    const preDealNames = getPlayerNames(replay, currentStep?.length || 2);
    header.innerHTML = preDealNames
      .map(
        (name, i) => `<span class="player sketched-border" style="color:${playerColor(i)};">${escapeHtml(name)}</span>`
      )
      .join('<span class="vs">&amp;</span>');
    statusContainer.textContent = 'Dealing...';
    return;
  }

  const playerNames = getPlayerNames(replay, observation.num_players);

  // Prefer the transformer's isTerminal: it also fires on forfeits, which the
  // raw OpenSpiel observation does not mark terminal.
  const isTerminal = !!stepData?.isTerminal || observation.is_terminal;
  const activeIdx = isTerminal ? -1 : observation.current_player;

  const actor = actingPlayer(currentStep);
  const highlight = computeHighlight(actor, currentStep?.[actor]?.action?.actionString, previous, observation);

  // Hanabi is cooperative: the seats are teammates, not opponents.
  header.innerHTML = playerNames
    .map(
      (name, i) =>
        `<span class="player sketched-border ${activeIdx === i ? 'active' : ''}" style="color:${playerColor(i)};">` +
        `${escapeHtml(name)}</span>`
    )
    .join('<span class="vs">&amp;</span>');

  board.innerHTML = `
    <div class="gauges">${buildTokens(observation, previous)}</div>
    ${buildFireworks(observation, highlight)}
    <div class="hands">
      ${playerNames.map((name, i) => buildHand(observation, i, name, activeIdx === i, highlight)).join('')}
    </div>
    ${buildDiscards(observation, highlight)}
  `;

  statusContainer.innerHTML = buildStatus(stepData, playerNames, activeIdx);
}
