import type { RendererOptions } from '@kaggle-environments/core';
import type { GinRummyStep } from './transformers/ginRummyTransformer';

interface GinRummyObservation {
  phase: string | null;
  current_player: number;
  is_terminal: boolean;
  winner: number | string | null;
  returns: number[];
  knock_card: number | null;
  prev_upcard: string | null;
  upcard: string | null;
  stock_size: number;
  discard_pile: string[];
  hands: { '0': string[]; '1': string[] };
  deadwood: { '0': number | null; '1': number | null };
  repeated_move: number;
}

const PLAYER_0_COLOR = '#1f77b4';
const PLAYER_1_COLOR = '#d62728';
const SUIT_GLYPH: Record<string, string> = { s: '\u2660', c: '\u2663', d: '\u2666', h: '\u2665' };
const SUIT_COLOR: Record<string, 'red' | 'black'> = { s: 'black', c: 'black', d: 'red', h: 'red' };
const RANK_DISPLAY: Record<string, string> = {
  A: 'A',
  T: '10',
  J: 'J',
  Q: 'Q',
  K: 'K',
};

function rankOf(card: string): string {
  const r = card[0];
  return RANK_DISPLAY[r] ?? r;
}

function suitOf(card: string): string {
  return card[1];
}

const RANK_ORDER: Record<string, number> = {
  A: 1,
  '2': 2,
  '3': 3,
  '4': 4,
  '5': 5,
  '6': 6,
  '7': 7,
  '8': 8,
  '9': 9,
  T: 10,
  J: 11,
  Q: 12,
  K: 13,
};
const SUIT_ORDER: Record<string, number> = { s: 0, h: 1, d: 2, c: 3 };

function sortHand(hand: string[]): string[] {
  return [...hand].sort((a, b) => {
    const sa = SUIT_ORDER[suitOf(a)] ?? 9;
    const sb = SUIT_ORDER[suitOf(b)] ?? 9;
    if (sa !== sb) return sa - sb;
    return (RANK_ORDER[a[0]] ?? 99) - (RANK_ORDER[b[0]] ?? 99);
  });
}

function parseObservation(step: any, playerIdx: number): GinRummyObservation | null {
  const raw = step?.[playerIdx]?.observation?.observationString;
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function mergedObservation(step: any): GinRummyObservation | null {
  // Player 0's observation has player 0's hand visible; player 1's observation
  // has player 1's hand visible. Merge so the spectator view shows both.
  const o0 = parseObservation(step, 0);
  const o1 = parseObservation(step, 1);
  const base = o0 ?? o1;
  if (!base) return null;
  const merged: GinRummyObservation = JSON.parse(JSON.stringify(base));
  if (o0 && o0.hands?.['0']?.length) {
    merged.hands['0'] = o0.hands['0'];
    merged.deadwood['0'] = o0.deadwood['0'];
  }
  if (o1 && o1.hands?.['1']?.length) {
    merged.hands['1'] = o1.hands['1'];
    merged.deadwood['1'] = o1.deadwood['1'];
  }
  return merged;
}

function getPlayerName(replay: any, idx: number): string {
  const info = replay?.info?.TeamNames?.[idx];
  if (info) return info;
  const fromAgent = replay?.agents?.[idx]?.name;
  if (fromAgent) return fromAgent;
  return idx === 0 ? 'Player 1' : 'Player 2';
}

function buildCard(
  card: string | null,
  options: { compact?: boolean; faceDown?: boolean; highlight?: boolean; dim?: boolean } = {}
): HTMLDivElement {
  const el = document.createElement('div');
  const classes = ['card'];
  if (options.compact) classes.push('compact');
  if (options.faceDown || !card) {
    classes.push('face-down');
  } else {
    const suit = suitOf(card);
    classes.push(SUIT_COLOR[suit]);
  }
  if (options.highlight) classes.push('highlight');
  if (options.dim) classes.push('dim');
  el.className = classes.join(' ');
  if (!options.faceDown && card) {
    const rank = rankOf(card);
    const glyph = SUIT_GLYPH[suitOf(card)] ?? '?';
    el.innerHTML = `
      <div class="corner top">
        <span>${rank}</span>
        <span class="suit">${glyph}</span>
      </div>
      <div class="corner bottom">
        <span>${rank}</span>
        <span class="suit">${glyph}</span>
      </div>
    `;
  }
  return el;
}

function actionLabel(submission: number, hand: string[] | null): string {
  if (submission === 52) return 'Draw upcard';
  if (submission === 53) return 'Draw stock';
  if (submission === 54) return 'Pass';
  if (submission === 55) return 'Knock';
  if (submission >= 0 && submission < 52) {
    // Single card action: drawn-card index 0..51 maps to a specific card.
    // We don't know the exact card from the action alone in all phases;
    // best effort: derive card string from action index using OpenSpiel's
    // canonical ordering: rank-major within suit (s, c, d, h).
    const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'];
    const suits = ['s', 'c', 'd', 'h'];
    const suit = suits[Math.floor(submission / 13)];
    const rank = ranks[submission % 13];
    const card = `${rank}${suit}`;
    if (hand && hand.includes(card)) return `Discard ${rank}${SUIT_GLYPH[suit]}`;
    return `${rank}${SUIT_GLYPH[suit]}`;
  }
  if (submission >= 56) return `Meld (action ${submission})`;
  return `Action ${submission}`;
}

function findLastAction(step: any): { actor: number; submission: number } | null {
  if (!Array.isArray(step)) return null;
  for (let i = 0; i < step.length; i++) {
    const sub = step[i]?.action?.submission;
    if (typeof sub === 'number' && sub >= 0) return { actor: i, submission: sub };
  }
  return null;
}

function phaseLabel(phase: string | null): string {
  if (!phase) return '';
  // Make camel-case phases more readable.
  return phase.replace(/([a-z])([A-Z])/g, '$1 $2');
}

function renderPlayerRow(
  container: HTMLDivElement,
  name: string,
  hand: string[],
  deadwood: number | null,
  isActive: boolean,
  isWinner: boolean,
  color: string,
  showFaceDown: boolean,
  highlightCard: string | null
) {
  container.innerHTML = '';

  const meta = document.createElement('div');
  meta.className = 'player-meta';
  meta.innerHTML = `
    <span class="sketched-border" style="padding:2px 10px;background:white;color:${color};font-weight:700;">
      ${name}${isActive ? ' \u25b6' : ''}${isWinner ? ' \u2605' : ''}
    </span>
    <span class="deadwood sketched-border">Deadwood: ${deadwood ?? '?'}</span>
    <span class="deadwood sketched-border">Cards: ${hand.length}</span>
  `;
  container.appendChild(meta);

  const handEl = document.createElement('div');
  handEl.className = 'hand';
  const sorted = showFaceDown ? hand : sortHand(hand);
  // If hand empty and showFaceDown, render placeholder face-down cards.
  const cardsToRender = sorted.length ? sorted : showFaceDown ? new Array(10).fill('XX') : [];
  for (const card of cardsToRender) {
    const isHighlight = !!highlightCard && card === highlightCard && !showFaceDown;
    handEl.appendChild(
      buildCard(card === 'XX' ? null : card, {
        faceDown: showFaceDown,
        highlight: isHighlight,
      })
    );
  }
  container.appendChild(handEl);
}

export function renderer(options: RendererOptions<GinRummyStep[]>) {
  const { parent, replay, step } = options;
  const steps = (replay?.steps ?? []) as GinRummyStep[];
  if (!steps.length) return;

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <div class="table">
        <div class="player-row top-row"></div>
        <div class="center-row"></div>
        <div class="player-row bottom-row"></div>
      </div>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const topRow = parent.querySelector('.top-row') as HTMLDivElement;
  const centerRow = parent.querySelector('.center-row') as HTMLDivElement;
  const bottomRow = parent.querySelector('.bottom-row') as HTMLDivElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;

  const currentStep = steps[step]?.rawStep;
  const observation = mergedObservation(currentStep);
  if (!observation) {
    statusContainer.textContent = 'Waiting for first observation...';
    return;
  }

  const playerNames = [getPlayerName(replay, 0), getPlayerName(replay, 1)];
  const isTerminal = observation.is_terminal;
  const activeIdx = isTerminal ? -1 : observation.current_player;
  const winnerIdx = typeof observation.winner === 'number' ? observation.winner : -1;

  // Header
  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${PLAYER_0_COLOR};">
      ${playerNames[0]}
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${PLAYER_1_COLOR};">
      ${playerNames[1]}
    </span>
  `;

  // Detect the action that produced this state (if any).
  const lastAction = findLastAction(currentStep);
  let highlightDiscard: string | null = null;
  let highlightTakenUpcard = false;
  let highlightDrawStock = false;
  let highlightCardP0: string | null = null;
  let highlightCardP1: string | null = null;
  if (lastAction) {
    const { actor, submission } = lastAction;
    if (submission === 52) highlightTakenUpcard = true;
    else if (submission === 53) highlightDrawStock = true;
    else if (submission >= 0 && submission < 52) {
      // Single-card action: top card of discard pile is the just-discarded card
      // (in Discard / Knock phases). Highlight it as the move.
      const discard = observation.discard_pile;
      if (discard.length) highlightDiscard = discard[discard.length - 1];
      // If knock, highlight that card in the actor's hand as well.
      if (actor === 0) highlightCardP0 = discard.length ? discard[discard.length - 1] : null;
      if (actor === 1) highlightCardP1 = discard.length ? discard[discard.length - 1] : null;
    }
  }

  // Top: player 1
  renderPlayerRow(
    topRow,
    playerNames[1],
    observation.hands['1'] ?? [],
    observation.deadwood['1'],
    activeIdx === 1,
    winnerIdx === 1,
    PLAYER_1_COLOR,
    false,
    highlightCardP1
  );

  // Center: stock | discard | (recent action info)
  centerRow.innerHTML = '';

  // Stock pile (face-down stack)
  const stockPile = document.createElement('div');
  stockPile.className = 'pile';
  const stockStack = document.createElement('div');
  stockStack.className = 'pile-stack';
  const stockCardCount = Math.min(observation.stock_size, 3);
  if (stockCardCount === 0) {
    const empty = document.createElement('div');
    empty.className = 'pile-empty';
    empty.textContent = 'empty';
    stockPile.appendChild(empty);
  } else {
    for (let i = 0; i < stockCardCount; i++) {
      stockStack.appendChild(
        buildCard(null, { faceDown: true, highlight: i === stockCardCount - 1 && highlightDrawStock })
      );
    }
    stockPile.appendChild(stockStack);
  }
  const stockLabel = document.createElement('div');
  stockLabel.textContent = `Stock (${observation.stock_size})`;
  stockPile.appendChild(stockLabel);
  centerRow.appendChild(stockPile);

  // Upcard / discard top -- in OpenSpiel gin_rummy, the upcard is the visible
  // top of the discard pile during draw phases; once drawn or after a discard,
  // the discard pile's last card is the visible top. We render the upcard slot
  // separately when present (Draw phase) and the rest of the pile beneath.
  const upcardPile = document.createElement('div');
  upcardPile.className = 'pile';
  if (observation.upcard) {
    upcardPile.appendChild(
      buildCard(observation.upcard, {
        highlight: highlightTakenUpcard || (highlightDiscard !== null && highlightDiscard === observation.upcard),
      })
    );
    const lbl = document.createElement('div');
    lbl.textContent = 'Upcard';
    upcardPile.appendChild(lbl);
  } else {
    const empty = document.createElement('div');
    empty.className = 'pile-empty';
    empty.textContent = 'no upcard';
    upcardPile.appendChild(empty);
    const lbl = document.createElement('div');
    lbl.textContent = 'Upcard';
    upcardPile.appendChild(lbl);
  }
  centerRow.appendChild(upcardPile);

  // Discard pile beneath the upcard slot. The OpenSpiel discard_pile and
  // upcard are disjoint (the upcard is reported on its own line and is not
  // included in the discard pile).
  const discardPile = document.createElement('div');
  discardPile.className = 'pile';
  const discardStack = document.createElement('div');
  discardStack.className = 'pile-stack';
  const dp = observation.discard_pile;
  const visible = dp.slice(-3);
  if (visible.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'pile-empty';
    empty.textContent = 'empty';
    discardPile.appendChild(empty);
  } else {
    visible.forEach((card, i) => {
      const isTop = i === visible.length - 1;
      discardStack.appendChild(
        buildCard(card, {
          highlight: isTop && highlightDiscard !== null && card === highlightDiscard,
        })
      );
    });
    discardPile.appendChild(discardStack);
  }
  const dpLabel = document.createElement('div');
  dpLabel.textContent = `Discard (${dp.length})`;
  discardPile.appendChild(dpLabel);
  centerRow.appendChild(discardPile);

  // Bottom: player 0
  renderPlayerRow(
    bottomRow,
    playerNames[0],
    observation.hands['0'] ?? [],
    observation.deadwood['0'],
    activeIdx === 0,
    winnerIdx === 0,
    PLAYER_0_COLOR,
    false,
    highlightCardP0
  );

  // Status: phase, last action, terminal result
  const parts: string[] = [];
  if (observation.phase) {
    parts.push(`<span class="phase-pill">${phaseLabel(observation.phase)}</span>`);
  }
  if (observation.knock_card !== null) {
    parts.push(`<span class="annotation">knock card: ${observation.knock_card}</span>`);
  }
  if (lastAction) {
    const { actor, submission } = lastAction;
    const moverColor = actor === 0 ? PLAYER_0_COLOR : PLAYER_1_COLOR;
    const moverHand = observation.hands[String(actor) as '0' | '1'] ?? [];
    const lbl = actionLabel(submission, moverHand);
    parts.push(
      `<span class="annotation">last move:</span> <span style="color:${moverColor};font-weight:700;">${playerNames[actor]} \u2192 ${lbl}</span>`
    );
  } else if (!isTerminal) {
    const turnColor = activeIdx === 0 ? PLAYER_0_COLOR : PLAYER_1_COLOR;
    const turnName = activeIdx >= 0 ? playerNames[activeIdx] : '';
    if (turnName) {
      parts.push(`<span>Turn: <span style="color:${turnColor};font-weight:700;">${turnName}</span></span>`);
    }
  }

  if (isTerminal) {
    let resultHTML = '';
    if (winnerIdx === 0) {
      resultHTML = `<span style="color:${PLAYER_0_COLOR};font-weight:700;">${playerNames[0]} wins</span>`;
    } else if (winnerIdx === 1) {
      resultHTML = `<span style="color:${PLAYER_1_COLOR};font-weight:700;">${playerNames[1]} wins</span>`;
    } else {
      resultHTML = `<span>Game over: ${observation.winner ?? 'draw'}</span>`;
    }
    const ret = observation.returns;
    if (ret && ret.length === 2) {
      resultHTML += ` <span class="annotation">(score ${ret[0]} : ${ret[1]})</span>`;
    }
    parts.push(resultHTML);
  }

  statusContainer.innerHTML = parts.join(' ');
}
