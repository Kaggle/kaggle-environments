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

// Grid layout: rows = suits (♠♥♦♣), cols = ranks (A..K). Used by both the
// player hand grid and the discard mini-grid so cards line up by suit/rank.
const GRID_SUITS = ['s', 'h', 'd', 'c'] as const;
const GRID_RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'] as const;

const ALL_RANKS: readonly string[] = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'];
const ALL_SUITS: readonly string[] = ['s', 'c', 'd', 'h'];

function rankOf(card: string): string {
  const r = card[0];
  return RANK_DISPLAY[r] ?? r;
}

function suitOf(card: string): string {
  return card[1];
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
  // Each player's observation only reveals their own hand; merge so the
  // spectator view shows both hands at once.
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

function findInitialUpcard(steps: GinRummyStep[]): string | null {
  // In Oklahoma the initial upcard sets the deadwood limit for knocking.
  for (const s of steps) {
    const obs = mergedObservation(s.rawStep);
    if (!obs) continue;
    if (obs.phase === 'FirstUpcard') return obs.upcard;
    return obs.upcard ?? null;
  }
  return null;
}

function getPlayerName(replay: any, idx: number): string {
  return replay?.info?.TeamNames?.[idx] ?? replay?.agents?.[idx]?.name ?? (idx === 0 ? 'Player 1' : 'Player 2');
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
    classes.push(SUIT_COLOR[suitOf(card)]);
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
    // OpenSpiel canonical ordering: rank-major within suit (s, c, d, h).
    const suit = ALL_SUITS[Math.floor(submission / 13)];
    const rank = ALL_RANKS[submission % 13];
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
  return phase.replace(/([a-z])([A-Z])/g, '$1 $2');
}

function buildHandGrid(hand: string[], highlightCard: string | null): HTMLDivElement {
  // 4x13 grid (rows = suits, cols = ranks). Empty cells render a card-slot
  // placeholder so swapping a slot for a card produces zero layout shift.
  const handSet = new Set(hand);
  const grid = document.createElement('div');
  grid.className = 'hand-grid';
  for (const suit of GRID_SUITS) {
    const row = document.createElement('div');
    row.className = 'hand-grid__row';
    for (const rank of GRID_RANKS) {
      const card = `${rank}${suit}`;
      if (handSet.has(card)) {
        row.appendChild(buildCard(card, { highlight: !!highlightCard && card === highlightCard }));
      } else {
        const slot = document.createElement('div');
        slot.className = 'card-slot';
        row.appendChild(slot);
      }
    }
    grid.appendChild(row);
  }
  return grid;
}

function buildFaceDownHand(handLength: number): HTMLDivElement {
  // Opponent hand placeholder: a flat overlapping row of face-down cards.
  // We can't lay these out by suit/rank because the opponent's hand is hidden.
  const el = document.createElement('div');
  el.className = 'hand';
  const count = handLength || 10;
  for (let i = 0; i < count; i++) {
    el.appendChild(buildCard(null, { faceDown: true }));
  }
  return el;
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
  container.appendChild(showFaceDown ? buildFaceDownHand(hand.length) : buildHandGrid(hand, highlightCard));
}

function buildPile(visual: HTMLElement, label: string, modifier?: string): HTMLDivElement {
  const pile = document.createElement('div');
  pile.className = modifier ? `pile ${modifier}` : 'pile';
  pile.appendChild(visual);
  const lbl = document.createElement('div');
  lbl.className = 'pile-label';
  lbl.textContent = label;
  pile.appendChild(lbl);
  return pile;
}

function buildPileSlot(): HTMLDivElement {
  const slot = document.createElement('div');
  slot.className = 'card-slot--pile';
  return slot;
}

function buildStockPile(stockSize: number): HTMLDivElement {
  const stack = document.createElement('div');
  stack.className = 'pile-stack';
  if (stockSize === 0) {
    stack.appendChild(buildPileSlot());
  } else {
    // Decorative offset stack of up to 3 face-down cards.
    for (let i = 0; i < Math.min(stockSize, 3); i++) {
      stack.appendChild(buildCard(null, { faceDown: true }));
    }
  }
  return buildPile(stack, `Stock (${stockSize})`);
}

function buildUpcardPile(upcard: string | null): HTMLDivElement {
  // Wrap in a pile-stack so the empty and non-empty states have the same
  // 64x90 footprint as the Stock pile; when empty, render a dashed
  // card-shaped slot so the slot stays visible.
  const stack = document.createElement('div');
  stack.className = 'pile-stack';
  stack.appendChild(upcard ? buildCard(upcard) : buildPileSlot());
  return buildPile(stack, 'Upcard');
}

function buildKnockCardPile(initialUpcard: string | null, knockLimit: number | null): HTMLDivElement {
  let visual: HTMLElement;
  if (initialUpcard) {
    visual = buildCard(initialUpcard);
  } else {
    visual = document.createElement('div');
    visual.className = 'pile-empty';
    visual.textContent = '?';
  }
  const label = knockLimit !== null ? `Knock Card (≤${knockLimit})` : 'Knock Card';
  return buildPile(visual, label);
}

function buildDiscardPile(discardPile: string[]): HTMLDivElement {
  // Mini 4x13 grid showing every discarded card; top card outlined.
  const topCard = discardPile.length ? discardPile[discardPile.length - 1] : null;
  const discardSet = new Set(discardPile);
  const grid = document.createElement('div');
  grid.className = 'discard-grid';
  for (const suit of GRID_SUITS) {
    const row = document.createElement('div');
    row.className = 'discard-grid__row';
    for (const rank of GRID_RANKS) {
      const code = `${rank}${suit}`;
      const cell = document.createElement('div');
      cell.className = 'discard-cell';
      if (discardSet.has(code)) {
        cell.classList.add(SUIT_COLOR[suit]);
        if (code === topCard) cell.classList.add('top');
        cell.innerHTML =
          `<span class="dc-rank">${RANK_DISPLAY[rank] ?? rank}</span>` +
          `<span class="dc-suit">${SUIT_GLYPH[suit] ?? ''}</span>`;
      }
      row.appendChild(cell);
    }
    grid.appendChild(row);
  }
  return buildPile(grid, `Discards (${discardPile.length})`, 'pile--discard');
}

function buildStatus(
  observation: GinRummyObservation,
  lastAction: { actor: number; submission: number } | null,
  playerNames: string[],
  activeIdx: number,
  winnerIdx: number,
  isTerminal: boolean
): string {
  const parts: string[] = [];
  if (observation.phase) {
    parts.push(`<span class="phase-pill">${phaseLabel(observation.phase)}</span>`);
  }
  if (observation.knock_card !== null) {
    parts.push(`<span class="annotation">knock card: ${observation.knock_card}</span>`);
  }
  if (lastAction) {
    const { actor, submission } = lastAction;
    const color = actor === 0 ? PLAYER_0_COLOR : PLAYER_1_COLOR;
    const hand = observation.hands[String(actor) as '0' | '1'] ?? [];
    const lbl = actionLabel(submission, hand);
    parts.push(
      `<span class="annotation">last move:</span> ` +
        `<span style="color:${color};font-weight:700;">${playerNames[actor]} \u2192 ${lbl}</span>`
    );
  } else if (!isTerminal && activeIdx >= 0) {
    const color = activeIdx === 0 ? PLAYER_0_COLOR : PLAYER_1_COLOR;
    parts.push(`<span>Turn: <span style="color:${color};font-weight:700;">${playerNames[activeIdx]}</span></span>`);
  }
  if (isTerminal) {
    let html: string;
    if (winnerIdx === 0) {
      html = `<span style="color:${PLAYER_0_COLOR};font-weight:700;">${playerNames[0]} wins</span>`;
    } else if (winnerIdx === 1) {
      html = `<span style="color:${PLAYER_1_COLOR};font-weight:700;">${playerNames[1]} wins</span>`;
    } else {
      html = `<span>Game over: ${observation.winner ?? 'draw'}</span>`;
    }
    const ret = observation.returns;
    if (ret && ret.length === 2) {
      html += ` <span class="annotation">(score ${ret[0]} : ${ret[1]})</span>`;
    }
    parts.push(html);
  }
  return parts.join(' ');
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

  header.innerHTML = `
    <span class="player sketched-border ${activeIdx === 0 ? 'active' : ''}" style="color: ${PLAYER_0_COLOR};">
      ${playerNames[0]}
    </span>
    <span class="vs">vs</span>
    <span class="player sketched-border ${activeIdx === 1 ? 'active' : ''}" style="color: ${PLAYER_1_COLOR};">
      ${playerNames[1]}
    </span>
  `;

  // Highlight the just-discarded card inside the actor's hand. The status bar
  // already names the move, so we don't highlight stock/upcard/discard piles.
  const lastAction = findLastAction(currentStep);
  let highlightCardP0: string | null = null;
  let highlightCardP1: string | null = null;
  if (lastAction && lastAction.submission >= 0 && lastAction.submission < 52) {
    const dp = observation.discard_pile;
    const top = dp.length ? dp[dp.length - 1] : null;
    if (lastAction.actor === 0) highlightCardP0 = top;
    if (lastAction.actor === 1) highlightCardP1 = top;
  }

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

  centerRow.innerHTML = '';
  const oklahoma = !!replay?.configuration?.openSpielGameParameters?.oklahoma;
  if (oklahoma) {
    centerRow.appendChild(buildKnockCardPile(findInitialUpcard(steps), observation.knock_card));
  }
  centerRow.appendChild(buildStockPile(observation.stock_size));
  centerRow.appendChild(buildUpcardPile(observation.upcard));
  centerRow.appendChild(buildDiscardPile(observation.discard_pile));

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

  statusContainer.innerHTML = buildStatus(observation, lastAction, playerNames, activeIdx, winnerIdx, isTerminal);
}
