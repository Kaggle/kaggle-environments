import type { RendererOptions } from '@kaggle-environments/core';
import type { BargainingObs, BargainingStep, ItemBundle } from './transformers/bargainingReplayTypes';

const ITEM_KEYS = ['book', 'hat', 'basketball'] as const;
const ITEM_LABELS: Record<string, string> = {
  book: 'Book',
  hat: 'Hat',
  basketball: 'Basketball',
};
const ITEM_COLORS: Record<string, string> = {
  book: '#1f4f8b',
  hat: '#9a3324',
  basketball: '#7a5a1f',
};

function itemUtility(items: ItemBundle | undefined, values: ItemBundle | null): number | null {
  if (!items || !values) return null;
  let total = 0;
  for (const k of ITEM_KEYS) {
    total += (items[k] ?? 0) * (values[k] ?? 0);
  }
  return total;
}

function valuationChips(values: ItemBundle | null, hidden: boolean): string {
  return ITEM_KEYS.map((key) => {
    const swatch = `<span class="brg-val-swatch" style="background-color:${ITEM_COLORS[key]}"></span>`;
    if (hidden || !values) {
      return `<span class="brg-val-chip hidden">${swatch}${ITEM_LABELS[key]}: ?</span>`;
    }
    return `<span class="brg-val-chip">${swatch}${ITEM_LABELS[key]}: ${values[key] ?? 0}</span>`;
  }).join('');
}

function renderPool(pool: ItemBundle): string {
  return ITEM_KEYS.map((key) => {
    const qty = pool[key] ?? 0;
    const color = ITEM_COLORS[key];
    const items =
      Array(qty)
        .fill(0)
        .map(() => `<span class="brg-pool-item" style="background-color:${color}"></span>`)
        .join('') || '<span style="opacity:0.4">none</span>';
    return `
      <div class="brg-pool-row">
        <div class="brg-pool-label">
          <span class="brg-pool-item" style="background-color:${color};width:14px;height:14px"></span>
          ${ITEM_LABELS[key]}
        </div>
        <div class="brg-pool-items">${items}</div>
        <div class="brg-pool-count">×${qty}</div>
      </div>`;
  }).join('');
}

function itemBundleHtml(items: ItemBundle): string {
  return ITEM_KEYS.map((key) => {
    const qty = items[key] ?? 0;
    return `<span class="brg-mini-item">
      <span class="brg-mini-swatch" style="background-color:${ITEM_COLORS[key]}"></span>
      <span class="brg-mini-count">${qty}</span>
    </span>`;
  }).join(' ');
}

function complement(items: ItemBundle, pool: ItemBundle): ItemBundle {
  const out: ItemBundle = {};
  for (const k of ITEM_KEYS) {
    out[k] = Math.max(0, (pool[k] ?? 0) - (items[k] ?? 0));
  }
  return out;
}

function renderOfferSplit(proposerItems: ItemBundle, pool: ItemBundle): string {
  const other = complement(proposerItems, pool);
  return `
    <div class="brg-bubble-split">
      <span class="brg-split-side"><b>keeps:</b>${itemBundleHtml(proposerItems)}</span>
      <span class="brg-split-arrow">|</span>
      <span class="brg-split-side"><b>offers:</b>${itemBundleHtml(other)}</span>
    </div>`;
}

function renderConversation(obs: BargainingObs, pVals: [ItemBundle | null, ItemBundle | null]): string {
  const messages: string[] = [];
  const history = obs.offer_history;
  // Track each player's most recent offer (what they wanted to keep) so we can
  // resolve "agree" -- the agreeing player accepts the OTHER player's most
  // recent offer, and receives the complement.
  const lastOfferByPlayer: Record<number, ItemBundle | null> = { 0: null, 1: null };

  for (let i = 0; i < history.length; i++) {
    const event = history[i];
    const isLatest = i === history.length - 1;
    const sideClass = event.player === 0 ? 'from-p0' : 'from-p1';
    const colorBg = event.player === 0 ? '#1f4f8b' : '#9a3324';

    if (event.type === 'agree') {
      const otherPlayer = 1 - event.player;
      const acceptedOffer = lastOfferByPlayer[otherPlayer];
      // What the agreeing player receives is the complement of the offer.
      const youGet = acceptedOffer ? complement(acceptedOffer, obs.pool) : null;
      const yourUtil = itemUtility(youGet ?? undefined, pVals[event.player]);
      const otherUtil = acceptedOffer ? itemUtility(acceptedOffer, pVals[otherPlayer]) : null;
      messages.push(`
        <div class="brg-msg ${sideClass}">
          <div class="brg-bubble accept ${isLatest ? 'latest' : ''}">
            <div class="brg-bubble-who">
              <span class="brg-glyph" style="background-color:${colorBg}"></span>
              Player ${event.player + 1} ACCEPTS
            </div>
            ${acceptedOffer ? renderOfferSplit(acceptedOffer, obs.pool) : ''}
            ${
              yourUtil !== null && otherUtil !== null
                ? `<span class="brg-bubble-utility">utility — Player ${otherPlayer + 1}: ${otherUtil} · Player ${event.player + 1}: ${yourUtil}</span>`
                : ''
            }
          </div>
        </div>`);
    } else if (event.items) {
      lastOfferByPlayer[event.player] = event.items;
      const selfUtil = itemUtility(event.items, pVals[event.player]);
      const utilHtml =
        selfUtil !== null
          ? `<span class="brg-bubble-utility">${selfUtil} utility for Player ${event.player + 1} if accepted</span>`
          : '';
      messages.push(`
        <div class="brg-msg ${sideClass}">
          <div class="brg-bubble ${isLatest ? 'latest' : ''}">
            <div class="brg-bubble-who">
              <span class="brg-glyph" style="background-color:${colorBg}"></span>
              Player ${event.player + 1} offers
            </div>
            ${renderOfferSplit(event.items, obs.pool)}
            ${utilHtml}
          </div>
        </div>`);
    }
  }

  if (messages.length === 0) {
    return `<div class="brg-log-empty">No offers yet — Player 1 to open.</div>`;
  }
  return messages.join('');
}

function turnBadge(obs: BargainingObs, step: number, totalSteps: number): string {
  if (obs.is_terminal) {
    return `<div class="brg-turn-badge">step ${step} of ${totalSteps - 1}<br/>game over</div>`;
  }
  const who = obs.current_player >= 0 ? `Player ${obs.current_player + 1}` : '—';
  const turnsLeft = Math.max(0, obs.max_turns - obs.num_offers);
  return `<div class="brg-turn-badge">${who}'s turn<br/>step ${step} of ${totalSteps - 1} · ${turnsLeft} turn${turnsLeft === 1 ? '' : 's'} left</div>`;
}

function statusText(obs: BargainingObs): string {
  if (!obs.is_terminal) {
    return `<div>Offer ${obs.num_offers + 1} of up to ${obs.max_turns}</div>
      <div class="brg-status-sub">each offer proposes what the offering player keeps — opponent receives the rest</div>`;
  }
  const r = obs.returns ?? [0, 0];
  let header: string;
  if (!obs.agreement_reached) {
    header = '<b>No agreement</b> — both players score 0';
  } else if (r[0] > r[1]) {
    header = `<b>Deal accepted</b> — Player 1 wins on utility`;
  } else if (r[1] > r[0]) {
    header = `<b>Deal accepted</b> — Player 2 wins on utility`;
  } else {
    header = `<b>Deal accepted</b> — tied utility`;
  }
  return `<div>${header}</div>
    <div class="brg-status-sub">final utility: Player 1 = ${r[0]} · Player 2 = ${r[1]}</div>`;
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = (replay.steps as unknown as BargainingStep[]) ?? [];
  const safeStep = Math.max(0, Math.min(step, steps.length - 1));
  const current = steps[safeStep];

  parent.innerHTML = `
    <div class="renderer-container">
      <div class="brg-header"></div>
      <div class="brg-pool sketched-border"></div>
      <div class="brg-log sketched-border"></div>
      <div class="brg-status sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.brg-header') as HTMLDivElement;
  const pool = parent.querySelector('.brg-pool') as HTMLDivElement;
  const log = parent.querySelector('.brg-log') as HTMLDivElement;
  const status = parent.querySelector('.brg-status') as HTMLDivElement;
  if (!header || !pool || !log || !status) return;

  const obs0 = current?.observations?.[0] ?? null;
  const obs1 = current?.observations?.[1] ?? null;
  const obs = current?.obs ?? obs0 ?? obs1;
  if (!obs) {
    status.textContent = 'Waiting for replay…';
    return;
  }

  const isTerm = obs.is_terminal;
  const activeP0 = !isTerm && obs.current_player === 0;
  const activeP1 = !isTerm && obs.current_player === 1;

  const p0Vals = obs0?.my_values ?? null;
  const p1Vals = obs1?.my_values ?? null;
  const r = obs.returns ?? [null, null];

  const playerCard = (pid: 0 | 1, vals: ItemBundle | null, active: boolean, reward: number | null): string => {
    const accepted =
      isTerm &&
      obs.agreement_reached &&
      obs.offer_history.length > 0 &&
      obs.offer_history[obs.offer_history.length - 1].player === pid &&
      obs.offer_history[obs.offer_history.length - 1].type === 'agree';
    const cls = `brg-player-card sketched-border p${pid}${active ? ' active' : ''}${accepted ? ' accepted' : ''}`;
    const scoreLine =
      reward !== null ? `<span class="brg-score">utility ${reward}</span>` : `<span class="brg-score">utility —</span>`;
    return `
      <div class="${cls}">
        <div class="brg-player-name">
          <span class="brg-player-tag"><span class="brg-glyph"></span>Player ${pid + 1}</span>
          ${scoreLine}
        </div>
        <div class="brg-val-row">${valuationChips(vals, false)}</div>
      </div>`;
  };

  header.innerHTML = `
    ${playerCard(0, p0Vals, activeP0, r[0])}
    ${turnBadge(obs, safeStep, steps.length)}
    ${playerCard(1, p1Vals, activeP1, r[1])}
  `;

  pool.innerHTML = `
    <div class="brg-pool-title"><span>Item pool</span><span class="brg-pool-sub">split it between the two players</span></div>
    ${renderPool(obs.pool)}
  `;

  log.innerHTML = renderConversation(obs, [p0Vals, p1Vals]);
  log.scrollTop = log.scrollHeight;

  status.innerHTML = statusText(obs);
}
