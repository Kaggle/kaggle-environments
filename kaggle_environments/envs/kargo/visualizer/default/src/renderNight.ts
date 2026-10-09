/** The overnight screen: what was on offer, what each agent did about it.
 *
 * Three of every eight steps are markets, not driving, and on the map they look
 * like a frozen frame -- the trucks are parked and nothing moves. The whole
 * game happens here, so it gets its own picture rather than a paused one.
 *
 * Every phase is the same three-part story and is laid out the same way: the
 * OFFER on the left (the board the engine posted), the REPLY in the middle
 * (what each seat submitted), the OUTCOME on the right (how the engine settled
 * it). Keeping the columns fixed across phases means a reader learns the
 * grammar once. Where a phase's offer is a list both seats bid on -- the
 * freight board -- the three collapse into one table instead, because splitting
 * a bid away from the lot it names would make the reader join them by eye.
 */

import { esc, flankMarkup, topbarMarkup } from './chrome';
import { INK_MUTED, PLAYER_INK } from './theme';
import type { Listing, View } from './types';
import { money, truckSprite } from './utils';

export interface NightRefs {
  board: HTMLElement;
}

export function buildNightShell(parent: HTMLElement, names: string[]): void {
  parent.innerHTML = `
    <div class="kargo">
      ${topbarMarkup()}
      <div class="main">
        ${flankMarkup(names, 'left')}
        <div class="night" data-night></div>
        ${flankMarkup(names, 'right')}
      </div>
    </div>`;
}

export function nightRefs(parent: HTMLElement): NightRefs {
  return { board: parent.querySelector('[data-night]') as HTMLElement };
}

/** A seat's name in its own ink. Used wherever a row belongs to one player. */
function seatChip(view: View, pid: number): string {
  const name = view.names[pid] ?? `P${pid + 1}`;
  return `<span class="seat-chip"><i style="background:${PLAYER_INK[pid] ?? INK_MUTED}"></i>${esc(name)}</span>`;
}

function section(title: string, note: string, body: string): string {
  return `
    <section class="nsec">
      <header class="nsec-head"><h3>${esc(title)}</h3><span>${esc(note)}</span></header>
      ${body}
    </section>`;
}

function empty(text: string): string {
  return `<p class="nempty">${esc(text)}</p>`;
}

/** Two seat columns side by side, each headed by its own name. */
function seatColumns(view: View, body: (pid: number) => string): string {
  return `<div class="seat-cols">${view.names
    .map(
      (_n, pid) => `<div class="seat-col" style="--seat-ink:${PLAYER_INK[pid] ?? INK_MUTED}">
        <header>${seatChip(view, pid)}</header>
        ${body(pid)}
      </div>`
    )
    .join('')}</div>`;
}

const asList = (v: unknown): any[][] => (Array.isArray(v) ? v.filter((e) => Array.isArray(e) && e.length) : []);

// --- CAPEX -------------------------------------------------------------------

function capexOffer(view: View): string {
  const used = view.market.used ?? [];
  const rentals = Object.entries(view.market.rentals ?? {});
  const usedRows = used.length
    ? `<table class="ntable">
        <thead><tr><th>Used</th><th>Type</th><th class="num">Odometer</th><th class="num">Age</th><th class="num">Price</th></tr></thead>
        <tbody>${used
          .map(
            (u) => `<tr>
              <td>${esc(u.id)}</td><td>${esc(u.type)}</td>
              <td class="num">${u.odometer.toLocaleString('en-US')} km</td>
              <td class="num">${Math.round(u.age_days / 30)} mo</td>
              <td class="num">${money(u.price)}</td>
            </tr>`
          )
          .join('')}</tbody>
      </table>`
    : empty('No used trucks tonight.');
  // The pool is shared: a rental one seat takes is one the other cannot.
  const pool = rentals.length
    ? `<div class="chips">${rentals
        .map(([type, n]) => `<span class="chip">${esc(type)} <b>${n}</b> left</span>`)
        .join('')}</div>`
    : '';
  return usedRows + pool;
}

function capexReply(view: View, pid: number): string {
  const act = view.night.actions[pid] ?? {};
  const stage = (act.stage ?? {}) as Record<string, string>;
  const staged = Object.entries(stage);
  const fuel = asList(act.fuel);
  const fleet = asList(act.fleet);
  const rows: string[] = [];
  if (fleet.length) rows.push(...fleet.map((e) => `<li><b>${esc(e[0])}</b> ${esc(e[1] ?? '')}</li>`));
  if (fuel.length) rows.push(`<li><b>REFUEL</b> ${esc(fuel.map((e) => e[1]).join(', '))}</li>`);
  if (staged.length)
    rows.push(`<li><b>STAGE</b> ${staged.map(([tid, wid]) => `${esc(tid)}&rarr;${esc(wid)}`).join(', ')}</li>`);
  // On the replay's last step the reply does not exist yet, which is not the
  // same as an empty one. "Stood pat" there would put words in an agent's
  // mouth it never got to open.
  if (rows.length) return `<ul class="nlist">${rows.join('')}</ul>`;
  return empty(view.night.pending ? 'Not submitted yet.' : 'Stood pat.');
}

function capexOutcome(view: View): string {
  const log = view.night.outcome?.capex ?? [];
  if (view.night.pending) return empty('Awaiting the reply.');
  if (!log.length) return empty('Nothing settled: staging and refuelling always succeed and are not logged.');
  return `<ul class="nlist">${log
    .map((r) => {
      const detail = [
        r.truck,
        r.type,
        r.account,
        r.proceeds != null ? money(r.proceeds) : '',
        r.fee != null ? money(r.fee) : '',
      ]
        .filter(Boolean)
        .join(' ');
      return `<li>${seatChip(view, r.player)} <b>${esc(r.op)}</b> ${esc(detail)}${
        r.arrives != null ? ` <em>arrives day ${r.arrives + 1}</em>` : ''
      }</li>`;
    })
    .join('')}</ul>`;
}

// --- LABOR -------------------------------------------------------------------

function laborOffer(view: View): string {
  const cands = view.market.candidates ?? [];
  if (!cands.length) return empty('Nobody on the market tonight.');
  return `<table class="ntable">
    <thead><tr><th>Candidate</th><th class="num">Résumé</th><th class="num">Asking</th></tr></thead>
    <tbody>${cands
      .map(
        (c) => `<tr>
          <td>${esc(c.name)} <em>${esc(c.id)}</em></td>
          <td class="num">${c.resume}</td>
          <td class="num">${money(c.asking)}/day</td>
        </tr>`
      )
      .join('')}</tbody>
  </table>`;
}

function laborReply(view: View, pid: number): string {
  const moves = asList((view.night.actions[pid] ?? {}).labor);
  if (!moves.length) return empty(view.night.pending ? 'Not submitted yet.' : 'No moves.');
  return `<ul class="nlist">${moves
    .map((e) => `<li><b>${esc(e[0])}</b> ${esc(e.slice(1).join(' '))}</li>`)
    .join('')}</ul>`;
}

/** The roster as it stands, so an ASSIGN has something to be an assignment to. */
function roster(view: View, pid: number): string {
  const drivers = view.drivers[pid] ?? [];
  if (!drivers.length) return empty('No drivers.');
  return `<ul class="nlist roster">${drivers
    .map(
      (d) =>
        `<li><b>${esc(d.name)}</b> <em>${esc(d.id)}</em> ${money(d.wage)}/day, résumé ${
          d.resume
        } &mdash; ${d.truck ? `driving ${esc(d.truck)}` : '<span class="warn">unassigned</span>'}${
          d.notice ? ' <span class="warn">on notice</span>' : ''
        }</li>`
    )
    .join('')}</ul>`;
}

const LABOR_TEXT: Record<string, (r: any) => string> = {
  HIRE: (r) => `hired ${r.name} as ${r.driver}`,
  FIRE: (r) => `fired ${r.driver}`,
  QUIT: (r) => `${r.driver} quit`,
  POACH_ACCEPTED: (r) => `${r.driver} gave notice to seat ${r.from + 1}, joins day ${r.departs + 1}`,
  POACH_MATCHED: (r) => `kept ${r.driver} by matching the offer`,
  POACH_TRANSFER: (r) => `${r.driver} joined from seat ${r.from + 1} as ${r.new_id}`,
  POACH_REFUSED: (r) => `${r.driver} turned down the offer`,
  OUTBID: () => `outbid on a poach`,
};

function laborOutcome(view: View): string {
  if (view.night.pending) return empty('Awaiting the reply.');
  const log = view.night.outcome?.labor ?? [];
  if (!log.length) return empty('Nothing changed hands. Wage rises and assignments settle silently.');
  return `<ul class="nlist">${log
    .map((r) => {
      // Which seat a row belongs to is named differently per op: `player` for
      // hire/fire/outbid, `to` for an accepted poach or transfer, `employer`
      // for a refused or matched one. Only QUIT names no seat -- it is the driver's decision, and the
      // driver id already carries the seat it was leaving.
      const seat = r.player ?? r.to ?? r.employer;
      const who = seat != null ? seatChip(view, seat) : '';
      const text = LABOR_TEXT[r.op]?.(r) ?? r.op;
      return `<li>${who} ${esc(text)}</li>`;
    })
    .join('')}</ul>`;
}

// --- CONTRACTS ---------------------------------------------------------------

interface LotRow {
  lot: Listing;
  bids: (number | null)[];
  won: number | null;
  ask: number | null;
}

/** The board, each seat's bid on it, and who took it -- one row per lot.
 *
 * Split into offer/reply/outcome columns this would be three lists of lot ids
 * that the reader has to join by hand. The auction is a reverse auction, so a
 * LOWER bid wins; that is unintuitive enough without making the reader hunt.
 */
function lotRows(view: View): LotRow[] {
  const n = view.names.length;
  const board = [...(view.market.listings ?? []), ...(view.market.accounts ?? [])];
  const byLot = new Map<string, LotRow>();
  for (const lot of board) byLot.set(lot.id, { lot, bids: Array(n).fill(null), won: null, ask: null });

  view.night.actions.forEach((act, pid) => {
    for (const e of asList(act?.bids)) {
      const row = byLot.get(String(e[0]));
      if (row) row.bids[pid] = Number(e[1]);
    }
    for (const e of asList(act?.standing_bids)) {
      const row = byLot.get(String(e[0]));
      if (row) row.bids[pid] = Number(e[1]);
    }
  });
  for (const r of view.night.outcome?.auction ?? []) {
    const row = byLot.get(r.lot);
    if (row) {
      row.won = r.player;
      row.ask = r.ask;
    }
  }
  // Contested lots first, then bid-on, then the rest of the board: the reader
  // should not scroll past forty untouched lots to find the fight.
  const rank = (r: LotRow) => -r.bids.filter((b) => b != null).length;
  return [...byLot.values()].sort((a, b) => rank(a) - rank(b) || b.lot.reserve - a.lot.reserve);
}

function lotBoard(view: View): string {
  const rows = lotRows(view);
  if (!rows.length) return empty('No freight posted tonight.');
  const seatHeads = view.names.map((_n, pid) => `<th class="num">${seatChip(view, pid)}</th>`).join('');
  const body = rows
    .map((r) => {
      const l = r.lot;
      const bids = r.bids
        .map((b, pid) => {
          if (b == null) return `<td class="num pass">--</td>`;
          const winner = r.won === pid;
          return `<td class="num${winner ? ' win' : ''}" style="--seat-ink:${PLAYER_INK[pid] ?? INK_MUTED}">${money(b)}</td>`;
        })
        .join('');
      // A bid that loses has lost one of two quite different ways, and saying
      // "over reserve" for both is a lie half the time: a bid AT reserve is
      // admissible and still goes unawarded when the winner is already out of
      // truck-days or deck space. Read the reserve, not the outcome.
      const offered = r.bids.filter((b): b is number => b != null);
      const admissible = offered.some((b) => b <= l.reserve + 0.005);
      const result = view.night.pending
        ? '<span class="muted">pending</span>'
        : r.won != null
          ? `${seatChip(view, r.won)} <b>${money(r.ask ?? 0)}</b>`
          : !offered.length
            ? '<span class="muted">unsold</span>'
            : admissible
              ? '<span class="muted">no capacity</span>'
              : '<span class="muted">over reserve</span>';
      return `<tr>
        <td>${esc(l.id)}${l.kind === 'STANDING' ? ' <span class="tag">standing</span>' : ''}${l.bulk ? ' <span class="tag bulk">bulk</span>' : ''}</td>
        <td>${esc(l.district.replace(/_/g, ' ').toLowerCase())} <em>${esc(l.warehouse)}</em></td>
        <td class="num">${l.packages}</td>
        <td class="num">${l.truck_days.toFixed(2)}</td>
        <td class="num">${money(l.reserve)}</td>
        ${bids}
        <td>${result}</td>
      </tr>`;
    })
    .join('');
  return `<table class="ntable lots">
    <thead><tr>
      <th>Lot</th><th>Territory</th><th class="num">Pkgs</th><th class="num">Truck-days</th>
      <th class="num">Reserve</th>${seatHeads}<th>Awarded</th>
    </tr></thead>
    <tbody>${body}</tbody>
  </table>`;
}

function contractsNote(view: View): string {
  const caps = view.night.actions
    .map((a, pid) =>
      a?.max_lots != null ? `${view.names[pid]} capped at ${Number(a.max_lots).toFixed(2)} truck-days` : ''
    )
    .filter(Boolean)
    .join(' · ');
  return caps || 'lowest admissible bid wins';
}

// --- assembly ----------------------------------------------------------------

/** Trucks as they stand tonight: what CAPEX and LABOR are deciding about. */
function fleetStrip(view: View, pid: number): string {
  const trucks = view.trucks[pid] ?? [];
  if (!trucks.length) return empty('No trucks.');
  return `<div class="chips">${trucks
    .map((t) => {
      const title = `${t.id} ${t.type} ${t.status} -- fuel ${Math.round(t.fuel)}, ${Math.round(t.km_since_service)} km since service, ${
        t.driver ?? 'no driver'
      }, staged ${t.staged ?? 'nowhere'}`;
      return `<span class="chip truck-chip" title="${esc(title)}">
        <img src="${esc(truckSprite(t.type, pid))}" alt="" />${esc(t.id)}
        <em>${Math.round(t.fuel)}L</em>
      </span>`;
    })
    .join('')}</div>`;
}

export function renderNight(refs: NightRefs, view: View): void {
  const phase = view.phase;
  let body: string;

  if (phase === 'CAPEX') {
    body =
      section('On offer', 'used trucks and the shared rental pool', capexOffer(view)) +
      section(
        'Fleet tonight',
        'fuel, service and where each truck sleeps',
        seatColumns(view, (pid) => fleetStrip(view, pid))
      ) +
      section(
        'Submitted',
        'buy, rent, sell, service, refuel, stage',
        seatColumns(view, (pid) => capexReply(view, pid))
      ) +
      section('Settled', 'the engine’s capex log', capexOutcome(view));
  } else if (phase === 'LABOR') {
    body =
      section('On offer', 'tonight’s candidates, ask and résumé', laborOffer(view)) +
      section(
        'Payroll',
        'who is on the books and in which cab',
        seatColumns(view, (pid) => roster(view, pid))
      ) +
      section(
        'Submitted',
        'hire, fire, re-wage, poach, assign',
        seatColumns(view, (pid) => laborReply(view, pid))
      ) +
      section('Settled', 'hires, departures and poaches', laborOutcome(view));
  } else {
    body = section('Freight board', contractsNote(view), lotBoard(view));
  }

  refs.board.innerHTML = body;
}
