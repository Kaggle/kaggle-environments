/** The furniture both screens share: the top bar and the seat panels.
 *
 * Day and night are different pictures of the same game, so the frame around
 * them has to be literally the same markup -- a panel that moves two pixels or
 * changes its number format between phases reads as a different game, not a
 * different view of one.
 */

import { PLAYER_INK } from './theme';
import type { View } from './types';
import { clockLabel, money, spriteSrc, truckSprite, weatherSprite } from './utils';

export const esc = (s: unknown) => String(s).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

export interface TopbarRefs {
  clock: HTMLElement;
  dayLabel: HTMLElement;
  phase: HTMLElement;
  weather: HTMLImageElement;
  forecast: HTMLElement;
}

export interface SeatRefs {
  cash: HTMLElement;
  debt: HTMLElement;
  worth: HTMLElement;
  tallyCap: HTMLElement;
  tallies: Record<string, HTMLElement>;
  fleet: HTMLElement;
}

export interface ScreenRefs {
  top: TopbarRefs;
  panels: SeatRefs[];
}

export function topbarMarkup(): string {
  return `
      <header class="topbar">
        <span class="brand">KARGO</span>
        <span class="day" data-day>Day --</span>
        <span class="clock" data-clock>--:--</span>
        <span class="phase" data-phase>--</span>
        <span class="wx"><img data-weather alt="" /><span data-forecast class="forecast"></span></span>
      </header>`;
}

export function panelMarkup(pid: number, name: string): string {
  const tally = (k: string, label: string) =>
    `<div class="tally"><span class="dot dot-${k} dot-key"></span><span class="tally-n" data-tally="${k}">0</span><span class="tally-label">${label}</span></div>`;
  return `
    <section class="panel" data-player="${pid}" style="--seat-ink:${PLAYER_INK[pid] ?? '#6a635a'}">
      <header class="panel-head">
        <span class="seat"></span>
        <h2>${esc(name)}</h2>
      </header>
      <div class="worth" data-worth>--</div>
      <div class="money">
        <span>Cash <b data-cash>--</b></span>
        <span>Debt <b data-debt>--</b></span>
      </div>
      <div class="tally-cap" data-tally-cap></div>
      <div class="tallies">
        ${tally('delivered', 'delivered')}
        ${tally('late', 'late')}
        ${tally('refused', 'refused')}
        ${tally('failed', 'missed')}
      </div>
      <div class="fleet" data-fleet></div>
    </section>`;
}

/** The seat panels for one flank, in seat order.
 *
 * Two seats per side at a four-player table, one each at two. Splitting down
 * the middle rather than stacking all of them on the left keeps the map
 * centred and keeps each seat the same width in both modes -- a panel that
 * halves when a third player joins reads as a different component.
 */
export function flankMarkup(names: string[], side: 'left' | 'right'): string {
  const half = Math.ceil(names.length / 2);
  const seats = side === 'left' ? names.slice(0, half) : names.slice(half);
  const base = side === 'left' ? 0 : half;
  return `<div class="flank flank-${side}">${seats
    .map((n, i) => panelMarkup(base + i, n || `Player ${base + i + 1}`))
    .join('')}</div>`;
}

function seatRefs(el: HTMLElement): SeatRefs {
  const tallies: Record<string, HTMLElement> = {};
  el.querySelectorAll<HTMLElement>('[data-tally]').forEach((n) => {
    tallies[n.dataset.tally as string] = n;
  });
  return {
    cash: el.querySelector('[data-cash]') as HTMLElement,
    debt: el.querySelector('[data-debt]') as HTMLElement,
    worth: el.querySelector('[data-worth]') as HTMLElement,
    tallyCap: el.querySelector('[data-tally-cap]') as HTMLElement,
    tallies,
    fleet: el.querySelector('[data-fleet]') as HTMLElement,
  };
}

export function screenRefs(parent: HTMLElement): ScreenRefs {
  const q = <T extends Element>(sel: string) => parent.querySelector(sel) as T;
  return {
    top: {
      clock: q('[data-clock]'),
      dayLabel: q('[data-day]'),
      phase: q('[data-phase]'),
      weather: q('[data-weather]'),
      forecast: q('[data-forecast]'),
    },
    panels: Array.from(parent.querySelectorAll<HTMLElement>('.panel')).map(seatRefs),
  };
}

const FLEET_SPRITE: Record<string, string> = {
  DISABLED: 'truck_disabled',
  ORDERED: 'truck_ordered',
};

export function paintScreen(refs: ScreenRefs, view: View): void {
  const t = refs.top;
  t.dayLabel.textContent = `Day ${view.day + 1}`;
  t.clock.textContent = view.phase === 'DRIVING' ? clockLabel(view.minute) : 'overnight';
  t.phase.textContent = view.phase;
  t.weather.src = weatherSprite(view.traffic.weather);
  t.weather.alt = view.traffic.weather;
  t.forecast.textContent = `next: ${view.traffic.forecast.toLowerCase()}`;

  view.players.forEach((p, pid) => {
    const r = refs.panels[pid];
    if (!r) return;
    r.cash.textContent = money(p.cash);
    r.debt.textContent = money(p.debt);
    r.worth.textContent = money(p.net_worth);

    // The live counter, not `results` -- that only gains today's row at 18:00,
    // so a scoreboard read from it shows four zeroes all day.
    //
    // Overnight it is still YESTERDAY's: the engine clears it in `_start_day`,
    // which runs when CONTRACTS resolves. Unlabelled, a night panel reads as
    // "Day 2, 182 delivered" before a single truck has moved, so the caption
    // says which day the numbers belong to rather than letting the heading
    // above them imply the wrong one.
    const today = view.dayReports[pid];
    r.tallyCap.textContent =
      view.phase === 'DRIVING' ? 'today' : view.day > 0 ? `Day ${view.day} final` : 'no deliveries yet';
    const tallies: Record<string, number> = {
      delivered: today?.delivered ?? 0,
      late: today?.late ?? 0,
      refused: today?.refused ?? 0,
      failed: today?.failed ?? 0,
    };
    for (const [k, el] of Object.entries(r.tallies)) el.textContent = String(tallies[k] ?? 0);

    r.fleet.innerHTML = (p.fleet as any[])
      .map((t2) => {
        const src = FLEET_SPRITE[t2.status] ? spriteSrc(FLEET_SPRITE[t2.status]) : truckSprite(t2.type, pid);
        return `<span class="fleet-chip" title="${esc(`${t2.id} ${t2.type} ${t2.ownership} ${t2.status}`)}">
          <img src="${src}" alt="" /><span>${esc(t2.id)}</span>
        </span>`;
      })
      .join('');
  });
}
