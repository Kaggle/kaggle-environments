/** Sprite registry and the replay -> view transform. */

import event_ran_dry from './assets/sprites/event_ran_dry.png';
import incident_accident from './assets/sprites/incident_accident.png';
import incident_closure from './assets/sprites/incident_closure.png';
import incident_construction from './assets/sprites/incident_construction.png';
import incident_unreported from './assets/sprites/incident_unreported.png';
import step_p1 from './assets/sprites/step_p1.png';
import step_p2 from './assets/sprites/step_p2.png';
import step_p3 from './assets/sprites/step_p3.png';
import step_p4 from './assets/sprites/step_p4.png';
import stop_kerb from './assets/sprites/stop_kerb.png';
import truck_disabled from './assets/sprites/truck_disabled.png';
import truck_ordered from './assets/sprites/truck_ordered.png';
import van_p1 from './assets/sprites/van_p1.png';
import van_p2 from './assets/sprites/van_p2.png';
import van_p3 from './assets/sprites/van_p3.png';
import van_p4 from './assets/sprites/van_p4.png';
import warehouse from './assets/sprites/warehouse.png';
import warehouse_active from './assets/sprites/warehouse_active.png';
import weather_clear from './assets/sprites/weather_clear.png';
import weather_rain from './assets/sprites/weather_rain.png';
import weather_snow from './assets/sprites/weather_snow.png';

import type { History, KargoEvent, Market, Night, Observation, StopState, View } from './types';

const SPRITE_URLS: Record<string, string> = {
  event_ran_dry,
  incident_accident,
  incident_closure,
  incident_construction,
  incident_unreported,
  step_p1,
  step_p2,
  step_p3,
  step_p4,
  stop_kerb,
  truck_disabled,
  truck_ordered,
  van_p1,
  van_p2,
  van_p3,
  van_p4,
  warehouse,
  warehouse_active,
  weather_clear,
  weather_rain,
  weather_snow,
};

export function spriteSrc(name: string): string {
  return SPRITE_URLS[name] ?? '';
}

/** Seat liveries run p1..p4 -- the engine's 2- and 4-player tables. Anything
 * past that wraps rather than resolving to a missing file and drawing nothing:
 * a wrong-coloured truck is a smaller lie than an invisible one. */
export function truckSprite(type: string, player: number): string {
  const kind = type === 'STEP' ? 'step' : 'van';
  return spriteSrc(`${kind}_p${(player % 4) + 1}`);
}

export function weatherSprite(w: string): string {
  return spriteSrc(`weather_${String(w).toLowerCase()}`) || spriteSrc('weather_clear');
}

/** Engine minute (0 = 08:00) as a wall clock. */
export function clockLabel(minute: number): string {
  const total = 8 * 60 + Math.max(0, Math.round(minute));
  const h = Math.floor(total / 60) % 24;
  const m = total % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

export function money(n: number): string {
  const v = Math.round(Number(n) || 0);
  return `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString('en-US')}`;
}

/** An event's outcome, as a stop state. Truck faults are not stop outcomes. */
function outcomeOf(e: KargoEvent): StopState | null {
  switch (e.kind) {
    case 'DELIVER':
      return e.late ? 'late' : 'delivered';
    case 'REFUSED':
      return 'refused';
    case 'UNDELIVERED':
      return 'failed';
    case 'ABANDONED':
      return 'abandoned';
    default:
      return null;
  }
}

/** Worse outcomes win, so one refused door colours the whole block face.
 *
 * A segment holds several addresses and they can end differently. Showing the
 * last one to resolve would make the marker flicker and would hide the
 * expensive failure behind a cheap success. */
const SEVERITY: Record<StopState, number> = {
  pending: 0,
  delivered: 1,
  late: 2,
  abandoned: 3,
  failed: 4,
  refused: 5,
};

function worst(a: StopState | undefined, b: StopState): StopState {
  return a && SEVERITY[a] >= SEVERITY[b] ? a : b;
}

function stepsOfDay(replay: any, step: number): any[] {
  const steps: any[] = replay?.steps ?? [];
  const here = steps[step];
  if (!here) return [];
  const day = here[0]?.observation?.day ?? 0;
  const out: any[] = [];
  for (let i = 0; i <= step; i++) {
    if ((steps[i]?.[0]?.observation?.day ?? -1) === day) out.push(steps[i]);
  }
  return out;
}

/** The night's offer, reply and outcome, with the replay's off-by-one undone.
 *
 * A step records the observation an agent was shown AND the action it produced
 * in reply -- but the reply belongs to the PREVIOUS step's observation, because
 * the interpreter reads last step's actions, resolves them, and only then
 * publishes. So step 9 is labelled LABOR yet carries `{stage, fuel, fleet}`,
 * which is a CAPEX action answering step 8's board.
 *
 * Shifting the action forward by one puts each decision under the heading it
 * was made for. The outcome shifts with it: `history.capex` on step 9 is what
 * step 9's CAPEX resolution produced. On the replay's last step nothing has
 * answered yet, so the board is shown with the reply marked pending rather than
 * paired with a decision that was never made.
 */
function buildNight(replay: any, step: number, n: number): Night {
  const next = replay?.steps?.[step + 1];
  if (!next) return { actions: Array.from({ length: n }, () => ({})), outcome: null, pending: true };
  const actions = Array.from({ length: n }, (_v, p) => {
    const a = next[p]?.action;
    return a && typeof a === 'object' ? a : {};
  });
  return { actions, outcome: (next[0]?.observation?.history ?? null) as History | null, pending: false };
}

/** Fold one step of the replay into everything the renderer needs.
 *
 * Stop outcomes are accumulated from the start of the current day rather than
 * read off the current step: events are published only in the step that raised
 * them, so a marker drawn from one step alone would blink on for a frame and
 * vanish. Scanning the day also means scrubbing backwards is correct.
 */
export function buildView(replay: any, step: number, names: string[]): View | null {
  const steps: any[] = replay?.steps ?? [];
  const current = steps[step];
  if (!current?.length) return null;

  const obs: Observation = current[0].observation;
  if (!obs?.city) return null;

  const n = current.length;
  const trucks: View['trucks'] = [];
  const drivers: View['drivers'] = [];
  const segments: View['segments'] = [];
  const stops: View['stops'] = [];
  const dayReports: View['dayReports'] = [];
  const blank = { delivered: 0, late: 0, failed: 0, refused: 0, revenue: 0, cost: 0 };
  for (let p = 0; p < n; p++) {
    const priv = current[p]?.observation?.private;
    trucks.push(priv?.trucks ?? []);
    drivers.push(priv?.drivers ?? []);
    segments.push(priv?.segments ?? []);
    stops.push(new Map<string, StopState>());
    dayReports.push(priv?.day_report ?? { ...blank });
  }

  // Every segment on today's manifest starts pending, then events overwrite.
  for (let p = 0; p < n; p++) {
    for (const seg of segments[p]) stops[p].set(seg.id, 'pending');
  }

  const events: KargoEvent[] = [];
  for (const s of stepsOfDay(replay, step)) {
    for (let p = 0; p < n; p++) {
      for (const e of (s[p]?.observation?.private?.events ?? []) as KargoEvent[]) {
        if (s === current) events.push(e);
        const outcome = outcomeOf(e);
        if (outcome && e.segment) stops[p].set(e.segment, worst(stops[p].get(e.segment), outcome));
      }
    }
  }

  return {
    step,
    day: obs.day,
    block: obs.block,
    minute: obs.minute,
    phase: obs.phase,
    city: obs.city,
    traffic: obs.traffic,
    players: obs.public ?? [],
    trucks,
    drivers,
    segments,
    stops,
    dayReports,
    events,
    names,
    market:
      obs.market ?? ({ listings: [], accounts: [], used: [], rentals: {}, candidates: [], fill_ceiling: 1 } as Market),
    night: buildNight(replay, step, n),
  };
}
