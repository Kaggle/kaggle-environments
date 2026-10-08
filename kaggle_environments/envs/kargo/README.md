# Kargo

A last-mile delivery business sim for 2 or 4 players. Players bid nightly for a shipper's freight, then dispatch trucks across a congested city.

## Overview

Each night players bid in a sealed-bid reverse auction for delivery lots. A listing gives the origin warehouse, destination district and territory, package count and penalties, but not the addresses, their service times, or their delivery windows. Those arrive at 08:00 with the manifest. Each day players load their lots onto trucks, route the trucks, and may abandon freight.

All freight comes from one external shipper whose demand grows over the episode along an undisclosed path. Its prices rise when demand exceeds the field's capacity and fall when capacity exceeds demand.

The episode is **60 days**. Each day opens with 3 overnight steps and then runs 5 driving blocks, for **480 turns** after the initial state. Score is net worth at the end.

## Time

| Unit | Value |
|---|---|
| Decision block | 2 simulated hours; players act once per block |
| Driving day | 5 blocks, 08:00 to 18:00 |
| Straight time | blocks 0-3 (08:00-16:00) |
| Overtime | block 4 (16:00-18:00), wages 1.5x, driver stats decay 2x |
| Overnight | 3 ordered steps: `CAPEX`, `LABOR`, `CONTRACTS` |
| Episode | 60 days x (3 + 5) = 480 turns, plus the initial state = 481 steps |

The simulation is continuous; service times are real numbers. The block sets how often players act. The observation after the initial reset is day 0's `CAPEX`; the episode ends at day 59's 18:00 close, which is the final observation.

Day 0 is a Monday. The day-of-week traffic multiplier keys off the day index; every seventh day is a Sunday and is driven like any other.

### Overnight steps

Each resolves and publishes its results before the next opens.

| Phase | Players decide | Published before the next phase |
|---|---|---|
| `CAPEX` | buy / finance / rent / sell trucks, buy used, service, refuel, break standing accounts, stage trucks | every player's fleet roster, and a log of fleet operations (buys, rentals, sales, services, breaks). Refuelling and staging are not logged |
| `LABOR` | hire, set wages, assign, poach, fire | driver rosters (name, tenure, résumé rating, notice flag, `departs`) and a log of hires, firings, poach outcomes and quits -- never anyone's pay |
| `CONTRACTS` | sealed-bid auction on tomorrow's spot lots and on standing accounts | every bid at or under reserve, and every award |

## The city

The city is two levels.

**The arterial grid** is 20x20 intersections (400 nodes, 760 edges) spaced 1.5 km. This is the travel network -- congestion, routing and warehouse deadhead live here. Node id is `row * 20 + col`; edge id is its index in `city.edges`.

| Road class | Free-flow min/edge | Free-flow km/h | Capacity (veh/min) | Share |
|---|---:|---:|---:|---:|
| `ARTERIAL` | 2.0 | 45 | 12 | every 4th row/col |
| `COLLECTOR` | 3.6 | 25 | 6 | 35% of the rest |
| `LOCAL` | 6.0 | 15 | 2.4 | remainder |

Every class is open to every vehicle.

**The interior street grid** is where addresses live. A lot's doors sit in a compact territory hung off one arterial intersection. Interior streets carry no modelled congestion -- 28 km/h with a 1.27 grid-detour factor.

### Districts

The grid is partitioned into six districts of five archetypes (`SUBURBS_N` and `SUBURBS_S` share one). Their **positions and sizes are drawn per episode** inside loose bands (downtown central, suburbs north and south, industry on the west edge); their **characters are fixed forever**.

| District | Character | `C_d` (structural load) | Expected cost at 17:30, Monday |
|---|---|---:|---:|
| `DOWNTOWN` | towers, loading docks, no parking | 1.02 | 1.93x |
| `RIVERSIDE` | dense mid-rise walk-ups | 0.88 | 1.51x |
| `MIDTOWN` | mixed commercial | 0.78 | 1.32x |
| `SUBURBS_N/S` | detached, long driveways | 0.55 | 1.08x |
| `INDUSTRIAL` | warehouses, docks | 0.42 | 1.03x |

Fixed to the archetype: `C_d`, the service-time distribution, addresses per block, commercial share, window rates. Drawn per episode: where each district sits, its patch size, and the four warehouse locations (at least 9 grid cells apart when the draw allows).

### Addresses and segments

A lot's manifest is a set of **segments** (block faces), each carrying one or more **doors** (addresses). A segment has a position `pos = [x, y]` in km, drawn uniformly inside a square of side `sqrt(area)` whose corner is the lot's `anchor` intersection. An address has a position `t` in `[0, 1]` along its segment.

Moving between segments is interior driving: Manhattan distance between `pos` values x 1.27 at 28 km/h, times the driver's speed multiplier, plus 0.75 min to park. Moving between doors on one segment is a walk of `block_m x |Δt|` at 1.3 m/s. Every lot on one `(warehouse, district)` pair shares the same anchor, so their territories overlap.

A stop is a door, not a dwelling: a 900-unit tower is one stop at its package room. Units per door vary by district.

| District | Block grid | Doors/segment | Units/door | Stops/segment | Pkg/stop | Service/stop |
|---|---:|---:|---:|---:|---:|---:|
| `DOWNTOWN` | 80 m | 6 | 24 | 5.3 | 2.4 | 4.6 min |
| `RIVERSIDE` | 110 m | 24 | 6 | 5.5 | 1.3 | 2.4 min |
| `MIDTOWN` | 160 m | 30 | 2.5 | 4.2 | 1.1 | 1.9 min |
| `SUBURBS_N/S` | 250 m | 28 | 1.05 | 2.1 | 1.05 | 1.6 min |
| `INDUSTRIAL` | 450 m | 16 | 1 | 4.1 | 1.2 | 7.5 min |

About 9% of units get a parcel on a given day (30% in `INDUSTRIAL`, which is B2B). Each address's true service time is the district figure x U(0.6, 1.4), then x the driver's service multiplier.

### What a truck-day looks like

A truck-day is what the engine's solver fits into one shift out of a given dock: `(480 - 45 load - 2 x deadhead) x (1 - 0.33 x windowed share)` minutes of interior work. Figures at the median near-dock deadhead (below), minutes:

| District | Stops/day | Pkg/day | Segments | Territory | Service | Walk | Drive | Park |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `DOWNTOWN` | 73 | 178 | 14 | 0.04 km² | 336 | 10 | 3 | 10 |
| `RIVERSIDE` | 133 | 172 | 24 | 0.15 km² | 319 | 24 | 7 | 18 |
| `MIDTOWN` | 154 | 172 | 36 | 0.47 km² | 293 | 46 | 15 | 27 |
| `SUBURBS_N/S` | 143 | 150 | 67 | 2.08 km² | 229 | 78 | 43 | 50 |
| `INDUSTRIAL` | 45 | 52 | 11 | 1.10 km² | 338 | 38 | 13 | 8 |

Walk is time on foot between doors on one segment.

## Travel time

The Bureau of Public Roads volume-delay function, per arterial edge:

```
t_edge = t_free * (1 + 0.15 * (v/c)^4) * weather * driver_speed
v/c    = (background(district, m, day) + PCU * n_edge / capacity) / incident_capacity
```

`n_edge` is the number of player trucks on the edge at that minute, from every truck already driven this block. Players' trucks are driven in a shuffled order each block. `PCU` is set by `truckPcu` (1, 3 or 8). `incident_capacity` is 1 unless an accident is active on the edge. `district` is the district of the edge's first node.

At `v/c = 1.0` an edge costs 1.15x, at 1.5 1.76x, at 2.0 3.4x.

Background `v/c`:

```
background(district, m, day) = C_d * P(m) * DOW(day) * L_day * (1 + 0.085 * e_t)
P(m) = 0.55 + 0.60 * exp(-(m - 10)^2 / (2 * 60^2)) + 0.92 * exp(-(m - 570)^2 / (2 * 100^2))
```

`m` is minutes since 08:00.

| Factor | What it is | Observed |
|---|---|---|
| `C_d` | district structural load | fixed, listed above |
| `P(m)` | time-of-day shape: AM tail, midday trough, PM peak at 17:30 | fixed, same every day |
| `DOW(day)` | Mon-Sun 1.00, 1.03, 1.05, 1.07, 1.12, 0.90, 0.70 | fixed |
| `L_day` | whole-day level, lognormal mean 1, σ 0.16, drawn at 08:00 | no, only through congestion reports |
| `e_t` | AR(1) noise, unit variance, ρ 0.94 per 6-minute tick, reset to 0 each morning | no, only through congestion reports |

The PM peak falls in overtime. On a Monday downtown, expected cost is 1.02x at 12:00, 1.20x at 15:30, 1.36x at 16:00 and 1.93x at 17:30 (2.45x on a Friday). Midday is nearly deterministic (p10/p90 of 1.01-1.04); at 17:30 the p10/p90 spread is 1.28x-2.81x.

### Congestion is reported in buckets

`v/c` is not observed. Each edge reports one of six ordinal levels, sampled at the start of the block:

| Level | `v/c` | Cost multiplier (before weather and driver) |
|---|---|---|
| `FREE` | < 0.70 | 1.00 - 1.04 |
| `LIGHT` | 0.70 - 1.00 | 1.04 - 1.15 |
| `MODERATE` | 1.00 - 1.25 | 1.15 - 1.37 |
| `HEAVY` | 1.25 - 1.50 | 1.37 - 1.76 |
| `SEVERE` | 1.50 - 1.80 | 1.76 - 2.57 |
| `GRIDLOCK` | >= 1.80 | 2.57 and up |

Cut-points are cost-spaced, not `v/c`-spaced. The report includes accidents and player-truck load.

### Routing

Players order targets; between consecutive targets the engine runs A* on current edge costs (background, player trucks, accidents, closures, weather), re-planned at every node it reaches. An edge once entered is committed, even past the end of the block.

Arterial driving happens between a warehouse and a territory's anchor, and on any `via` detour. Everything inside a territory is interior driving.

A `via` entry in a `route` pins a grid node the truck must pass through:

```json
"route": [{"via": 247}, "lot_4_seg_0", "lot_4_seg_3"]
```

`via` takes an integer node id. It is never served and costs no service time. A target no path reaches is dropped from the route.

### Incidents

| Event | Trigger | Effect | Notice |
|---|---|---|---|
| Accident | per 6-min tick, hazard ∝ (background `v/c`)² x weather | capacity to 30-50% for 30 min to 2.5 h | in the incident feed 10 min after onset, with its end time |
| Construction | 0-3 random edges per day | edge closed all day | drawn at 08:00, in the feed from block 0 |
| Emergency closure | ~0.25 per day | edge closed 1.5-5 h | in the feed immediately |
| Weather | per-day draw: `CLEAR` 68%, `RAIN` 24%, `SNOW` 8% | travel x1.00 / 1.12 / 1.25; accident hazard x1.0 / 1.6 / 2.4 | drawn at 08:00 |

A closed edge is impassable. `traffic.weather` is today's weather during a driving day and the last driven day's overnight (`null` on day 0's night). `traffic.forecast` names the next driving day's weather: the coming day overnight, tomorrow during a day. It copies the true weather with probability 0.75, otherwise it is a uniform draw over the three kinds (right about 83% of the time overall), and is fixed once published.

## Delivery windows

Some doors carry a promised time window. Arriving early means **waiting**.

| Kind | Miss it |
|---|---|
| `DOCK` | a scheduled receiving appointment. Closed on arrival: `SKIP` fails the door at once as undelivered ($45/package) and the truck moves on; `ATTEMPT` knocks. Half of dock doors refuse any late truck; the rest accept one up to a hidden 15-60 minutes past the window. Inside that grace the door is served; past it, it is **refused**: $45/package plus a $45 refusal fee |
| `PROMISED` | a slot the customer was quoted. Delivered anyway, **$4/package premium forfeited** |

| District | Commercial | `DOCK` rate / width | `PROMISED` rate | Windowed share of packages | `DOCK` / `PROMISED` packages per truck-day |
|---|---:|---|---:|---:|---|
| `DOWNTOWN` | 55% | 45% / 120 min | 40% | 43% | 44.1 / 32.0 |
| `RIVERSIDE` | 15% | 45% / 120 min | 40% | 41% | 11.6 / 58.5 |
| `MIDTOWN` | 40% | 40% / 120 min | 30% | 34% | 27.5 / 31.0 |
| `SUBURBS_N/S` | 5% | 40% / 120 min | 20% | 21% | 3.0 / 28.5 |
| `INDUSTRIAL` | 90% | 20% / **240 min** | 20% | 20% | 9.4 / 1.0 |

A listing's `dock_packages` and `promised_packages` are drawn per package (commercial x `DOCK` rate, residential x `PROMISED` rate). The manifest then places windows on doors at random until those budgets run out, so each of the manifest's two windowed package counts matches the listing about 80% of the time (both together about 66%), and a mismatch is usually a few packages, occasionally up to about 15.

**Every windowed door on a block face shares a start.** Starts fall on the half hour: 08:00-12:30 in `DOWNTOWN`, `RIVERSIDE` and `MIDTOWN`, 08:00-12:00 in `SUBURBS_N/S`, 08:00-09:30 in `INDUSTRIAL`. `PROMISED` windows are always 120 min wide, so a `DOCK` and a `PROMISED` door on one `INDUSTRIAL` face open together and shut apart. Every window closes by 14:30.

## Fleet

| Type | Capacity (parcel-units) | Buy | Rent/day | Own/day | Depreciation/day | Fuel (units per arterial min) | Tank | Service |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `VAN` | 200 | $38,000 | $170 | $60 | $11 | 1.0 | 640 | $100 |
| `STEP` | 340 | $62,000 | $270 | $92 | $18 | 1.6 | 1,000 | $140 |

A lot is 0.15-0.70 of a truck-day, and the freight on a truck at any moment is from one `(warehouse, district)` pair. A full truck-day on any pair is at most ~196 parcel-units (`INDUSTRIAL` bulk is 1.5 units a package, `SUBURBS_N/S` 1.3, `MIDTOWN` 1.1).

**Own, finance, or rent.**

| | Daily cash | Upfront | Lead time | Exit |
|---|---|---|---|---|
| Own outright (`BUY`) | $60 / $92 | full price | drives the same day | `SELL`: book value less a 25-45% haircut |
| Finance (`FINANCE`) | $60 / $92 + principal of price/365 + 0.035%/day interest | 20% | drives the same day | `SELL`: proceeds less remaining principal |
| Rent (`RENT`) | $170 / $270 from arrival | none | **2-3 days** | `RETURN_RENTAL` after 7 rental days |

`BUY`, `FINANCE` and `BUY_USED` are refused if the payment would take cash below -$20,000, or if the player already has 20 trucks (ordered rentals included). **No new truck comes with a driver** -- bought, used or rented, it needs a `HIRE` and an `ASSIGN`.

`RENT` draws from a nightly pool per type of `6 - floor(rented trucks in the field / 2) ± 1`; the night's requests from all players are filled in a shuffled order. A rented truck shows `ORDERED` until it arrives, at `wh_0` unless it was staged elsewhere.

A used market posts 2 listings each night during `CAPEX`: a random type at 25-45% off new, 200-1,400 days old, with its odometer shown. Its `km_since_service`, drawn uniformly from 0-250, is revealed only after purchase. A listing two players buy goes to one of them at random.

### Fuel and maintenance

Fuel burns only on arterial edges, at the type's rate per minute of edge time. Interior driving burns none. `BULK_REFUEL` during `CAPEX` fills the tank at $0.42/unit; there is no other refuelling. Running dry costs $150 and 60 minutes and leaves the tank half full.

`km_since_service` grows by distance driven x the driver's `CARE` multiplier: 1.5 km per arterial edge plus interior km x 1.27. The odometer counts raw distance.

At the 18:00 close, a truck with `km_since_service` at or past **250** is grounded: status `DISABLED`, and a private `SERVICE_DUE` event. `SERVICE` during that night's `CAPEX` ($100 / $140) resets `km_since_service` and returns it to `IDLE`, and it works the next day. Unserviced, it takes no lots and stays grounded until serviced. `SERVICE` works on any truck on any night.

## Drivers

Three hidden stats, 1-100.

| Stat | Effect |
|---|---|
| `SPEED` | travel-time multiplier `1.14 - 0.26 x SPEED/100`, arterial and interior |
| `CARE` | wear multiplier on `km_since_service`, `1.60 - 1.20 x CARE/100` |
| `SERVICE` | service-time multiplier `1.30 - 0.60 x SERVICE/100` |

Stats are drawn around a common base; a driver's `SPEED` deviation is offset by opposite deviations in `CARE` and `SERVICE`. Effective stats fall 1.6 points per hour since 08:00, overtime hours counting double, and a further `0.4 x (35 - morale)` when morale is below 35.

**Stats are hidden**, to the owner as well. What is public is a **résumé rating** (1-5), drawn once from the stat mean plus noise and never updated. It travels with the driver between employers.

### Wages, morale, quitting

Players set each driver's daily wage (floor $120). Pay is prorated to minutes worked: `wage x min(worked, 480)/480` plus overtime minutes at 1.5x the per-minute rate. `worked` is the last minute the truck spent driving, serving or waiting on a window; a truck with no work pays nothing. A driver on no truck, or on a truck that could not run that day (not arrived, `DISABLED`), draws a retainer of 50% of their wage instead. Rosters are capped at 30 drivers, counting drivers departing to join.

Each driver has a hidden reservation wage. Morale starts at 60 and updates nightly after the `LABOR` actions resolve:

```
morale += 0.06 * (wage - reservation) - 0.02 * overtime_minutes + 0.4 - 0.25 * restlessness
```

Morale is clamped to 0-100. `overtime_minutes` is cumulative and halves nightly, and restlessness decays x0.85 nightly. Below 22 the driver's public `notice` flag is set; below 8 they quit that same night. Notice and quit can land on the same night.

The `LABOR` pool posts 2-4 candidates nightly, with asking wages of $173-348. Quality correlates with asking wage, imperfectly. `HIRE` is a sealed offer, resolved after every player's other labor actions: each candidate goes to the highest offer that meets their hidden reservation (0.88-1.10x their ask), ties broken at random. A hired driver starts unassigned.

### Poaching

Every player's drivers are public by name, tenure, résumé rating, notice flag and `departs`. True stats, pay and morale are hidden.

During `LABOR` a player may `POACH` a rival's driver with a wage offer, at most one attempt per rival per night (later entries against the same rival are ignored). Only the highest offer per driver is considered (ties random); a player at the roster cap cannot poach, and an offer under $120 is always refused. The driver weighs the raise over their current pay, plus their restlessness, against their morale and tenure.

A driver who accepts is **departing**: they keep working for their employer for 2 more days, and their public roster entry shows `departs` (bidder and day). At the `LABOR` step on that day they join the bidder unassigned, at the offered wage, with tenure reset to 0 and a new id. The employer keeps the driver by setting their `WAGE` at or above the offer in any `LABOR` step up to and including that one. Firing a departing driver sends them to the bidder at once, without severance. A departing driver cannot be poached again.

The night's labor log is public: `POACH_ACCEPTED` shows both players and the departure day, `POACH_REFUSED` shows the employer and the driver but not the bidder or amount, each losing bidder is logged as `OUTBID` by player index, and a departure ends as `POACH_MATCHED` or `POACH_TRANSFER`.

A refused offer raises the driver's restlessness by 14, which lowers morale and raises the chance of accepting a later offer. Restlessness decays x0.85 nightly; tenure accumulates nightly and lowers the chance of accepting.

## Contracts and the auction

**4 communal warehouses, shipper-owned, placed per episode** (`wh_0`-`wh_3`). Every night the shipper posts lots for the next day.

### The shipper

Every quantity below is drawn once per episode and never published. Observations show reserves, retry counts and awards, not these parameters.

**Demand.** Fresh truck-days posted per night:

```
demand = starting trucks in the field x trend(day) x regime x weekday x exp(noise)
trend  = lo + (hi - lo) / (1 + exp(-k x (day - mid)))
```

| Factor | Draw |
|---|---|
| `lo`, `hi` | 0.20-0.30 and 0.55-0.85 truck-days per starting truck |
| `mid`, `k` | 30-65% of the way through the episode; 0.10-0.30 per day |
| regime | `SLUMP` 0.8 / `NORMAL` 1.0 / `BOOM` 1.2 (±5%); each night it holds with probability 0.90-0.96, otherwise moves one step |
| weekday | Mon-Sun around 1.05, 1.05, 1.00, 1.00, 1.10, 0.75, 0.55, jittered ±10% |
| noise | AR(1), ρ 0.70, σ 0.07 |

Demand scales with the field's starting fleet, not its current one. Held standing accounts count against it.

Each pair has a demand weight that random-walks around a hidden home level. Fresh lots land on about one pair per 1.5 truck-days posted, drawn by weight, as lots of U(0.15, 0.70) truck-days, at most 160 listings a night. Retries take places first; a retry beyond the cap is lost as if past its patience.

**Price.**

```
reserve = base price x lot size x index x (1 + markup)^retry x U(0.96, 1.04)
index   = exp(market + deviation), clamped to 0.5-3.0
```

The base price is the pair's solved price (see Base prices).

- **Market level.** Each night it moves toward `elasticity x ln(demand / (supply x throughput))` at `speed`, plus noise. `supply` comes from the public roster: per player, trucks that have arrived and are not grounded, capped by the number of drivers. Hidden draws: throughput 0.30-0.42, elasticity 0.50-0.90, speed 0.15-0.35.
- **Territory deviation.** Rises with the share of the pair's last-posted packages that went unserved (rate 0.08-0.16). Falls when several carriers bid under its reserve while the market as a whole had slack (rate 0.15-0.35). Each night it is pulled 10-30% back toward zero, and it is capped at ±0.5.

**Retries.** A package that goes unsold (no award) or undelivered (failed, refused, abandoned, or never loaded) is posted again the next night. It comes back as a `SPOT` lot of the remaining packages, with `retry` counting up and the reserve marked up 6-18% per retry. Each fresh lot carries a hidden patience of 1 to 2-5 retries. Past it, the packages are lost and the pair's demand weight drops 4-10%. An unsold standing account comes back the same way.

### Where a truck starts the day

Each pair is a fixed **territory**: an anchor intersection drawn from the six nodes of that district nearest the warehouse. One-way deadhead from warehouse to anchor is priced at 45 km/h on Manhattan distance and runs 0-60 minutes (median 16). A lot is priced by its `(warehouse, district)` pair, not its district. The pair also sets the lot size, since longer deadhead leaves less of the shift for deliveries. Medians over 150 maps, nearest and farthest warehouse:

| District | Near deadhead | Far deadhead | Pkg near | Pkg far | Near->far | Extra deadhead (both ways) | % of shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DOWNTOWN` | 8 min | 30 min | 178 | 159 | -11% | 44 min | 9% |
| `RIVERSIDE` | 4 min | 32 min | 172 | 150 | -13% | 56 min | 12% |
| `MIDTOWN` | 2 min | 24 min | 172 | 154 | -10% | 44 min | 9% |
| `SUBURBS_N` | 2 min | 34 min | 150 | 128 | -15% | 64 min | 13% |
| `SUBURBS_S` | 2 min | 32 min | 150 | 129 | -14% | 60 min | 12% |
| `INDUSTRIAL` | 4 min | 36 min | 52 | 44 | -15% | 64 min | 13% |

Starting trucks sit at `wh_0`, `wh_1`, `wh_2`. At 18:00 every truck that is not already `DISABLED` is placed, free, at the warehouse nearest its position. `stage` during `CAPEX` moves a truck to any warehouse overnight, free, before the auction resolves.

### Loading

At 08:00 every lot a player won waits at its warehouse. A truck takes lots on with `load` (see Driving block), in any block. A truck not at the lot's warehouse first drives there over the arterial grid, with the usual time, fuel and congestion. Loading takes 45 minutes per lot, and a lot loads whole onto one truck.

A load is refused, with a private `LOAD_REFUSED` event giving the reason, if:

| Reason | Condition |
|---|---|
| `NO_DRIVER` | the truck has no driver |
| `NOT_AT_DOCK` | the lot is not waiting at its warehouse (unknown, already loaded, or written off) |
| `PAIR` | the truck carries freight from a different `(warehouse, district)` pair |
| `CAPACITY` | the lot's parcel-units do not fit the deck space left |
| `UNREACHABLE` | no open path reaches the warehouse |
| `TOO_LATE` | loading would not finish before 16:00 |

A successful load raises `LOADED`. A truck that has delivered or written off everything on board can load another pair the same day. Lots still at a warehouse at 18:00 fail like any undelivered freight.

### What a lot listing discloses

**Disclosed:** `id`, `kind` (`SPOT` / `STANDING`), `warehouse`, `district`, `anchor` node, `packages`, `stops`, `truck_days`, territory `area` (km²), `parcel_units`, `deadline` (480 = 16:00), `payout_per_package` at reserve, `late_penalty` ($6), `fail_penalty` ($45), `dock_packages`, `promised_packages`, `reserve`, `retry` (0 for fresh freight). Standing accounts add `term_options`.

**Not disclosed:** the individual addresses, their positions, their service times, or when any window falls. Those arrive in the private manifest at 08:00. Each address's `service_estimate` is its true service time x U(0.7, 1.3), redrawn in every observation.

### Auction format

First-price sealed-bid **reverse** auction: each bid is the price the player will accept, lowest ask wins, and the winner is paid their own ask, as `ask / packages` per package delivered. Bids above the reserve are discarded, and only each player's lowest bid per lot counts. Ties break at random from the episode seed. The bid book (every bid at or under reserve, per lot) is published for the following driving day; the award list stays up until the next `CONTRACTS`.

Players may bid on any number of lots and declare `max_lots` alongside. The engine awards the lowest bids, then trims each player back to their constraints, keeping their best-margin wins (ask over the lot's expected cost: the pair's `E[cost]` from Base prices times the lot's `truck_days`) and releasing the rest to the next-lowest bidder, repeating until stable.

Bids beyond a player's limits carry no penalty. A player who bids on more lots than they can run keeps the best-margin feasible set of whatever they win, so extra bids act as backups for lots lost to lower asks.

**`max_lots` is denominated in truck-days, not lots.** It defaults to 1.00 per truck and is capped at 3.00 per truck, counting only trucks that can run tomorrow: arrived, not `DISABLED`, with a driver. A player's wins in one night must also satisfy:

- at most 1.00 truck-day per `(warehouse, district)` pair
- no more pairs than trucks
- each pair's parcel-units fit a distinct truck's deck

Tonight's standing-account wins count against these, and so do tomorrow's lots from accounts already held.

### Standing accounts

Each night a third of fresh listings (rounded) post as `STANDING` accounts; retries never do. Accounts are bid only through `standing_bids`; a plain `bids` entry on an account is ignored. An account's `payout_per_package` is at its discounted reserve.

| | `SPOT` | `STANDING` |
|---|---|---|
| Term | tomorrow | 5, 10 or 20 consecutive days, bidder's choice |
| Rate | bid per lot | bid per lot-day, fixed for the term |
| Volume | this lot only | one lot of the listed size per day, same pair, fresh manifest |
| Reserve | the shipper's price for the lot | **8-15% below** that |
| Exit | n/a | `BREAK`: fee = 3 x rate, any `CAPEX` |

An account's first lot runs the morning after the award, like a spot lot. A held account posts its lot to the holder every day of the term whether or not the holder can run it. Held accounts appear in each player's public roster with `remaining` days, and in `history.standing` with live `remaining` days.

### Delivery outcomes

| Outcome | Payment |
|---|---|
| Delivered by the deadline (16:00) | payout (`ask / packages`) per package |
| Delivered inside a `PROMISED` window | payout **plus $4/package** |
| Delivered after the deadline | payout minus $6/package |
| `DOCK` window closed, `on_missed_window: SKIP` | payout forfeited, $45/package |
| `DOCK` window closed, `on_missed_window: ATTEMPT`, past the door's grace | **refused** -- $45/package plus $45 |
| `abandon`, block 0 only | payout forfeited, $15 per parcel-unit |
| Undelivered at 18:00, on a truck or still at the warehouse | payout forfeited, $45/package |

There is no carrying work into tomorrow. Credits and penalties settle at the 18:00 close. Every package not delivered goes back to the shipper as a retry.

### Abandoning freight

In block 0 (08:00-10:00) a top-level `abandon` list writes off held freight before it is driven. It takes lot, segment or door ids. The fee is $15 per parcel-unit, charged at the 18:00 close, and the payout is forfeited. The packages go back to the shipper. Outside block 0 the list is ignored.

| District | Parcel-units / package | Abandon fee / package | Fail penalty / package |
|---|---:|---:|---:|
| `DOWNTOWN`, `RIVERSIDE` | 1.0 | $15.00 | $45 |
| `MIDTOWN` | 1.1 | $16.50 | $45 |
| `SUBURBS_N/S` | 1.3 | $19.50 | $45 |
| `INDUSTRIAL` | 1.5 | $22.50 | $45 |

There is no brokerage and no player-to-player resale.

## Actions

Every action is a JSON object. Fields the current phase does not read are ignored, and so is any entry of the wrong shape: ids must be strings of up to 64 characters, numbers must be finite (and are clamped: wages to $0-2,000, asks to $0-1,000,000), and asks below zero are dropped. Lists are capped: 50 fleet ops, 100 labor ops, 300 bids, 100 standing bids, 200 route entries per truck, 500 abandon ids.

### Driving block

Each truck carries a plan: lots to load, an ordered route, and settings.

The routing unit is the segment. Doors cluster onto segments and within a segment are served in `t` order, so a 45-154 stop day is 11-67 segments. A segment carrying both windowed and free doors can be **split** with `only`, and repeating a segment in one route is legal.

```json
{
  "trucks": {
    "T1": { "load": ["lot_4"],
            "route": [{"via": 247}, "lot_4_seg_0",
                      {"seg": "lot_4_seg_5", "only": "windowed"},
                      "lot_4_seg_2", "lot_4_seg_5", "lot_4_a_17"],
            "on_missed_window": "SKIP",
            "wait_cap": 20,
            "then": "RETURN" },
    "T2": { "hold": true }
  },
  "abandon": ["lot_9", "lot_4_a_9"]
}
```

| Field | Meaning |
|---|---|
| `load` | lot ids to load, in order, before the route continues (see Loading) |
| `route` | ordered targets: a segment id, an address id, `{"seg": id, "only": "windowed" \| "free"}`, or `{"via": node}` |
| `on_missed_window` | on reaching a `DOCK` door whose window has closed: `SKIP` (default, the door fails at once as undelivered) or `ATTEMPT` (served if inside the door's hidden grace, refused otherwise) |
| `wait_cap` | maximum minutes to idle for a window to open; a door further off is skipped for now. Default 20 |
| `then` | when the route empties: `RETURN` (default) drives to the warehouse the truck last loaded at once the truck is empty; any other value waits |
| `hold` | `true`: the truck does not move this block |

Route mechanics:

- A target resolves to the doors on it that this truck still carries. A target with none is dropped.
- After a target is worked it leaves the head of the route. If some of its doors are still servable (skipped for `wait_cap`, window not yet closed), it goes to the back, at most twice per segment per day.

**Plans persist.** Omitting a truck means "carry on with the standing plan and remaining route." Sending a truck replaces all its plan fields; its route is replaced only if `route` is present, and its pending loads only if `load` is present. Loads are worked before the route. `hold: true` stays in force until the plan is re-sent without it. A truck without a driver does not move.

`abandon` is top-level, not per truck, and read only in block 0.

### `CAPEX`

```json
{ "fleet": [["BUY", "VAN"], ["FINANCE", "STEP"], ["BUY_USED", "used_3_0"],
            ["RENT", "VAN"], ["SELL", "T3"], ["RETURN_RENTAL", "T5"],
            ["SERVICE", "T1"], ["BREAK", "acct_1"]],
  "fuel":  [["BULK_REFUEL", "T1"]],
  "stage": { "T1": "wh_0", "T2": "wh_2" } }
```

Each player's `fleet` entries resolve in order, except `BUY_USED` and `RENT`, which resolve after every player's other fleet entries; then `fuel`, then `stage`. `SELL` applies to owned and financed trucks only, and unassigns the truck's driver. `stage` moves an idle truck, or a rental arriving for that day's driving, to the named warehouse; a grounded truck stays put unless `SERVICE` cleared it earlier in the same step.

### `LABOR`

```json
{ "labor": [["HIRE", "cand_3_1", 240], ["ASSIGN", "D0_4", "T4"],
            ["WAGE", "D0_1", 265], ["POACH", 1, "D1_2", 310], ["FIRE", "D0_3"]] }
```

`HIRE` takes a candidate id and a daily wage. `ASSIGN` puts a driver on a truck, unseating whoever drove either before. `POACH` takes the target player index, the driver id, and the offered daily wage. `FIRE` charges severance of 3 days' wage immediately. Entries resolve in order; then departures settle, hires resolve, and all poaches resolve together.

### `CONTRACTS`

```json
{ "bids": [["lot_4", 430], ["lot_9", 580], ["lot_11", 445]],
  "standing_bids": [["acct_2", 405, 10]],
  "max_lots": 3.0 }
```

`standing_bids` entries are `[account_id, rate_per_lot_day, term_days]`; the term must be one of the account's `term_options`.

## Observation

| Key | Content |
|---|---|
| `phase`, `day`, `block`, `minute`, `step`, `player` | `phase` is what this step accepts: `CAPEX`, `LABOR`, `CONTRACTS` or `DRIVING`. `minute` is `block x 120` |
| `city` | `size`, `edges` (`[u, v, class]`), `node_district`, `warehouses` (`{id: node}`). Static |
| `traffic` | `congestion` (one level per edge), `incidents` (`{edge: "e_<id>", kind, until}`), `weather`, `forecast` |
| `market` | `listings`, `accounts`, `used`, `rentals` (pool per type), `candidates` (id, name, résumé, asking wage), `fill_ceiling`, `service_interval_km` |
| `public` | per player: `cash`, `debt`, `net_worth`, `fleet`, `drivers`, `standing`, `results` |
| `history` | `auction` (last night's awards), `bids` (last night's bid book, cleared when the next night opens), `capex` and `labor` (the latest night's logs), `standing` (live accounts, as awarded) |
| `private` | this player only -- see below |

### Public to everyone

| Item | Note |
|---|---|
| Fleet roster | id, type, ownership, age, odometer, status ∈ `IDLE`/`ACTIVE`/`DISABLED`/`ORDERED` |
| Driver roster | id, name, tenure, résumé rating, notice flag, `departs` -- **never wages** |
| Cash, debt and net worth | net worth is the scoreboard |
| Day results, last 3 days | packages delivered / late / failed / refused, delivery revenue and the day's charges (holding costs and overhead excluded), net worth |
| Standing accounts held | id, pair, rate, term, days remaining |
| Tonight's board | listings, accounts, used trucks, rental pool, candidates |
| Auction, fleet and labor logs | see `history` above |

### Private to the owner

| Key | Content |
|---|---|
| `trucks` | id, type, node, status, clock, fuel, `km_since_service`, driver, staged, carried door ids, `lots` on board, pending `load`, the next 50 route entries, and `trail` (up to 24 position samples from the block just run) |
| `drivers` | id, name, **wage**, résumé, assigned truck, notice, `departs` (with the offered wage) |
| `segments`, `addresses` | today's manifest, pending doors only (at a warehouse or on a truck). Windows are `[start, end]` in minutes since 08:00 |
| `lots`, `pending_lots` | today's lots, each with `status` (`AT_DOCK`, `ON_TRUCK` with `truck`, or `DONE`); lots won for tomorrow |
| `events` | this step's events for the player's trucks: `LOADED`, `LOAD_REFUSED`, `DELIVER`, `REFUSED`, `ABANDONED`, `UNDELIVERED`, `SERVICE_DUE`, `RAN_DRY`. Freight failed or written off at a warehouse is one event per lot, with an empty `truck` |
| `sightings` | last 20 |
| `day_report` | today's running package counts; revenue and cost fill in at 18:00 |

Driver true stats, morale, restlessness and reservation wages are not in any observation.

### Sightings

At the end of each driving block, trucks with status `ACTIVE` from different players standing on the same arterial node see each other. A sighting records day, the sighted truck's clock, player, node, vehicle type, load band (`EMPTY` / `LIGHT` / `HALF` / `FULL`), and driver id. It persists as stale intel and is never refreshed.

A truck working a territory stands on that pair's anchor node.

### Grounded trucks

A grounded truck shows as `DISABLED` in the public fleet roster. `km_since_service` is private.

## Scoring and termination

```
reward = cash + fleet book value - outstanding principal - credit-line debt
```

Assigned at the final step, after day 59's 18:00 close; rewards are 0 before that.

**Owned and financed trucks are marked at book value**, not at what a buyer would pay: purchase price (the new price, or the used listing's price) less depreciation per day owned ($11 `VAN`, $18 `STEP`), floored at 25% of new. Rented trucks are not assets and add nothing. The sale haircut applies only on `SELL`.

At the 18:00 close the engine settles, in order:

1. Delivery credits and the day's charges: wages, retainers, penalties, fuel calls.
2. Per truck: own or rent cost and finance payment. Then $55 overhead per player.
3. The credit line: negative cash becomes debt, positive cash repays debt first, and debt then grows 0.25% a day.

There is no bankruptcy elimination. The episode always runs its full 60 days. Players are ranked by final net worth.

## Starting position

$12,000 cash and 3 owned trucks -- `VAN`, `VAN`, `STEP` -- at `wh_0`, `wh_1`, `wh_2`. Each has a driver of middling quality at $240/day. Starting net worth is $150,000.

The -$20,000 cash floor blocks `BUY` of either type on night 1. `FINANCE` of a `VAN` costs $7,600 down.

## Base prices

Each pair's base price is solved at init for a $220 net margin per full truck-day at index 1.0. The live reserve is the base price times the shipper's index and retry markup (see The shipper). Figures at the median near-dock deadhead:

| District | Pkg | Stops | E[cost] | E[premium] | Reserve | Payout/pkg | Net at reserve |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DOWNTOWN` | 178 | 73 | $540 | $128 | $632 | $3.55 | +$220 |
| `RIVERSIDE` | 172 | 133 | $532 | $234 | $518 | $3.01 | +$220 |
| `MIDTOWN` | 172 | 154 | $537 | $124 | $633 | $3.68 | +$220 |
| `SUBURBS_N/S` | 150 | 143 | $527 | $114 | $633 | $4.22 | +$220 |
| `INDUSTRIAL` | 52 | 45 | $413 | $4 | $629 | $12.10 | +$220 |

`E[cost]` is wages $217-232, fuel $2-4, fixed $60, overhead $55, maintenance accrual $4-8 (priced per km at the `VAN` service cost), and expected failures $58-200 (2.5% of packages at $45). `E[premium]` is the expected `PROMISED` window premium. Because the city is drawn per episode, these figures differ by episode.

## Built-in agents

| Agent | Behaviour |
|---|---|
| `idle` | returns `{}` every step |
| `random` | bids on ~30% of spot lots at 0.80-1.00x reserve; each block, loads one random waiting lot onto each empty truck and sends it up to 12 of that lot's segments in random order |
| `greedy` | bids at reserve across up to 2x its crewed trucks in pairs, within 1.00 truck-day per pair, its crewed trucks' total truck-days, and its decks. Each block, loads waiting lots onto empty, crewed trucks: territories by parcel-units, largest first, each onto a truck already given that pair, otherwise the smallest deck that holds the whole pair, then one at the warehouse, then the earliest clock, within 1.00 truck-day per truck. Routes segments bucketed by window close, nearest-neighbour within a bucket. Refuels below 300 units, services when due, and hires the best-résumé candidate at 1.10x ask when short of drivers |

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `episodeSteps` | 481 | 60 days x (3 overnight + 5 driving) + the initial state |
| `agents` | 2 or 4 | shared city, separate fleets |
| `truckPcu` | `low` | player-truck congestion weight: `low` 1, `medium` 3, `high` 8 |
| `startingCash` | 12000 | |
| `startingTrucks` | 3 | taken in order from `VAN`, `VAN`, `STEP`, repeating |
| `seed` | null | episode seed; scrubbed from the configuration and stored in `env.info['seed']` |
| `actTimeout` | 3 | seconds per turn, plus a 120 s overage bank |
| `runTimeout` | 9600 | seconds per episode |

`days`, `blockMinutes`, `shiftMinutes`, `deadlineMinutes`, `gridSize` and `warehouses` appear in `kargo.json` but the engine does not read them; it uses fixed values of 60, 120, 480, 480, 20 and 4.
