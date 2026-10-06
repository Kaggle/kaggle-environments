"""The two-level city: arterial travel grid plus interior delivery streets.

The arterial grid is 20x20 intersections at ~1.5 km spacing and carries
congestion and routing. The interior street grid is generated lazily, per lot.
"""

import heapq
import math
import random

from .constants import (
    ACCIDENT_CAPACITY,
    ACCIDENT_DURATION,
    ACCIDENT_HAZARD_PER_EDGE_MIN,
    ACCIDENT_REPORT_DELAY,
    AR1_RHO,
    AR1_SIGMA,
    ARTERIAL_EVERY,
    BPR_ALPHA,
    BPR_BETA,
    CLOSURE_DAILY_P,
    CLOSURE_DURATION,
    COLLECTOR_SHARE,
    CONGESTION_CUTS,
    CONGESTION_LEVELS,
    CONSTRUCTION_PER_DAY,
    DAY_END_MINUTES,
    DISTRICT_NAMES,
    DISTRICTS,
    DOW_MULT,
    EDGE_KM,
    GRID_DETOUR,
    GRID_SIZE,
    LDAY_SIGMA,
    LOCAL_SPEED_KMH,
    ROAD_CLASSES,
    TOD_BASE,
    TOD_BUMPS,
    TRUCK_PCU_LEVELS,
    WAREHOUSES,
    WEATHER,
    WEATHER_FORECAST_ACCURACY,
)

CITY_TICK = 6  # minutes between congestion updates; AR(1) rho is per tick


def time_of_day_shape(minutes):
    """`P(m)`, m in minutes since 08:00."""
    total = TOD_BASE
    for mu, sigma, amp in TOD_BUMPS:
        total += amp * math.exp(-((minutes - mu) ** 2) / (2.0 * sigma * sigma))
    return total


def congestion_level(vc):
    """Ordinal bucket for a `v/c`."""
    for i, cut in enumerate(CONGESTION_CUTS):
        if vc < cut:
            return CONGESTION_LEVELS[i]
    return CONGESTION_LEVELS[-1]


def bpr_multiplier(vc):
    return 1.0 + BPR_ALPHA * (vc**BPR_BETA)


class City:
    """Static geometry plus the live congestion field."""

    def __init__(self, rng, truck_pcu="low"):
        self.rng = rng
        self.size = GRID_SIZE
        self.pcu = TRUCK_PCU_LEVELS.get(truck_pcu, TRUCK_PCU_LEVELS["low"])
        self.weather_plan = {}  # day -> weather
        self.forecasts = {}  # day -> the published (noisy) forecast for it
        self.occupancy = {}
        self._build_grid()
        self._draw_districts()
        self._place_warehouses()
        self.reset_day(0, rng)

    # --- static geometry ---------------------------------------------------

    def _build_grid(self):
        n = self.size
        self.num_nodes = n * n
        self.edges = []  # (u, v, klass, t_free, capacity)
        self.adjacency = [[] for _ in range(self.num_nodes)]
        self.edge_index = {}

        def add(u, v):
            ru, cu = divmod(u, n)
            rv, cv = divmod(v, n)
            arterial = (ru == rv and ru % ARTERIAL_EVERY == 0) or (cu == cv and cu % ARTERIAL_EVERY == 0)
            if arterial:
                klass = "ARTERIAL"
            elif self.rng.random() < COLLECTOR_SHARE:
                klass = "COLLECTOR"
            else:
                klass = "LOCAL"
            spec = ROAD_CLASSES[klass]
            idx = len(self.edges)
            self.edges.append((u, v, klass, spec["t_free"], spec["capacity"]))
            self.adjacency[u].append((v, idx))
            self.adjacency[v].append((u, idx))
            self.edge_index[(u, v)] = idx
            self.edge_index[(v, u)] = idx

        for r in range(n):
            for c in range(n):
                u = r * n + c
                if c + 1 < n:
                    add(u, u + 1)
                if r + 1 < n:
                    add(u, u + n)

    def _draw_districts(self):
        """Positions and patch sizes are drawn; archetypes are fixed."""
        n = self.size
        rng = self.rng
        # (row lo, row hi, col lo, col hi) bands for each district seed.
        bands = {
            "DOWNTOWN": (0.30, 0.70, 0.30, 0.70),
            "RIVERSIDE": (0.15, 0.85, 0.15, 0.85),
            "MIDTOWN": (0.15, 0.85, 0.15, 0.85),
            "SUBURBS_N": (0.00, 0.30, 0.05, 0.95),
            "SUBURBS_S": (0.70, 1.00, 0.05, 0.95),
            "INDUSTRIAL": (0.05, 0.95, 0.00, 0.25),
        }
        seeds = {}
        for name in DISTRICT_NAMES:
            r0, r1, c0, c1 = bands[name]
            seeds[name] = (
                rng.uniform(r0, r1) * (n - 1),
                rng.uniform(c0, c1) * (n - 1),
                DISTRICTS[name]["weight"] * rng.uniform(0.75, 1.25),
            )
        self.district_seeds = seeds

        self.node_district = [None] * self.num_nodes
        for r in range(n):
            for c in range(n):
                best, best_d = None, None
                for name, (sr, sc, w) in seeds.items():
                    d = math.hypot(r - sr, c - sc) / w
                    if best_d is None or d < best_d:
                        best, best_d = name, d
                self.node_district[r * n + c] = best

        self.edge_district = [self.node_district[u] for (u, _v, _k, _t, _c) in self.edges]
        self.district_nodes = {name: [] for name in DISTRICT_NAMES}
        for i, name in enumerate(self.node_district):
            self.district_nodes[name].append(i)

    def _place_warehouses(self):
        """Four communal warehouses, at least `min_sep` cells apart when possible."""
        n = self.size
        rng = self.rng
        candidates = list(range(self.num_nodes))
        rng.shuffle(candidates)
        chosen = []
        min_sep = n * 0.45
        for node in candidates:
            r, c = divmod(node, n)
            if all(math.hypot(r - divmod(p, n)[0], c - divmod(p, n)[1]) >= min_sep for p in chosen):
                chosen.append(node)
            if len(chosen) == WAREHOUSES:
                break
        while len(chosen) < WAREHOUSES:  # degenerate draw; fill anywhere unused
            node = candidates.pop()
            if node not in chosen:
                chosen.append(node)
        self.warehouse_nodes = chosen
        self.warehouse_ids = [f"wh_{i}" for i in range(len(chosen))]
        self.warehouse_of = dict(zip(self.warehouse_ids, chosen))

    # --- live state --------------------------------------------------------

    def reset_day(self, day, rng):
        """Draw the day's level, weather, construction, and clear yesterday."""
        self.day = day
        self.dow = day % 7
        self.l_day = math.exp(LDAY_SIGMA * rng.gauss(0, 1) - LDAY_SIGMA**2 / 2)

        self.weather = self._weather_for(day, rng)
        self.occupancy = {}  # edge -> [(enter, leave)] of player trucks today

        self.closures = {}  # edge -> (start, end, reason)
        low, high = CONSTRUCTION_PER_DAY
        for _ in range(rng.randint(low, high)):
            self.closures[rng.randrange(len(self.edges))] = (0, 10**6, "CONSTRUCTION")
        self.incidents = {}  # edge -> [{capacity, start, end, reported}]
        self.incident_log = []
        self._draw_day(random.Random(rng.getrandbits(64)))

    def _draw_day(self, rng):
        """The whole day's noise path and incidents, per tick, from 08:00 to 18:00."""
        scale = math.sqrt(1.0 - AR1_RHO * AR1_RHO)
        noise = [0.0] * len(self.edges)
        self.noise_path = [noise]
        ticks = int(DAY_END_MINUTES // CITY_TICK)
        for tick in range(1, ticks + 1):
            gauss = rng.gauss
            noise = [AR1_RHO * x + scale * gauss(0, 1) for x in noise]
            self.noise_path.append(noise)
        for tick in range(ticks):
            self._roll_incidents(tick * CITY_TICK, rng)

    def _weather_for(self, day, rng):
        if day not in self.weather_plan:
            roll, acc = rng.random(), 0.0
            self.weather_plan[day] = "SNOW"
            for name, spec in WEATHER.items():
                acc += spec["p"]
                if roll < acc:
                    self.weather_plan[day] = name
                    break
        return self.weather_plan[day]

    def forecast_for(self, day, rng):
        """`day`'s weather, right `WEATHER_FORECAST_ACCURACY` of the time. Fixed once drawn."""
        if day not in self.forecasts:
            truth = self._weather_for(day, rng)
            names = list(WEATHER)
            self.forecasts[day] = (
                truth if rng.random() < WEATHER_FORECAST_ACCURACY else names[rng.randrange(len(names))]
            )
        return self.forecasts[day]

    def occupy(self, edge, enter, leave):
        self.occupancy.setdefault(edge, []).append((enter, leave))

    def trucks_on(self, edge, minute):
        """Player trucks on `edge` at `minute`, from today's recorded traversals."""
        return sum(1 for a, b in self.occupancy.get(edge, ()) if a <= minute < b)

    def _closure(self, edge, minute, known):
        """The closure in force on `edge` at `minute`, if it had begun by `known`."""
        closure = self.closures.get(edge)
        if closure and closure[0] <= known and closure[0] <= minute < closure[1]:
            return closure
        return None

    def _accident(self, edge, minute, known):
        for inc in self.incidents.get(edge, ()):
            if inc["start"] <= known and inc["start"] <= minute < inc["end"]:
                return inc
        return None

    def _roll_incidents(self, minute, rng):
        """Accident hazard scales with the edge's own background congestion."""
        weather_mult = WEATHER[self.weather]["accident"]
        # Sample rather than sweep all 760 edges: hazard is tiny and uniform
        # sampling with a scaled rate is distributionally the same.
        trials = 24
        rate = ACCIDENT_HAZARD_PER_EDGE_MIN * CITY_TICK * weather_mult * len(self.edges) / trials
        for _ in range(trials):
            edge = rng.randrange(len(self.edges))
            if self._accident(edge, minute, minute) or self._closure(edge, minute, minute):
                continue
            vc = self.background_vc(edge, minute)
            if rng.random() < rate * vc * vc:
                lo, hi = ACCIDENT_DURATION
                self.incidents.setdefault(edge, []).append(
                    {
                        "capacity": rng.uniform(*ACCIDENT_CAPACITY),
                        "start": minute,
                        "end": minute + rng.uniform(lo, hi),
                        "reported": minute + ACCIDENT_REPORT_DELAY,
                    }
                )
                self.incident_log.append({"edge": edge, "minute": minute, "kind": "ACCIDENT"})
        if rng.random() < CLOSURE_DAILY_P * CITY_TICK / DAY_END_MINUTES:
            edge = rng.randrange(len(self.edges))
            lo, hi = CLOSURE_DURATION
            if edge not in self.closures:
                self.closures[edge] = (minute, minute + rng.uniform(lo, hi), "EMERGENCY")
                self.incident_log.append({"edge": edge, "minute": minute, "kind": "CLOSURE"})

    # --- costs -------------------------------------------------------------

    def background_vc(self, edge, minute, known=None):
        """`v/c` from background traffic alone, with the noise as of `known`."""
        c_d = DISTRICTS[self.edge_district[edge]]["c_d"]
        shape = time_of_day_shape(minute)
        tick = min(int((minute if known is None else known) // CITY_TICK), len(self.noise_path) - 1)
        noise = 1.0 + AR1_SIGMA * self.noise_path[max(0, tick)][edge]
        return max(0.0, c_d * shape * DOW_MULT[self.dow] * self.l_day * noise)

    def edge_vc(self, edge, minute, known=None):
        """`v/c` at `minute`, from what the city looked like at `known` (default: `minute`)."""
        known = minute if known is None else known
        vc = self.background_vc(edge, minute, known)
        trucks = self.trucks_on(edge, minute)
        if trucks:
            vc += self.pcu * trucks / self.edges[edge][4]
        inc = self._accident(edge, minute, known)
        if inc:
            vc /= max(0.05, inc["capacity"])
        return vc

    def edge_time(self, edge, minute, known=None):
        """Minutes to traverse, or None if the edge is shut."""
        known = minute if known is None else known
        if self._closure(edge, minute, known):
            return None
        t_free = self.edges[edge][3]
        vc = self.edge_vc(edge, minute, known)
        return t_free * bpr_multiplier(vc) * WEATHER[self.weather]["travel"]

    def route(self, source, target, minute):
        """A* on the costs known at `minute`. Returns (minutes, [edge ids])."""
        if source == target:
            return 0.0, []
        n = self.size
        tr, tc = divmod(target, n)
        floor = min(spec["t_free"] for spec in ROAD_CLASSES.values())

        def h(node):
            r, c = divmod(node, n)
            return (abs(r - tr) + abs(c - tc)) * floor

        dist = {source: 0.0}
        prev = {}
        heap = [(h(source), source)]
        seen = set()
        while heap:
            _f, node = heapq.heappop(heap)
            if node in seen:
                continue
            seen.add(node)
            if node == target:
                break
            base = dist[node]
            for nbr, edge in self.adjacency[node]:
                if nbr in seen:
                    continue
                cost = self.edge_time(edge, minute + base, known=minute)
                if cost is None:
                    continue
                nd = base + cost
                if nd < dist.get(nbr, math.inf):
                    dist[nbr] = nd
                    prev[nbr] = (node, edge)
                    heapq.heappush(heap, (nd + h(nbr), nbr))
        if target not in dist:
            return None, []
        path = []
        node = target
        while node != source:
            node, edge = prev[node]
            path.append(edge)
        path.reverse()
        return dist[target], path

    def local_minutes(self, km):
        """Interior streets: no modelled congestion, grid detour applied."""
        return km * GRID_DETOUR / LOCAL_SPEED_KMH * 60.0

    def node_km(self, a, b):
        n = self.size
        ra, ca = divmod(a, n)
        rb, cb = divmod(b, n)
        return (abs(ra - rb) + abs(ca - cb)) * EDGE_KM

    # --- public view -------------------------------------------------------

    def congestion_report(self, minute):
        """One ordinal level per edge."""
        return [congestion_level(self.edge_vc(i, minute)) for i in range(len(self.edges))]

    def incident_report(self, minute):
        out = []
        for edge, incs in self.incidents.items():
            for inc in incs:
                if inc["reported"] <= minute < inc["end"]:
                    out.append({"edge": f"e_{edge}", "kind": "ACCIDENT", "until": round(inc["end"], 1)})
        for edge, (start, end, reason) in self.closures.items():
            if start <= minute < end:
                out.append({"edge": f"e_{edge}", "kind": reason, "until": None if end > 10**5 else round(end, 1)})
        return out

    def static_view(self):
        """The road graph, known from day 0."""
        return {
            "size": self.size,
            "edges": [[u, v, klass] for (u, v, klass, _t, _c) in self.edges],
            "node_district": self.node_district,
            "warehouses": {wid: node for wid, node in self.warehouse_of.items()},
        }
