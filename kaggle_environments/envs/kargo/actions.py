"""Action sanitising: every agent action passes through here before the engine.

Each phase's action is rebuilt: entries of the wrong shape are dropped, numbers
must be finite and are clamped, ids must be strings, and every list is capped.
"""

import math

from .constants import GRID_SIZE

MAX_FLEET_OPS = 50
MAX_LABOR_OPS = 100
MAX_BIDS = 300
MAX_STANDING_BIDS = 100
MAX_TRUCK_PLANS = 100
MAX_ROUTE = 200
MAX_ABANDON = 500
MAX_ID = 64  # characters
MAX_WAGE = 2000.0
MAX_ASK = 1.0e6
MAX_WAIT = 600.0

FLEET_OPS = {"BUY", "FINANCE", "BUY_USED", "RENT", "SELL", "RETURN_RENTAL", "SERVICE", "BREAK"}


def _id(v):
    return v if isinstance(v, str) and 0 < len(v) <= MAX_ID else None


def _num(v, lo, hi):
    """A finite number clamped to [lo, hi], or None. Numeric strings are accepted."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return None
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return max(lo, min(hi, float(v)))


def _int(v, lo, hi):
    n = _num(v, lo, hi)
    return None if n is None or n != int(n) else int(n)


def _entries(v, cap):
    """List entries that are themselves non-empty lists, capped."""
    if not isinstance(v, list):
        return []
    return [e for e in v[:cap] if isinstance(e, (list, tuple)) and e]


def sanitize(phase, act):
    if not isinstance(act, dict):
        return {}
    return {"CAPEX": _capex, "LABOR": _labor, "CONTRACTS": _contracts, "DRIVING": _driving}.get(phase, lambda a: {})(
        act
    )


def _capex(act):
    fleet = []
    for e in _entries(act.get("fleet"), MAX_FLEET_OPS):
        op, arg = e[0], _id(e[1]) if len(e) > 1 else None
        if op in FLEET_OPS and arg is not None:
            fleet.append([op, arg])
    fuel = [
        ["BULK_REFUEL", _id(e[1])]
        for e in _entries(act.get("fuel"), MAX_FLEET_OPS)
        if e[0] == "BULK_REFUEL" and len(e) > 1 and _id(e[1])
    ]
    stage = {}
    if isinstance(act.get("stage"), dict):
        for k, v in list(act["stage"].items())[:MAX_TRUCK_PLANS]:
            if _id(k) and _id(v):
                stage[k] = v
    return {"fleet": fleet, "fuel": fuel, "stage": stage}


def _labor(act):
    out = []
    for e in _entries(act.get("labor"), MAX_LABOR_OPS):
        op = e[0]
        if op in ("HIRE", "WAGE") and len(e) >= 3:
            did, wage = _id(e[1]), _num(e[2], 0.0, MAX_WAGE)
            if did and wage is not None:
                out.append([op, did, wage])
        elif op == "FIRE" and len(e) >= 2 and _id(e[1]):
            out.append([op, e[1]])
        elif op == "ASSIGN" and len(e) >= 3 and _id(e[1]) and _id(e[2]):
            out.append([op, e[1], e[2]])
        elif op == "POACH" and len(e) >= 4:
            target, did, offer = _int(e[1], 0, 63), _id(e[2]), _num(e[3], 0.0, MAX_WAGE)
            if target is not None and did and offer is not None:
                out.append([op, target, did, offer])
    return {"labor": out}


def _contracts(act):
    bids = []
    for e in _entries(act.get("bids"), MAX_BIDS):
        if len(e) >= 2 and _id(e[0]):
            ask = _num(e[1], -1.0, MAX_ASK)
            if ask is not None and ask >= 0:  # a negative ask is not a price
                bids.append([e[0], ask])
    standing = []
    for e in _entries(act.get("standing_bids"), MAX_STANDING_BIDS):
        if len(e) >= 3 and _id(e[0]):
            rate, term = _num(e[1], -1.0, MAX_ASK), _int(e[2], 1, 365)
            if rate is not None and rate >= 0 and term is not None:
                standing.append([e[0], rate, term])
    out = {"bids": bids, "standing_bids": standing}
    cap = _num(act.get("max_lots"), 0.0, 1.0e6)
    if cap is not None:
        out["max_lots"] = cap
    return out


def _route_entry(v):
    if isinstance(v, str):
        return v if _id(v) else None
    if not isinstance(v, dict):
        return None
    if "via" in v:
        node = _int(v["via"], 0, GRID_SIZE * GRID_SIZE - 1)
        return {"via": node} if node is not None else None
    seg = _id(v.get("seg"))
    if seg is None:
        return None
    only = v.get("only")
    return {"seg": seg, "only": only} if only in ("windowed", "free") else {"seg": seg}


def _driving(act):
    trucks = {}
    if isinstance(act.get("trucks"), dict):
        for tid, plan in list(act["trucks"].items())[:MAX_TRUCK_PLANS]:
            if not _id(tid) or not isinstance(plan, dict):
                continue
            clean = {}
            if isinstance(plan.get("route"), list):
                clean["route"] = [r for r in (_route_entry(x) for x in plan["route"][:MAX_ROUTE]) if r is not None]
            if plan.get("on_missed_window") in ("SKIP", "ATTEMPT"):
                clean["on_missed_window"] = plan["on_missed_window"]
            wait = _num(plan.get("wait_cap"), 0.0, MAX_WAIT)
            if wait is not None:
                clean["wait_cap"] = wait
            if isinstance(plan.get("then"), str):
                clean["then"] = plan["then"][:16]
            if plan.get("hold") is True:
                clean["hold"] = True
            trucks[tid] = clean
    abandon = []
    if isinstance(act.get("abandon"), list):
        abandon = [x for x in act["abandon"][:MAX_ABANDON] if _id(x)]
    return {"trucks": trucks, "abandon": abandon}
