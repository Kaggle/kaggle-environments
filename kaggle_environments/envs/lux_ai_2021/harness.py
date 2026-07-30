"""LLM harness for the Lux AI 2021 environment.

Implements the three module-level functions of the ``GameHarness`` protocol:
``get_legal_moves``, ``generate_prompt``, ``parse_response``. The action space
is free-form (per-unit command strings are combinatorial), so
``get_legal_moves`` always returns ``None`` and ``freeForm: true`` in the env
spec routes the parser's ``submission`` list through as ``state[team].action``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from kaggle_environments.core_harness import (
    ParseResult,
    extract_last_json_object,
    get_telemetry_logger,
    render_rethink_suffix,
)
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game import Game
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_map import GameMap
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux.game_objects import Player

_TELEMETRY = get_telemetry_logger(__name__)

# Fraction of malformed commands above which we reject the whole turn and
# ask the model to rethink. At/below this fraction, valid commands are
# submitted and the invalid ones are silently dropped (the Node engine
# ignores unrecognized commands anyway).
_INVALID_COMMAND_THRESHOLD = 0.5

# Lux command grammar. Each pattern anchors the full string; ``re.fullmatch``
# is used against the head-keyed matcher below.
_DIR = r"[nsewc]"
_INT = r"\d+"
_UNIT_ID = r"u_\d+"
_RESOURCE = r"(?:wood|coal|uranium)"

# The engine's wire protocol uses "uranium" as one of the resource types. To
# avoid tripping LLM safety filters that flag nuclear-material references, the
# prompt talks about "crystal" instead; the harness rewrites crystal→uranium
# in the outgoing action list and uranium→crystal when rendering the incoming
# observation summary. Nothing else changes -- the on-wire keyword is still
# "uranium" as far as the Node engine and the game_constants file are
# concerned.
_DISPLAY_URANIUM = "crystal"

_COMMAND_VALIDATORS: dict[str, re.Pattern[str]] = {
    "m": re.compile(rf"m {_UNIT_ID} {_DIR}"),  # move
    "bcity": re.compile(rf"bcity {_UNIT_ID}"),  # build city
    "p": re.compile(rf"p {_UNIT_ID}"),  # pillage
    "t": re.compile(rf"t {_UNIT_ID} {_UNIT_ID} {_RESOURCE} {_INT}"),  # transfer
    "r": re.compile(rf"r {_INT} {_INT}"),  # research
    "bw": re.compile(rf"bw {_INT} {_INT}"),  # build worker
    "bc": re.compile(rf"bc {_INT} {_INT}"),  # build cart
}


# --- Prompt templates -------------------------------------------------------


LUX_PROMPT_TEMPLATE = """Let's play Lux AI 2021.

Rules: {width}x{width} square map. Two players simultaneously issue actions
for their workers, carts, and city tiles. The game runs 360 turns split into
day/night cycles (30 day turns then 10 night turns).

Resources and fuel. Workers can mine wood (always), coal (requires >= 50
research points), and crystal (requires >= 200 research points). Each `r`
command issued from a city tile adds 1 research point (subject to the
city tile's cooldown), and unlocks tiers apply immediately once the
thresholds are crossed. Once burned by a city, each unit of wood yields
1 fuel, coal 10 fuel, crystal 40 fuel -- so fuel-per-city and
resource-units-carried-by-a-worker are DIFFERENT quantities. Only WOOD
regenerates (~1.025x per turn, capped at 500 per tile); coal and crystal
tiles do not replenish once mined.

Building units and cities.
  - `bw` / `bc` at a city tile SPAWN a new worker / cart on that tile.
    Spawning is FREE (no resources consumed), but the total number of
    friendly units may not exceed the total number of friendly city
    tiles -- spawn commands over that cap are rejected by the engine.
  - `bcity` from a worker creates a new city tile on the worker's
    current cell. It consumes 100 resource units total from the worker's
    cargo, drawn in the order wood -> coal -> crystal until 100 units are
    spent (any mix works). The cell MUST be empty (no resource on it,
    see the command list below).

Carts have larger cargo and can transfer resources to workers.

Movement and stacking.
  - Moves resolve simultaneously across both players. Direction `c` (stay)
    is always safe.
  - You may NOT move onto an opponent city tile (the engine rejects that
    command outright).
  - CITY TILES allow UNLIMITED stacking of FRIENDLY units. Any number of
    your workers/carts may occupy the same friendly city tile (and any
    friendly city tile they stand on at night shields them from the
    per-unit fuel burn).
  - Outside city tiles, at most ONE unit may occupy a cell after the
    move. If two or more of your units try to enter the same non-city
    cell, ALL of them are cancelled back to their starting cells. That
    cancellation can cascade (a bounced-back unit may in turn collide
    with someone who was trying to move into ITS old cell), so lining
    up two moves into the same empty square typically loses both units'
    turns rather than picking one.
  - You cannot enter a non-city cell that is already held by any unit
    (friend or foe) that stayed put or was cancelled -- your move gets
    cancelled too.

Night survival. At NIGHT (turns 30-39, 70-79, ...):
  - Each city tile burns fuel from its city's shared pool. A city's
    per-night-turn upkeep is `23 * num_tiles - 10 * num_adjacent_pairs`
    (i.e. base 23 per tile, minus 10 for each orthogonal (N/S/E/W) pair
    of tiles that both belong to the same city). If a city's fuel drops
    below its upkeep on any night turn, the whole city collapses. The
    player-state section below shows each city's already-computed net
    upkeep so you don't have to recompute the adjacency bonus by hand.
  - A worker OUTSIDE a city tile burns 4 of its own cargo per night turn
    (wood first, then coal, then crystal, each converted using the
    fuel-yield rates above); a cart outside a city burns 10. If a
    worker/cart can't cover its upkeep it dies. A worker or cart STANDING
    ON a friendly city tile at night pays nothing personally.

The game ends at turn 360, or earlier if a team has no units AND no city
tiles (elimination). Winner: more city tiles at end, tiebreak more units;
if both are still tied it is a draw.

Commands (one per line in your JSON response):
  m <unit_id> <direction>          -- move unit (direction: n, s, e, w, c=stay)
  bcity <unit_id>                  -- worker builds a city tile at its position
                                      (worker must be on an empty cell -- not
                                      on a wood/coal/crystal resource tile and
                                      not on an existing city tile)
  p <unit_id>                      -- worker pillages road at its position
  t <from_id> <to_id> <res> <amt>  -- transfer resources between adjacent units
                                      (res: wood, coal, or crystal). BOTH ids
                                      must be UNIT ids (u_N) -- you cannot
                                      transfer into a city; a unit fuels a city
                                      by standing on it or building it.
  r <x> <y>                        -- city tile at (x,y) researches
  bw <x> <y>                       -- city tile at (x,y) builds a worker
  bc <x> <y>                       -- city tile at (x,y) builds a cart

A unit or city tile can only act when its cooldown is < 1. Each unit and
each city tile may perform AT MOST ONE command per turn -- any second
command for the same id in the same turn is silently dropped by the
engine. Across the fleet you may issue zero or more commands per turn.
Every command you emit must exactly match one of the forms above -- a
small amount of leeway is applied (leading bullets, uppercase directions
and unit ids, extra whitespace, `[u_1]`/`<u_2>` brackets, trailing `#`
or `//` comments) but arbitrary syntax is NOT accepted. If more than
half of your commands are malformed the whole turn is rejected and you
are asked to try again; otherwise the malformed ones are dropped and
the valid ones are submitted.

Coordinates are (x, y) with x=0 on the left and y=0 at the top.

You are player {player_id}. Turn {turn} of 360 ({phase}).

Map (uppercase = your units/cities, lowercase = opponent's). Every cell is
TWO characters: <terrain><occupant>, so a unit never hides the tile it stands on.
  terrain (1st char): `.` empty, `w` wood, `k` coal, `x` crystal,
          `T` your city tile, `t` opponent city tile
  occupant (2nd char): ` ` none, `U`/`A` your worker/cart, `u`/`a` opponent
          worker/cart, or a DIGIT = that many units stacked on the cell (`+`
          for 10+). Only a city tile can hold a stack, and units never enter
          an enemy city, so the terrain char says whose the stack is: `T`+digit
          is yours, `t`+digit is the opponent's. E.g. `wU` = your worker on
          wood (you CANNOT `bcity` there); `T3` = your city tile sheltering 3
          of your units at night; `t2` = opponent city tile with 2 of theirs.
{ascii_map}

Your state:
{your_summary}

Opponent state:
{opponent_summary}

{recent_moves_block}
First give a short justification of your plan for this turn (a few
sentences -- which units and city tiles you are acting and why), then
conclude with your commands as a JSON object of the form:

```json
{{"actions": ["m u_1 n", "bcity u_2", "r 5 7"]}}
```

Put the JSON object last; it is read as your final answer. An empty list
(`{{"actions": []}}`) is legal and means "do nothing this turn".
"""


_THRESHOLD_PCT = int(_INVALID_COMMAND_THRESHOLD * 100)

RETHINK_ILLEGAL = (
    "\n\nMore than "
    + str(_THRESHOLD_PCT)
    + "% of your previous commands were malformed:\n"
    + "{previous_action}\n\n"
    + "Re-read the command grammar above and reply again with a JSON object "
    + 'of the form `{{"actions": [...]}}`. Every command must match one of '
    + "the listed forms exactly.\n"
)

RETHINK_UNPARSABLE = (
    "\n\nYour previous response ended with:\n"
    + "{previous_response}\n\n"
    + "No JSON `actions` object could be parsed from that. Conclude your "
    + "response with a JSON object in a fenced code block, exactly like:\n\n"
    + "```json\n"
    + '{{"actions": ["m u_1 n"]}}\n'
    + "```\n"
)


# --- Helpers ----------------------------------------------------------------


def _rebuild_game(observation: Mapping[str, Any]) -> Game:
    """Reconstruct a fully-populated ``Game`` from a single turn's observation.

    ``lux.game.Game._update`` fully rebuilds the map and both players from
    the update messages, so one call per turn is enough as long as we seed
    the width/height/players ourselves. At step 0 the engine's frame carries
    a two-line header (game id + "W H") that ``_initialize`` consumes;
    ``_update`` doesn't handle it, so we strip it there. From step 1 on
    every frame is turn-shaped.
    """
    width = int(observation["width"])
    height = int(observation.get("height", width))
    updates = list(observation.get("updates") or [])
    step = int(observation.get("step", 0))

    game = Game()
    game.id = 0
    game.map_width = width
    game.map_height = height
    game.map = GameMap(width, height)
    game.players = [Player(0), Player(1)]
    # ``_update`` increments ``turn`` before consuming messages, so the value
    # after N cumulative ``_update`` calls is N-1. Step 0 has the header
    # (which ``_update`` treats as no-ops) but no turn has been applied yet;
    # step 1 is the first real turn.
    game.turn = step - 2

    if step == 0 and len(updates) >= 2:
        # Step-0 header: drop game id + "W H" lines before ``_update``.
        updates = updates[2:]
    game._update(updates)
    return game


def _phase(turn: int) -> str:
    """Return ``"day"`` or ``"night"`` for the current turn."""
    return "night" if (turn % 40) >= 30 else "day"


def _render_ascii_map(game: Game, player_id: int) -> str:
    """Render the game map as an ASCII grid, uppercase = ``player_id``'s side.

    Each cell is rendered as TWO characters so that terrain and occupant are
    BOTH visible even when a unit stands on a resource or several units share a
    city tile (a single-glyph grid would hide one behind the other):

    - Column 1 (terrain): ``.`` empty, ``w`` wood, ``k`` coal, ``x`` crystal,
      ``T`` your city tile, ``t`` opponent city tile.
    - Column 2 (occupant): `` `` (space) none, ``U``/``A`` your worker/cart,
      ``u``/``a`` opponent worker/cart, or a DIGIT giving the number of units
      stacked on the cell (``+`` for 10 or more). Stacking is only possible on
      a city tile, and a unit can never enter an enemy city tile, so the cell
      only ever holds the city owner's units -- the ``T``/``t`` in column 1
      says whose they are (``T`` + digit = your stack, ``t`` + digit =
      opponent stack).

    So ``wU`` is your worker standing on wood (``bcity`` there is rejected by
    the engine), ``T3`` is your city tile sheltering three of your units at
    night, and ``t2`` is an opponent city tile sheltering two of theirs.
    Coordinate axes are printed above and to the left so the model can read
    positions off the grid.
    """
    width, height = game.map.width, game.map.height

    terrain = [["."] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            cell = game.map.get_cell(x, y)
            if cell.has_resource():
                r = cell.resource.type
                # Glyphs chosen to not collide with unit/city letters.
                # "coal" → "k" (c is stay-in-place direction), "uranium" → "x"
                # (also avoids the "u" that opponent workers use).
                terrain[y][x] = "w" if r == "wood" else ("k" if r == "coal" else "x")

    occupant = [[" "] * width for _ in range(height)]
    # Count units per cell per team. A cell only ever holds one team's units
    # (two-team co-occupation of a non-city cell is cancelled by the engine,
    # and a unit can't enter an enemy city tile), so the per-team counts never
    # contend for the same cell -- whichever team occupies it, the terrain
    # char (T vs t) identifies the owner.
    unit_count: dict[tuple[int, int, int], int] = {}
    unit_kind: dict[tuple[int, int, int], str] = {}
    for team, player in enumerate(game.players):
        for unit in player.units:
            x, y = unit.pos.x, unit.pos.y
            if not (0 <= x < width and 0 <= y < height):
                continue
            key = (team, x, y)
            unit_count[key] = unit_count.get(key, 0) + 1
            unit_kind[key] = ("A" if unit.is_cart() else "U") if team == player_id else ("a" if unit.is_cart() else "u")

    for (team, x, y), n in unit_count.items():
        if n == 1:
            occupant[y][x] = unit_kind[(team, x, y)]
        else:
            occupant[y][x] = str(n) if n <= 9 else "+"

    # City tiles are terrain (they persist and can be built upon / sheltered
    # in); write them after resources so a city tile always shows as T/t.
    for team, player in enumerate(game.players):
        for city in player.cities.values():
            for tile in city.citytiles:
                x, y = tile.pos.x, tile.pos.y
                if 0 <= x < width and 0 <= y < height:
                    terrain[y][x] = "T" if team == player_id else "t"

    # Header: two-digit column indices left-justified into the 2-char cells.
    tens = "   " + "".join(f"{x // 10 if x >= 10 else ' ':<2}" for x in range(width))
    ones = "   " + "".join(f"{x % 10:<2}" for x in range(width))
    body = [f"{y:2d} " + "".join(terrain[y][x] + occupant[y][x] for x in range(width)) for y in range(height)]
    return "\n".join([tens, ones, *body])


def _render_player(game: Game, player_id: int) -> str:
    """Render a structured per-player summary (research, units, cities)."""
    player = game.players[player_id]
    rp = player.research_points
    # Distance to the next unlock is what the model needs to plan research
    # investment; surface it explicitly.
    if not player.researched_coal():
        next_unlock = f"coal in {max(0, 50 - rp)} more points"
    elif not player.researched_uranium():
        next_unlock = f"{_DISPLAY_URANIUM} in {max(0, 200 - rp)} more points"
    else:
        next_unlock = "all resources unlocked"
    lines = [
        f"  research points: {rp} (coal @ 50, {_DISPLAY_URANIUM} @ 200; next: {next_unlock})",
        f"    coal researched: {player.researched_coal()}, "
        f"{_DISPLAY_URANIUM} researched: {player.researched_uranium()}",
    ]

    if player.units:
        lines.append(f"  units ({len(player.units)}):")
        for u in player.units:
            kind = "worker" if u.is_worker() else "cart"
            cargo = f"wood={u.cargo.wood} coal={u.cargo.coal} {_DISPLAY_URANIUM}={u.cargo.uranium}"
            lines.append(f"    {u.id} {kind} at ({u.pos.x},{u.pos.y}) cooldown={u.cooldown:g} cargo=[{cargo}]")
            # A worker on a resource tile cannot build a city there -- the
            # engine rejects ``bcity`` on any non-empty resource cell. The
            # ASCII map's terrain column shows the resource, but call it out
            # explicitly here since this is where the model decides to build.
            if u.is_worker():
                cell = game.map.get_cell(u.pos.x, u.pos.y)
                if cell.has_resource():
                    res = cell.resource.type
                    display = _DISPLAY_URANIUM if res == "uranium" else res
                    lines.append(
                        f"      -- standing on {display}; cannot build a city here (move to an empty cell first)"
                    )
    else:
        lines.append("  units: (none)")

    if player.cities:
        total_tiles = sum(len(c.citytiles) for c in player.cities.values())
        # Aggregate net upkeep across all cities so the model can see the
        # total per-night fuel drain without having to sum by hand.
        total_upkeep = sum(c.light_upkeep for c in player.cities.values())
        lines.append(f"  cities ({len(player.cities)}, {total_tiles} tiles, total night upkeep {total_upkeep:g}/turn):")
        for city in player.cities.values():
            # Per-tile cooldown so the model can see which tiles can act this
            # turn (a city tile can issue a command only when cooldown < 1).
            tiles = ", ".join(f"({t.pos.x},{t.pos.y} cd={t.cooldown:g})" for t in city.citytiles)
            lines.append(f"    {city.cityid} fuel={city.fuel:g} upkeep={city.light_upkeep:g} tiles=[{tiles}]")
    else:
        lines.append("  cities: (none)")

    return "\n".join(lines)


def _render_recent_moves(move_history: Sequence[str]) -> str:
    """Render the previous turn's action summary if we have one."""
    if not move_history:
        return ""
    return f"Your previous turn's commands: {move_history[-1]}\n\n"


# Full-word direction aliases the parser accepts and folds to the single-letter
# form the engine expects.
_DIRECTION_ALIASES = {
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "center": "c",
}

_LEADING_JUNK_RE = re.compile(r"^[\s\-*•]+|^\d+[.)]\s*")
_TRAILING_JUNK_RE = re.compile(r"[\s.,;:!?]+$")
_TRAILING_COMMENT_RE = re.compile(r"\s*(?:#|//).*$")
# Unit/city id token, tolerating a MISSING underscore. The engine's wire form
# is ``u_N`` / ``c_N``, but models routinely drop the underscore (``u9``) or
# uppercase the letter (``U_9``) to match the ASCII map glyphs. The optional
# ``_`` lets ``_normalize_command`` fold ``u9``/``U9``/``U_9`` all to ``u_9``.
# The letter class is only ``u``/``c`` so a bare direction (``c`` = stay, no
# digits) and coordinate integers (``5``) are never mistaken for ids.
_ID_TOKEN_RE = re.compile(r"([uUcC])_?(\d+)")

# Which argument slots of each command actually hold a unit/city id (and so are
# safe to canonicalise). ``r``/``bw``/``bc`` take *coordinate integers*, and a
# ``c``-prefixed coordinate token (``c5``) would otherwise be mis-rewritten to
# ``c_5``, so those commands appear here with no id slots at all.
_ID_ARG_SLOTS: dict[str, tuple[int, ...]] = {
    "m": (1,),  # m <unit_id> <dir>
    "bcity": (1,),  # bcity <unit_id>
    "p": (1,),  # p <unit_id>
    "t": (1, 2),  # t <from_id> <to_id> <res> <amt>
}


def _canonical_id(token: str) -> str:
    """Canonicalise a unit/city id token, or return it unchanged.

    ``u_9``/``U_9``/``u9``/``U9`` (and the ``c`` city variants) all fold to
    ``u_9`` / ``c_9``. Only a token that is *entirely* an id is rewritten, so
    coordinate integers and the ``c`` (stay) direction pass through untouched.
    Callers additionally restrict this to id-argument slots (see
    ``_ID_ARG_SLOTS``) so ``c``-prefixed coordinates in ``r``/``bw``/``bc``
    are never touched.
    """
    m = _ID_TOKEN_RE.fullmatch(token)
    if m is None:
        return token
    return f"{m.group(1).lower()}_{m.group(2)}"


def _normalize_command(cmd: str) -> str:
    """Fold common LLM variants into the engine's canonical command form.

    - Strips surrounding whitespace and list-marker leaders (``-``, ``*``,
      ``1.``, bullets) and trailing punctuation (``.``, ``,``, ``;``).
    - Strips trailing line-comments (``# ...`` and ``// ...``).
    - Strips ``[]``, ``()``, and ``<>`` wrappers around any token.
    - Collapses runs of internal whitespace to a single space.
    - Lowercases command head, canonicalises unit/city id tokens (folding
      case AND inserting a missing underscore, so ``U_1``/``u1``/``U1`` all
      become ``u_1``), and folds full direction names for the ``m`` command.
    - Rewrites the display resource name (``crystal``) back to the engine's
      wire keyword (``uranium``) in ``t`` transfer commands.
    """
    if not cmd:
        return ""
    cmd = _TRAILING_COMMENT_RE.sub("", cmd)
    cmd = _LEADING_JUNK_RE.sub("", cmd)
    cmd = _TRAILING_JUNK_RE.sub("", cmd)
    # Strip bracket/angle/paren wrappers around any token
    # (e.g. ``[u_1]`` -> ``u_1``, ``<u_2>`` -> ``u_2``, ``(u_3)`` -> ``u_3``).
    cmd = cmd.translate(str.maketrans("", "", "[]<>()"))
    cmd = " ".join(cmd.split())  # collapse internal whitespace
    if not cmd:
        return ""
    parts = cmd.split(" ")
    # All command heads (``m``, ``bcity``, ``p``, ``t``, ``r``, ``bw``, ``bc``)
    # are lowercase in the engine's grammar.
    parts[0] = parts[0].lower()
    # Unit and city ids are exclusively lowercase ``u_N`` / ``c_N`` in the
    # engine's wire protocol. Models routinely uppercase them to match the
    # ASCII map letters (``U``/``T``) and/or drop the underscore (``u9``), so
    # canonicalise the id-shaped tokens: lowercase the letter and insert the
    # underscore if it's missing. Only the slots that actually hold ids are
    # touched, so a ``c``-prefixed coordinate (``r c5 c7``) is left intact.
    for i in _ID_ARG_SLOTS.get(parts[0], ()):
        if i < len(parts):
            parts[i] = _canonical_id(parts[i])
    if parts[0] == "m" and len(parts) == 3:
        # direction lowercased and folded from full-word aliases.
        d = parts[2].lower()
        parts[2] = _DIRECTION_ALIASES.get(d, d)
    elif parts[0] == "t" and len(parts) == 5:
        # Resource keyword is always lowercase in the engine grammar.
        # Fold case, then map the display name (``crystal``) to the wire
        # keyword (``uranium``).
        parts[3] = parts[3].lower()
        if parts[3] == _DISPLAY_URANIUM:
            parts[3] = "uranium"
    return " ".join(parts)


def _validate_command(cmd: str) -> bool:
    """True iff ``cmd`` matches one of the Lux command patterns exactly."""
    if not cmd:
        return False
    head = cmd.split(" ", 1)[0]
    matcher = _COMMAND_VALIDATORS.get(head)
    return matcher is not None and matcher.fullmatch(cmd) is not None


# --- Public functions (GameHarness protocol) --------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str] | None:
    """Free-form action space. ``freeForm: true`` in the env spec is required."""
    del observation
    return None


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current Lux turn."""
    if not observation.get("updates"):
        return "Lux AI has not started this turn yet. Waiting for the game to initialize."

    player_id = int(observation.get("player", 0))
    game = _rebuild_game(observation)
    # On step 0 (before any turn has been played), ``game.turn`` is -1;
    # clamp to 0 so the prompt reads "Turn 0 (day)" rather than the
    # nonsensical "Turn -1 (night)".
    turn = max(game.turn, 0)
    width = game.map.width

    ascii_map = _render_ascii_map(game, player_id)
    your_summary = _render_player(game, player_id)
    opponent_summary = _render_player(game, 1 - player_id)
    recent_moves_block = _render_recent_moves(move_history)

    prompt = LUX_PROMPT_TEMPLATE.format(
        width=width,
        player_id=player_id,
        turn=turn,
        phase=_phase(turn),
        ascii_map=ascii_map,
        your_summary=your_summary,
        opponent_summary=opponent_summary,
        recent_moves_block=recent_moves_block,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL,
        RETHINK_UNPARSABLE,
        previous_response,
        previous_action,
    )

    return prompt


def parse_response(
    response: str,
    legal_action_strings: Sequence[str] | None,
    *,
    observation: Mapping[str, Any] | None = None,
) -> ParseResult:
    """Extract a JSON ``{"actions": [...]}`` list of Lux command strings.

    Malformed commands are dropped and valid ones submitted, unless more
    than ``_INVALID_COMMAND_THRESHOLD`` of the list is malformed -- in that
    case the whole turn is rejected and ``core_harness`` triggers a rethink.
    An empty ``actions`` list is legal ("do nothing this turn").
    """
    del legal_action_strings, observation  # free-form; not used

    obj = extract_last_json_object(response, required_keys=("actions",))
    if obj is None:
        return ParseResult(raw_action=None)
    actions_raw = obj.get("actions")
    if not isinstance(actions_raw, list):
        # ``actions`` is present but not a list (string, null, dict). The
        # failure is structural, not command-level, so route to the
        # unparseable rethink ("send an array") rather than the malformed
        # rethink ("fix your commands"). raw_action=None triggers that path.
        _TELEMETRY(actions_not_a_list={"got": str(actions_raw)[:200]})
        return ParseResult(raw_action=None)

    actions = [_normalize_command(str(a)) for a in actions_raw]
    valid = [a for a in actions if _validate_command(a)]
    invalid = [a for a in actions if not _validate_command(a)]
    summary = "[" + ", ".join(actions) + "]"

    if actions and (len(invalid) / len(actions)) > _INVALID_COMMAND_THRESHOLD:
        _TELEMETRY(
            too_many_invalid_commands={
                "valid": len(valid),
                "invalid": len(invalid),
                "invalid_commands": invalid,
            },
        )
        return ParseResult(raw_action=summary)

    if invalid:
        _TELEMETRY(
            some_invalid_commands={
                "valid": len(valid),
                "invalid": len(invalid),
                "invalid_commands": invalid,
            },
        )
    return ParseResult(submission=valid, raw_action=summary)
