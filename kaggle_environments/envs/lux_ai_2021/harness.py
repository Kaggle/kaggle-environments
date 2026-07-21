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
                                      (res: wood, coal, or crystal)
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

Map (uppercase = your units/cities, lowercase = opponent's):
  legend: `.` empty, `w` wood, `k` coal, `x` crystal,
          `U/u` worker, `A/a` cart, `T/t` city tile
{ascii_map}

Your state:
{your_summary}

Opponent state:
{opponent_summary}

{recent_moves_block}
Reply with a JSON object of the form:

```json
{{"actions": ["m u_1 n", "bcity u_2", "r 5 7"]}}
```

An empty list (`{{"actions": []}}`) is legal and means "do nothing this turn".
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

    Precedence per cell (last write wins visually): resources → carts →
    workers → city tiles. Coordinate axes are printed above and to the
    left so the model can read positions off the grid.
    """
    width, height = game.map.width, game.map.height
    grid = [["."] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            cell = game.map.get_cell(x, y)
            if cell.has_resource():
                r = cell.resource.type
                # Glyphs chosen to not collide with unit/city letters.
                # "coal" → "k" (c is stay-in-place direction), "uranium" → "x"
                # (also avoids the "u" that opponent workers use).
                grid[y][x] = "w" if r == "wood" else ("k" if r == "coal" else "x")

    def _place(char_you: str, char_them: str, team: int, x: int, y: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = char_you if team == player_id else char_them

    for team, player in enumerate(game.players):
        for unit in player.units:
            char_you, char_them = ("A", "a") if unit.is_cart() else ("U", "u")
            _place(char_you, char_them, team, unit.pos.x, unit.pos.y)
        for city in player.cities.values():
            for tile in city.citytiles:
                _place("T", "t", team, tile.pos.x, tile.pos.y)

    # Header: two-digit column indices, top row is tens, second row is units.
    tens = "   " + "".join(str(x // 10) if x >= 10 else " " for x in range(width))
    ones = "   " + "".join(str(x % 10) for x in range(width))
    body = [f"{y:2d} " + "".join(row) for y, row in enumerate(grid)]
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
    else:
        lines.append("  units: (none)")

    if player.cities:
        total_tiles = sum(len(c.citytiles) for c in player.cities.values())
        # Aggregate net upkeep across all cities so the model can see the
        # total per-night fuel drain without having to sum by hand.
        total_upkeep = sum(c.light_upkeep for c in player.cities.values())
        lines.append(
            f"  cities ({len(player.cities)}, {total_tiles} tiles, "
            f"total night upkeep {total_upkeep:g}/turn):"
        )
        for city in player.cities.values():
            tiles = ", ".join(f"({t.pos.x},{t.pos.y})" for t in city.citytiles)
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
_ID_TOKEN_RE = re.compile(r"[uUcC]_\d+")


def _normalize_command(cmd: str) -> str:
    """Fold common LLM variants into the engine's canonical command form.

    - Strips surrounding whitespace and list-marker leaders (``-``, ``*``,
      ``1.``, bullets) and trailing punctuation (``.``, ``,``, ``;``).
    - Strips trailing line-comments (``# ...`` and ``// ...``).
    - Strips ``[]``, ``()``, and ``<>`` wrappers around any token.
    - Collapses runs of internal whitespace to a single space.
    - Lowercases command head, unit/city id tokens (``U_1`` -> ``u_1``),
      and full direction names for the ``m`` command.
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
    # Unit and city ids are exclusively lowercase in the engine's wire
    # protocol (``u_N`` / ``c_N``). Models routinely uppercase them to match
    # the ASCII map letters (``U``/``T``), so fold every id-shaped token.
    parts = [p.lower() if _ID_TOKEN_RE.fullmatch(p) else p for p in parts]
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
