"""
Kaggle Environments interpreter for Reinforce Tactics.

This module implements the kaggle-environments interface, bridging the
Kaggle agent evaluation framework with the Reinforce Tactics game engine.

Required exports for kaggle-environments:
    - interpreter(state, env): Core game logic called each step
    - renderer(state, env): ASCII text renderer
    - specification: JSON specification dict
    - html_renderer(): Optional HTML/JS renderer
    - agents: Dict of built-in agent functions
"""

import json
import logging
import random
from os import path

import numpy as np

from .agents.random_agent import agent as _random_agent
from .agents.simple_bot_agent import agent as _simple_bot_agent
from .maps import BUILTIN_MAPS
from .reinforce_tactics_engine import UNIT_DATA, GameState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level game state storage
# Kaggle's interpreter is called once per step; we need to persist the
# GameState object across calls within the same episode.
# ---------------------------------------------------------------------------
_games = {}


# ---------------------------------------------------------------------------
# Engine balance overrides
# Applied to every game so the competition environment matches the balance the
# reference agents were trained under -- the engine_overrides block of
# configs/ppo/bootstrap_sweep/v52a_maxturn_scaled_draw.yaml:
#   * Warrior cost 200 -> 300 (economy; equalises it with the Mage so the
#     mono-Warrior local optimum loses its cost-efficiency edge)
#   * damage_model "flat" -> "hp_scaled" (a wounded unit deals proportionally
#     less, so focus-fire is decisive and the even-attrition stalemate that
#     drives max-turn draws is broken)
# Starting gold and income are left at the engine defaults (v52a does not
# override them). Resolved by GameState through the same engine_overrides path
# the trainer uses, so the vendored constants.py stays a faithful copy.
# ---------------------------------------------------------------------------
ENGINE_OVERRIDES = {
    "unit_data": {"W": {"cost": 300}},
    "damage_model": "hp_scaled",
}


# ---------------------------------------------------------------------------
# Specification (loaded from JSON)
# ---------------------------------------------------------------------------
_dirpath = path.dirname(__file__)
_jsonpath = path.abspath(path.join(_dirpath, "reinforce_tactics.json"))
with open(_jsonpath, encoding="utf-8") as _f:
    specification = json.load(_f)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
def interpreter(state, env):
    """
    Core game logic. Called once per step by the kaggle-environments engine.

    On the first call (``env.done == True``), this initialises the game.
    On subsequent calls it processes the active agent's actions, checks
    win/draw conditions, updates observations, and swaps the active player.

    Args:
        state: list of per-agent state structs. Each has:
            .action        - the action returned by the agent
            .reward        - read/write reward
            .status        - ACTIVE / INACTIVE / DONE / ERROR / INVALID / TIMEOUT
            .observation   - per-agent observation struct
        env: environment handle with:
            .configuration - merged configuration struct
            .done          - True on the initialisation call
            .steps         - list of all previous steps

    Returns:
        state (modified in-place)
    """
    key = id(env)

    # ------------------------------------------------------------------
    # Initialisation (first call after env.reset)
    # ------------------------------------------------------------------
    if env.done:
        return _interpreter_init(state, env, key)

    game = _games.get(key)
    if game is None:
        for agent in state:
            agent.status = "ERROR"
        return state

    # ------------------------------------------------------------------
    # Determine which agent is active
    # ------------------------------------------------------------------
    active_idx = _get_active_index(state)
    if active_idx is None:
        return state  # both done / error

    # ------------------------------------------------------------------
    # Execute agent actions, end turn, check outcomes
    # ------------------------------------------------------------------
    _process_turn(state, env, game, active_idx, key)

    return state


def _interpreter_init(state, env, key):
    """Handle the first interpreter call (game initialisation)."""
    seed = _resolve_seed(env)
    game = _init_game(env.configuration, seed)
    _games[key] = game
    _update_observations(state, game, env.configuration)
    state[0].status = "ACTIVE"
    state[1].status = "INACTIVE"
    return state


def _resolve_seed(env):
    """Resolve the episode seed from env.info / configuration, then scrub.

    Mirrors crawl/kaggriculture/orbit_wars: read env.info["seed"] first (the
    harness may have set it), then configuration.seed, then fall back to a
    random value. Clear configuration.seed so agents can't read it, and stash
    the resolved value on env.info["seed"] so it persists into the replay.
    """
    if not hasattr(env, "info") or env.info is None:
        env.info = {}
    seed = env.info.get("seed")
    if seed is None:
        seed = getattr(env.configuration, "seed", None)
    if seed is None:
        seed = random.randrange(2**31)
    try:
        env.configuration.seed = None
    except (AttributeError, TypeError):
        env.configuration["seed"] = None
    env.info["seed"] = seed
    return seed


def _process_turn(state, env, game, active_idx, key):
    """Process actions, end turn, and check win/draw conditions."""
    agent = state[active_idx]
    actions = agent.action if agent.action else []

    if not isinstance(actions, list):
        actions = [actions]

    game_player = active_idx + 1  # GameState uses 1-indexed players

    # Execute each action in the agent's action list
    if not _run_actions(state, game, actions, active_idx, game_player):
        return  # Agent lost due to invalid action

    # End the turn (income, healing, status effects, etc.)
    if not game.game_over:
        game.end_turn()

    # Check win condition
    if game.game_over:
        if game.winner is None:
            # Draw (e.g., game's own max_turns cap)
            for i in range(2):
                state[i].reward = 0
                state[i].status = "DONE"
        else:
            winner_idx = game.winner - 1
            state[winner_idx].reward = 1
            state[winner_idx].status = "DONE"
            state[1 - winner_idx].reward = -1
            state[1 - winner_idx].status = "DONE"
        _update_observations(state, game, env.configuration)
        _games.pop(key, None)
        return

    # Check draw (max turns)
    if game.turn_number >= env.configuration.episodeSteps:
        for i in range(2):
            state[i].reward = 0
            state[i].status = "DONE"
        _update_observations(state, game, env.configuration)
        _games.pop(key, None)
        return

    # Normal continuation: update observations and swap active player
    _update_observations(state, game, env.configuration)
    state[active_idx].status = "INACTIVE"
    state[1 - active_idx].status = "ACTIVE"


def _run_actions(state, game, actions, active_idx, game_player):
    """
    Execute the active agent's action list for the turn.

    Well-formed but illegal actions (unaffordable, occupied tile, out of
    range, at the per-player unit cap, unknown action/unit type) are skipped
    as no-ops and the turn continues. This suits the multi-action-per-turn
    API: one rejected action in a list shouldn't forfeit the whole episode,
    and ``get_legal_actions`` already lets agents avoid illegal moves. Only a
    malformed action (not a dict) is treated as a broken agent and forfeits.

    Returns True normally, or False if the agent forfeited (malformed action),
    in which case it has already been marked as lost.
    """
    for action in actions:
        if not isinstance(action, dict):
            _mark_agent_loss(state, active_idx)
            return False

        if action.get("type", "") == "end_turn":
            break

        # Illegal-but-well-formed action: skip it (no-op) and keep going.
        if not _execute_action(game, action, game_player):
            continue

        if game.game_over:
            break

    return True


def _mark_agent_loss(state, losing_idx):
    """Mark the agent at losing_idx as having lost."""
    state[losing_idx].status = "DONE"
    state[losing_idx].reward = -1
    state[1 - losing_idx].reward = 1
    state[1 - losing_idx].status = "DONE"


def _pad_map(map_rows, min_size=20):
    """
    Pad a small map to at least ``min_size x min_size`` by centering it in an
    ocean border, matching the upstream ``FileIO._pad_map`` behaviour.

    Args:
        map_rows: 2D list of tile code strings.
        min_size: Minimum dimension (default 20).

    Returns:
        A pandas DataFrame of the padded map.
    """
    import pandas as pd

    rows = len(map_rows)
    cols = len(map_rows[0]) if rows > 0 else 0

    if rows >= min_size and cols >= min_size:
        return pd.DataFrame(map_rows)

    new_h = max(rows, min_size)
    new_w = max(cols, min_size)

    padded = np.full((new_h, new_w), "o", dtype=object)

    offset_y = (new_h - rows) // 2
    offset_x = (new_w - cols) // 2

    for y in range(rows):
        for x in range(cols):
            padded[offset_y + y, offset_x + x] = map_rows[y][x]

    return pd.DataFrame(padded)


# ---------------------------------------------------------------------------
# Game Initialisation
# ---------------------------------------------------------------------------
def _select_map_by_seed(seed):
    """Pick a built-in map name. None picks randomly; otherwise deterministic."""
    names = sorted(BUILTIN_MAPS)
    if seed is None:
        return names[np.random.randint(0, len(names))]
    return names[int(seed) % len(names)]


# Tiles that can be flipped between by _mutate_map. Everything else
# (HQs, player buildings, towers, roads, water, ocean) is structural and
# left alone so balance and connectivity are preserved.
_MUTABLE_TERRAIN = ("p", "f", "m")
_MUTATIONS_PER_MAP = 4


def _mutate_map(map_rows, seed, num_mutations=_MUTATIONS_PER_MAP):
    """Apply small symmetric tile flips to a built-in map for variety.

    Each mutation picks a non-structural tile (plain/forest/mountain), changes
    it to a different terrain in the same set, and applies the same change at
    the point-symmetric position so the two starting sides stay balanced.
    """
    rng = random.Random(seed)
    rows = [list(r) for r in map_rows]
    height = len(rows)
    width = len(rows[0]) if height else 0

    candidates = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if rows[y][x] in _MUTABLE_TERRAIN
    ]
    if not candidates:
        return rows

    applied = 0
    attempts = 0
    while applied < num_mutations and attempts < num_mutations * 10:
        attempts += 1
        x, y = rng.choice(candidates)
        mx, my = width - 1 - x, height - 1 - y
        # Skip the centre tile of odd-dimensioned maps (no distinct mirror) and
        # any cell whose mirror is structural (so we don't asymmetrically flip).
        if (x, y) == (mx, my) or rows[my][mx] not in _MUTABLE_TERRAIN:
            continue
        choices = [t for t in _MUTABLE_TERRAIN if t != rows[y][x]]
        new_tile = rng.choice(choices)
        rows[y][x] = new_tile
        rows[my][mx] = new_tile
        applied += 1

    return rows


def _init_game(config, seed=None):
    """Create a new GameState from the Kaggle configuration.

    If ``mapName`` is set and matches a built-in, that map is used. Otherwise
    ``seed`` deterministically selects a map from BUILTIN_MAPS (sorted by name).
    The chosen map then gets small seed-driven, point-symmetric terrain flips
    so no two episodes play on an identical layout.
    """
    map_name = getattr(config, "mapName", "")

    if not map_name or map_name not in BUILTIN_MAPS:
        map_name = _select_map_by_seed(seed)

    map_rows = _mutate_map(BUILTIN_MAPS[map_name], seed)
    map_data = _pad_map(map_rows)

    enabled_units = [u.strip() for u in config.enabledUnits.split(",") if u.strip()]
    fog_of_war = bool(config.fogOfWar)

    game = GameState(
        map_data,
        num_players=2,
        max_turns=config.episodeSteps,
        enabled_units=enabled_units,
        fog_of_war=fog_of_war,
        engine_overrides=ENGINE_OVERRIDES,
    )

    # Override starting gold if configured
    starting_gold = config.startingGold
    game.player_gold = {1: starting_gold, 2: starting_gold}

    return game


# ---------------------------------------------------------------------------
# Action Execution
# ---------------------------------------------------------------------------
def _execute_action(game, action, player):
    """
    Translate a single action dict into a GameState method call.

    Returns True on success, False on invalid action.
    """
    atype = action.get("type", "")

    handlers = {
        "create_unit": _exec_create_unit,
        "move": _exec_move,
        "attack": _exec_attack,
        "seize": _exec_seize,
        "heal": _exec_heal,
        "cure": _exec_cure,
        "paralyze": _exec_paralyze,
        "haste": _exec_haste,
        "defence_buff": _exec_defence_buff,
        "attack_buff": _exec_attack_buff,
        "end_turn": lambda _g, _a, _p: True,
    }

    handler = handlers.get(atype)
    if handler is None:
        return False

    try:
        return handler(game, action, player)
    except Exception:  # pylint: disable=broad-except
        logger.exception("Error executing action: %s", action)
        return False


def _exec_create_unit(game, action, player):
    """Handle create_unit action."""
    unit_type = action.get("unit_type", "")
    x = int(action.get("x", -1))
    y = int(action.get("y", -1))
    if unit_type not in UNIT_DATA:
        return False
    return game.create_unit(unit_type, x, y, player) is not None


def _exec_move(game, action, player):
    """Handle move action."""
    from_x = int(action.get("from_x", -1))
    from_y = int(action.get("from_y", -1))
    to_x = int(action.get("to_x", -1))
    to_y = int(action.get("to_y", -1))
    unit = game.get_unit_at_position(from_x, from_y)
    if unit is None or unit.player != player:
        return False
    return game.move_unit(unit, to_x, to_y)


def _exec_attack(game, action, player):
    """Handle attack action."""
    from_x = int(action.get("from_x", -1))
    from_y = int(action.get("from_y", -1))
    to_x = int(action.get("to_x", -1))
    to_y = int(action.get("to_y", -1))
    attacker = game.get_unit_at_position(from_x, from_y)
    target = game.get_unit_at_position(to_x, to_y)
    if attacker is None or target is None:
        return False
    # ``not attacker.can_attack`` mirrors get_legal_actions (which only offers
    # attack/seize/abilities while can_attack) and prevents a unit from acting
    # more than once per turn -- e.g. attacking twice or seizing a structure
    # to completion in a single turn.
    if attacker.player != player or target.player == player or not attacker.can_attack:
        return False
    game.attack(attacker, target)
    return True


def _exec_seize(game, action, player):
    """Handle seize action."""
    x = int(action.get("x", -1))
    y = int(action.get("y", -1))
    unit = game.get_unit_at_position(x, y)
    # ``not unit.can_attack`` keeps a unit to one action per turn: without it a
    # unit could seize a structure repeatedly in a single turn (each seize deals
    # its HP as damage), capturing a tower/HQ far faster than intended.
    if unit is None or unit.player != player or not unit.can_attack:
        return False
    tile = game.grid.get_tile(x, y)
    if tile is None or not tile.is_capturable() or tile.player == player:
        return False
    game.seize(unit)
    return True


def _exec_heal(game, action, player):
    """Handle heal action."""
    healer, target = _get_source_target(game, action, player, "C")
    if healer is None:
        return False
    return game.heal(healer, target) > 0


def _exec_cure(game, action, player):
    """Handle cure action."""
    curer, target = _get_source_target(game, action, player, "C")
    if curer is None:
        return False
    return game.cure(curer, target)


def _exec_paralyze(game, action, player):
    """Handle paralyze action."""
    mage, target = _get_source_target(game, action, player, "M")
    if mage is None:
        return False
    return game.paralyze(mage, target)


def _exec_haste(game, action, player):
    """Handle haste action."""
    sorcerer, target = _get_source_target(game, action, player, "S")
    if sorcerer is None:
        return False
    return game.haste(sorcerer, target)


def _exec_defence_buff(game, action, player):
    """Handle defence_buff action."""
    sorcerer, target = _get_source_target(game, action, player, "S")
    if sorcerer is None:
        return False
    return game.defence_buff(sorcerer, target)


def _exec_attack_buff(game, action, player):
    """Handle attack_buff action."""
    sorcerer, target = _get_source_target(game, action, player, "S")
    if sorcerer is None:
        return False
    return game.attack_buff(sorcerer, target)


def _get_source_target(game, action, player, required_type):
    """
    Extract source and target units from an action dict.

    Returns (source, target) or (None, None) if validation fails.
    """
    from_x = int(action.get("from_x", -1))
    from_y = int(action.get("from_y", -1))
    to_x = int(action.get("to_x", -1))
    to_y = int(action.get("to_y", -1))
    source = game.get_unit_at_position(from_x, from_y)
    target = game.get_unit_at_position(to_x, to_y)
    if source is None or target is None:
        return None, None
    # ``not source.can_attack``: heal/cure/paralyze/haste/buffs all consume the
    # unit's action (like attack/seize), so gate them the same way the action
    # mask does -- one such action per unit per turn.
    if source.player != player or source.type != required_type or not source.can_attack:
        return None, None
    return source, target


# ---------------------------------------------------------------------------
# Observation Serialisation
# ---------------------------------------------------------------------------
def _update_observations(state, game, config):
    """Serialise the current GameState into each agent's observation."""
    board = _serialize_board(game)
    structures = _serialize_structures(game)
    gold = [game.player_gold.get(1, 0), game.player_gold.get(2, 0)]

    fog_of_war = bool(config.fogOfWar)

    for i in range(2):
        obs = state[i].observation
        obs.board = board
        obs.structures = structures
        obs.gold = gold
        obs.turnNumber = game.turn_number
        obs.mapWidth = game.grid.width
        obs.mapHeight = game.grid.height

        player = i + 1  # 1-indexed game player

        if fog_of_war:
            # Update visibility and filter units per player
            game.update_visibility(player)
            obs.units = _serialize_units(game, visible_for_player=player)
        else:
            obs.units = _serialize_units(game)


def _serialize_board(game):
    """Convert the game grid to a 2D array of terrain type codes."""
    board = []
    for y in range(game.grid.height):
        row = []
        for x in range(game.grid.width):
            tile = game.grid.get_tile(x, y)
            row.append(tile.type)
        board.append(row)
    return board


def _serialize_structures(game):
    """Convert capturable structures to a list of dicts."""
    structures = []
    for row in game.grid.tiles:
        for tile in row:
            if tile.is_capturable():
                structures.append(
                    {
                        "x": tile.x,
                        "y": tile.y,
                        "type": tile.type,
                        "owner": tile.player if tile.player else 0,
                        "hp": tile.health if tile.health is not None else 0,
                        "maxHp": tile.max_health if tile.max_health is not None else 0,
                    }
                )
    return structures


def _serialize_units(game, visible_for_player=None):
    """
    Convert units to a list of dicts.

    If ``visible_for_player`` is set and fog-of-war is enabled, only units
    visible to that player (own units + units in visible tiles) are included.
    """
    units = []
    for unit in game.units:
        # Fog of war filtering
        if visible_for_player is not None:
            if unit.player != visible_for_player:
                if not game.is_position_visible(unit.x, unit.y, visible_for_player):
                    continue

        units.append(
            {
                "type": unit.type,
                "owner": unit.player,
                "x": unit.x,
                "y": unit.y,
                "hp": unit.health,
                "maxHp": unit.max_health,
                "canMove": unit.can_move,
                "canAttack": unit.can_attack,
                "paralyzedTurns": unit.paralyzed_turns,
                "isHasted": unit.is_hasted,
                "distanceMoved": unit.distance_moved,
                "defenceBuffTurns": unit.defence_buff_turns,
                "attackBuffTurns": unit.attack_buff_turns,
            }
        )
    return units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_active_index(state):
    """Return the index of the ACTIVE agent, or None if none."""
    for i in range(2):
        if state[i].status == "ACTIVE":
            return i
    return None


# ---------------------------------------------------------------------------
# Renderer (ASCII / ANSI)
# ---------------------------------------------------------------------------
def renderer(state, env):
    """Return an ASCII text representation of the current board."""
    if not state or len(state) < 2:
        return "No state available."

    obs = state[0].observation
    board = obs.board if hasattr(obs, "board") else []
    units_list = obs.units if hasattr(obs, "units") else []
    gold = obs.gold if hasattr(obs, "gold") else [0, 0]
    turn = obs.turnNumber if hasattr(obs, "turnNumber") else 0

    if not board:
        return "Board not initialised."

    # Build unit lookup
    unit_map = {}
    for u in units_list:
        unit_map[(u["x"], u["y"])] = u

    # Tile display characters
    tile_chars = {
        "p": ".",
        "w": "~",
        "m": "^",
        "f": "T",
        "r": "=",
        "b": "B",
        "h": "H",
        "t": "#",
        "o": "~",
    }

    lines = []
    lines.append(f"Turn {turn}  |  P1 Gold: {gold[0]}  |  P2 Gold: {gold[1]}")
    lines.append(f"P1 Status: {state[0].status}  |  P2 Status: {state[1].status}")
    lines.append("")

    for y, row in enumerate(board):
        line = ""
        for x, cell in enumerate(row):
            pos = (x, y)
            if pos in unit_map:
                u = unit_map[pos]
                # Show unit type with player indicator (lowercase=p1, uppercase=p2)
                ch = u["type"]
                line += ch.lower() if u["owner"] == 1 else ch.upper()
            else:
                line += tile_chars.get(cell, "?")
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML Renderer
# ---------------------------------------------------------------------------
def html_renderer():
    """Return the built visualizer HTML, or an empty string if not built yet."""
    htmlpath = path.join(_dirpath, "visualizer", "default", "dist", "index.html")
    if path.exists(htmlpath):
        with open(htmlpath, encoding="utf-8") as f:
            return f.read()
    return ""


# ---------------------------------------------------------------------------
# Built-in Agents
# ---------------------------------------------------------------------------


def _noop_agent(observation, configuration):
    """Agent that always ends its turn immediately (does nothing)."""
    return [{"type": "end_turn"}]


agents = {
    "random": _random_agent,
    "aggressive": _simple_bot_agent,
    "simple_bot": _simple_bot_agent,
    "noop": _noop_agent,
}
