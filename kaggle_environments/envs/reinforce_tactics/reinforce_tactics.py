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
from os import path

import numpy as np

from .agents.random_agent import agent as _random_agent
from .agents.simple_bot_agent import agent as _simple_bot_agent
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
    game = _init_game(env.configuration)
    _games[key] = game
    _update_observations(state, game, env.configuration)
    state[0].status = "ACTIVE"
    state[1].status = "INACTIVE"
    return state


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
    Execute all actions for the active agent.

    Returns True if all actions were valid, False if the agent made an
    invalid action (in which case the agent is marked as lost).
    """
    for action in actions:
        if not isinstance(action, dict):
            _mark_agent_loss(state, active_idx)
            return False

        if action.get("type", "") == "end_turn":
            break

        if not _execute_action(game, action, game_player):
            _mark_agent_loss(state, active_idx)
            return False

        if game.game_over:
            break

    return True


def _mark_agent_loss(state, losing_idx):
    """Mark the agent at losing_idx as having lost."""
    state[losing_idx].status = "DONE"
    state[losing_idx].reward = -1
    state[1 - losing_idx].reward = 1
    state[1 - losing_idx].status = "DONE"


# ---------------------------------------------------------------------------
# Built-in maps (vendored from the repository's maps/1v1/*.csv).
# Format: CSV-style rows of tile-code strings (see README for tile codes),
# parsed into a 2D list. Small maps are auto-padded to 20x20 with ocean
# borders at game start.
# ---------------------------------------------------------------------------
def _map_rows(block):
    """Parse a CSV-style map block into a 2D list of tile-code strings."""
    return [[cell.strip() for cell in line.split(",")] for line in block.strip().splitlines()]


BUILTIN_MAPS = {
    "beginner": _map_rows(
        """
            h_1,b_1,p,p,p,p
            b_1,p,p,p,p,p
            p,p,t,t,p,p
            p,p,t,t,p,p
            p,p,p,p,p,b_2
            p,p,p,p,b_2,h_2
        """
    ),
    "cavalry_charge": _map_rows(
        """
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,h_1,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,r,r,r,r,r,r,r,r,r,r,r,r,r,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,t,t,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,t,t,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,p,p,p,p,p,p,p,p,r,p,p,p
            p,p,p,r,r,r,r,r,r,r,r,r,r,r,r,r,r,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,h_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
        """
    ),
    "center_mountains": _map_rows(
        """
            b_1,t,p,p,p,p,p,p,p,p,p,p,p,p,p,t,p,t
            t,p,p,p,p,p,p,p,p,p,p,p,p,m,m,m,m,p
            p,p,h_1,p,b_1,p,p,r,r,r,r,r,p,p,p,p,m,t
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,m,p
            p,p,b_1,p,p,p,p,p,p,p,p,p,p,b,r,p,m,p
            p,p,p,p,p,t,p,p,f,p,p,p,p,p,r,p,p,p
            p,p,p,p,p,p,p,f,m,m,p,b,p,p,r,p,p,p
            p,p,p,r,p,p,f,f,m,m,p,p,r,r,r,p,p,p
            p,p,p,r,p,f,f,p,m,m,p,p,p,p,r,p,p,p
            p,p,p,r,p,p,p,p,m,m,p,f,f,p,r,p,p,p
            p,p,p,r,r,r,p,p,m,m,f,f,p,p,r,p,p,p
            p,p,p,r,p,p,b,p,m,m,f,p,p,p,p,p,p,p
            p,p,p,r,p,p,p,p,p,f,p,p,t,p,p,p,p,p
            p,m,p,r,b,p,p,p,p,p,p,p,p,p,p,b_2,p,p
            p,m,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            t,m,p,p,p,p,r,r,r,r,r,p,p,b_2,p,h_2,p,p
            p,m,m,m,m,p,p,p,p,p,p,p,p,p,p,p,p,t
            t,p,t,p,p,p,p,p,p,p,p,p,p,p,p,p,t,b_2
        """
    ),
    "cleric_vigil": _map_rows(
        """
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,h_1,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,m,m,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,t,t,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,t,t,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,m,m,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,h_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
        """
    ),
    "corner_points": _map_rows(
        """
            h_1,b_1,p,p,p,p,p,p,p,p,f,b
            b_1,f,p,t,p,p,m,p,p,p,f,f
            p,p,p,p,p,p,p,f,p,b,p,p
            p,t,p,p,o,p,p,f,f,p,p,p
            p,p,p,p,p,t,t,p,p,m,p,p
            p,p,m,p,p,t,t,p,p,p,p,p
            p,p,p,f,f,p,p,o,p,p,t,p
            p,p,b,p,f,p,p,p,p,p,p,p
            f,f,p,p,p,m,p,p,t,p,f,b_2
            b,f,p,p,p,p,p,p,p,p,b_2,h_2
        """
    ),
    "crossroads": _map_rows(
        """
            w,w,w,w,w,p,p,p,p,p,p,p,p,p,p
            w,h_1,b_1,p,w,p,t,p,p,p,p,p,p,p,p
            w,b_1,p,p,p,f,p,p,p,p,p,p,t,p,p
            w,w,p,p,t,f,p,p,p,p,p,p,p,p,p
            p,p,p,p,f,p,p,b,p,p,p,p,p,p,p
            p,t,p,p,p,p,r,r,r,p,p,t,p,p,p
            p,p,p,p,p,r,r,t,r,r,p,p,p,p,p
            p,p,p,b,r,r,t,b,t,r,r,b,p,p,p
            p,p,p,p,p,r,r,t,r,r,p,p,p,p,p
            p,p,p,t,p,p,r,r,r,p,p,p,p,t,p
            p,p,p,p,p,p,p,b,p,p,p,p,p,p,p
            p,p,t,p,p,p,p,p,p,p,p,p,p,w,w
            p,p,p,p,p,p,p,p,p,p,p,p,p,b_2,w
            p,p,p,p,p,p,t,p,p,p,w,b_2,h_2,w,w
            p,p,p,p,p,p,p,p,p,p,w,w,w,w,w
        """
    ),
    "difficult_terrain": _map_rows(
        """
            p,p,p,p,p,p,p,p,p,p
            p,f,p,m,p,r,f,m,b,w
            p,f,h_1,b_1,w,p,w,p,m,w
            p,w,f,r,m,b,m,m,p,w
            p,f,b_1,r,p,r,r,p,m,w
            p,m,p,r,m,m,r,b_2,p,w
            p,p,f,b,m,p,r,p,m,m
            p,w,w,m,m,b_2,r,h_2,f,w
            p,b,f,m,p,f,f,f,p,w
            p,m,f,w,w,f,m,f,f,w
        """
    ),
    "funnel_point": _map_rows(
        """
            h_1,p,p,p,p,p,b,p,p,p,p,p,b_1
            p,p,p,o,p,p,p,p,p,o,p,o,p
            p,p,b_1,p,m,p,m,p,m,o,p,p,p
            p,o,p,t,m,p,m,p,m,t,o,o,p
            p,p,m,m,m,p,m,p,m,m,m,p,p
            p,f,p,f,p,f,b,f,p,f,p,f,p
            p,p,m,m,m,p,m,p,m,m,m,p,p
            p,o,o,t,m,p,m,p,m,t,p,o,p
            p,p,p,o,m,p,m,p,m,p,b_2,p,p
            p,o,p,o,p,p,p,p,p,o,p,p,p
            b_2,p,p,p,p,p,b,p,p,p,p,p,h_2
        """
    ),
    "intermediate": _map_rows(
        """
            h_1,b_1,p,p,p,o,o
            b_1,p,p,p,p,p,o
            p,p,t,p,p,f,p
            p,p,p,b,p,p,p
            p,f,p,p,t,p,p
            o,p,p,p,p,p,b_2
            o,o,p,p,p,b_2,h_2
        """
    ),
    "island_fortress": _map_rows(
        """
            h_1,b_1,p,p,w,w,w,p,p,w,w,w,p,p,p,p
            b_1,p,p,t,w,w,p,p,p,p,w,w,t,p,p,p
            p,p,f,p,w,p,p,b,b,p,p,w,p,f,p,p
            p,t,p,p,p,p,t,p,p,t,p,p,p,p,t,p
            w,w,w,p,p,p,p,w,w,p,p,p,p,w,w,w
            w,w,p,p,t,p,w,w,w,w,p,t,p,p,w,w
            w,p,p,b,p,p,p,t,t,p,p,p,b,p,p,w
            p,p,r,p,w,p,r,r,r,r,p,w,p,r,p,p
            p,p,r,p,w,p,r,r,r,r,p,w,p,r,p,p
            w,p,p,b,p,p,p,t,t,p,p,p,b,p,p,w
            w,w,p,p,t,p,w,w,w,w,p,t,p,p,w,w
            w,w,w,p,p,p,p,w,w,p,p,p,p,w,w,w
            p,t,p,p,p,p,t,p,p,t,p,p,p,p,t,p
            p,p,f,p,w,p,p,b,b,p,p,w,p,f,p,p
            p,p,p,t,w,w,p,p,p,p,w,w,t,p,p,b_2
            p,p,p,p,w,w,w,p,p,w,w,w,p,p,b_2,h_2
        """
    ),
    "last_stand": _map_rows(
        """
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,h_1,t_1,p,p,p,p,m,m,p,p,m,m,p,p,p,p,p,p,p
            p,t_1,p,p,f,p,p,p,p,p,p,p,p,p,p,f,p,p,p,p
            p,p,p,p,f,p,p,p,p,p,p,p,p,p,p,f,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,m,p,p,p,t,p,p,t,p,p,p,m,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,f,p,p,p,p,p,p,p,p,p,p,p,p,f,p,p,p
            p,p,p,p,p,p,p,t,p,p,p,p,t,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,m,m,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,m,m,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,t,p,p,p,p,t,p,p,p,p,p,p,p
            p,p,p,f,p,p,p,p,p,p,p,p,p,p,p,p,f,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,m,p,p,p,t,p,p,t,p,p,p,m,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,f,p,p,p,p,p,p,p,p,p,p,f,p,p,p,p
            p,p,p,p,f,p,p,p,p,p,p,p,p,p,p,f,p,t_2,p,p
            p,p,p,p,p,p,p,m,m,p,p,m,m,p,p,p,p,t_2,h_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
        """
    ),
    "mage_showdown": _map_rows(
        """
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,h_1,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,m,m,p,p,p,p,m,m,p,p,p,p,p,p
            p,p,p,p,p,p,m,p,p,p,p,p,p,m,p,p,p,p,p,p
            p,p,p,p,p,p,m,p,p,p,p,p,p,m,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,w,w,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,w,t,t,w,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,w,t,t,w,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,w,w,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,m,p,p,p,p,p,p,m,p,p,p,p,p,p
            p,p,p,p,p,p,m,p,p,p,p,p,p,m,p,p,p,p,p,p
            p,p,p,p,p,p,m,m,p,p,p,p,m,m,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,h_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
        """
    ),
    "mountain_snipers": _map_rows(
        """
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,h_1,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,m,m,m,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,m,m,m,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,m,m,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,m,m,p,p,p,p,m,m,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,t,t,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,t,t,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,m,m,p,p,p,p,m,m,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,m,m,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,m,m,m,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,m,m,m,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,h_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
        """
    ),
    "rogue_flank": _map_rows(
        """
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,h_1,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,f,f,p,p,p,p,p,p,f,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,f,f,f,f,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,t,t,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,t,t,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,f,f,f,f,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,p,p,p,p,p,p,p,p,f,p,p,p,p,p
            p,p,p,p,p,f,f,p,p,p,p,p,p,f,f,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,h_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
        """
    ),
    "skirmish": _map_rows(
        """
            h_1,b_1,p,p,p,p,p,b
            b_1,p,f,p,p,p,p,p
            p,f,p,t,p,m,p,p
            p,p,p,p,b,p,p,p
            p,p,p,b,p,p,p,p
            p,p,m,p,t,p,f,p
            p,p,p,p,p,f,p,b_2
            b,p,p,p,p,p,b_2,h_2
        """
    ),
    "sorcerer_cabal": _map_rows(
        """
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,h_1,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,t_1,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,f,f,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,f,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,m,m,p,p,m,m,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,t,t,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,t,t,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,t,t,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,t,t,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,m,m,p,p,m,m,p,p,p,p,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,f,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,f,f,p,p,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,t_2,h_2,p
            p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p
        """
    ),
    "starter": _map_rows(
        """
            o,o,o,o,o,o
            o,h_1,b_1,p,p,o
            o,b_1,p,t,p,o
            o,p,t,p,b_2,o
            o,p,p,b_2,h_2,o
            o,o,o,o,o,o
        """
    ),
    "the_narrows": _map_rows(
        """
            m,m,m,m,m,m,p,p,p,p,m,m,m,m,m,m
            m,m,h_1,b_1,p,p,p,t,p,p,p,b_2,h_2,m,m,m
            m,m,b_1,p,p,f,p,p,p,f,p,p,b_2,m,m,m
            m,m,p,p,t,p,p,r,p,p,t,p,p,m,m,m
            m,p,p,f,p,p,r,r,r,p,p,f,p,p,m,m
            p,p,t,p,m,m,r,b,r,m,m,p,t,p,p,m
            p,p,p,m,m,m,r,t,r,m,m,m,p,p,p,m
            p,r,r,r,r,r,r,p,r,r,r,r,r,r,p,m
            p,r,r,r,r,r,r,p,r,r,r,r,r,r,p,m
            p,p,p,m,m,m,r,t,r,m,m,m,p,p,p,m
            p,p,t,p,m,m,r,b,r,m,m,p,t,p,p,m
            m,p,p,f,p,p,r,r,r,p,p,f,p,p,m,m
            m,m,p,p,t,p,p,r,p,p,t,p,p,m,m,m
            m,m,b_1,p,p,f,p,p,p,f,p,p,b_2,m,m,m
            m,m,b_1,p,p,p,p,t,p,p,p,p,b_2,m,m,m
            m,m,m,m,m,m,p,p,p,p,m,m,m,m,m,m
        """
    ),
    "tower_rush": _map_rows(
        """
            h_1,b_1,p,p,t,p,p,p,p,t,p,p,b_2,h_2
            b_1,p,p,f,p,p,t,t,p,p,f,p,p,b_2
            p,p,t,p,p,b,p,p,b,p,p,t,p,p
            p,f,p,p,t,p,p,p,p,t,p,p,f,p
            t,p,p,t,p,p,b,b,p,p,t,p,p,t
            p,p,b,p,p,t,p,p,t,p,p,b,p,p
            p,t,p,p,b,p,t,t,p,b,p,p,t,p
            p,t,p,p,b,p,t,t,p,b,p,p,t,p
            p,p,b,p,p,t,p,p,t,p,p,b,p,p
            t,p,p,t,p,p,b,b,p,p,t,p,p,t
            p,f,p,p,t,p,p,p,p,t,p,p,f,p
            p,p,t,p,p,b,p,p,b,p,p,t,p,p
            b_1,p,p,f,p,p,t,t,p,p,f,p,p,b_2
            h_1,b_1,p,p,t,p,p,p,p,t,p,p,b_2,h_2
        """
    ),
}


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
# Map Generation (inlined to avoid pygame dependency from utils package)
# ---------------------------------------------------------------------------
def _generate_map(width, height, num_players=2):
    """
    Generate a random map as a pandas DataFrame.

    This mirrors ``FileIO.generate_random_map`` but is self-contained so the
    kaggle adapter does not need to import the ``reinforcetactics.utils``
    package (which transitively pulls in pygame via ReplayPlayer).
    """
    import pandas as pd

    width = max(width, 20)
    height = max(height, 20)

    map_data = np.full((height, width), "o", dtype=object)

    num_tiles = width * height

    # Forests (10%)
    for _ in range(num_tiles // 10):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        map_data[y, x] = "f"

    # Mountains (5%)
    for _ in range(num_tiles // 20):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        map_data[y, x] = "m"

    # Water (3%)
    for _ in range(num_tiles // 33):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        map_data[y, x] = "w"

    # Player headquarters and buildings
    if num_players >= 1:
        map_data[1, 1] = "h_1"
        map_data[1, 2] = "b_1"
        map_data[2, 1] = "b_1"

    if num_players >= 2:
        map_data[height - 2, width - 2] = "h_2"
        map_data[height - 2, width - 3] = "b_2"
        map_data[height - 3, width - 2] = "b_2"

    if num_players >= 3:
        map_data[1, width - 2] = "h_3"
        map_data[1, width - 3] = "b_3"
        map_data[2, width - 2] = "b_3"

    if num_players >= 4:
        map_data[height - 2, 1] = "h_4"
        map_data[height - 2, 2] = "b_4"
        map_data[height - 3, 1] = "b_4"

    # Neutral towers in centre
    cx, cy = width // 2, height // 2
    for dx, dy in [(0, 0), (3, 0), (0, 3), (3, 3)]:
        x, y = cx + dx - 2, cy + dy - 2
        if 0 <= x < width and 0 <= y < height:
            if map_data[y, x] == "p":
                map_data[y, x] = "t"

    return pd.DataFrame(map_data)


# ---------------------------------------------------------------------------
# Game Initialisation
# ---------------------------------------------------------------------------
def _init_game(config):
    """Create a new GameState from the Kaggle configuration."""
    map_name = getattr(config, "mapName", "")

    if map_name and map_name in BUILTIN_MAPS:
        # Use a built-in map (padded to minimum 20x20)
        map_data = _pad_map(BUILTIN_MAPS[map_name])
    else:
        # Random generation
        width = config.mapWidth
        height = config.mapHeight
        seed = config.mapSeed

        if seed >= 0:
            np.random.seed(seed)

        map_data = _generate_map(width, height, num_players=2)

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
    if attacker.player != player or target.player == player:
        return False
    game.attack(attacker, target)
    return True


def _exec_seize(game, action, player):
    """Handle seize action."""
    x = int(action.get("x", -1))
    y = int(action.get("y", -1))
    unit = game.get_unit_at_position(x, y)
    if unit is None or unit.player != player:
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
    if source.player != player or source.type != required_type:
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
    """Return JavaScript for browser-based rendering (placeholder)."""
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
