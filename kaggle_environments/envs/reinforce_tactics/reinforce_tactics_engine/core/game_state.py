"""
Core game state management without rendering dependencies.

Vendored copy for Kaggle Environments — save/replay features that depend
on ``reinforcetactics.utils.file_io`` are stubbed out.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..constants import (
    ALL_UNIT_TYPES,
    BUILDING_INCOME,
    HEADQUARTERS_INCOME,
    MAX_UNITS_PER_PLAYER,
    STARTING_GOLD,
    TOWER_INCOME,
    UNIT_DATA,
    TileType,
)
from ..game.mechanics import GameMechanics
from .grid import TileGrid
from .unit import Unit
from .visibility import VISIBLE, VisibilityMap, get_visible_units

# Configure logging
logger = logging.getLogger(__name__)


class GameState:
    """Manages the core game state without rendering."""

    # All available unit types (sourced from the shared engine constant)
    ALL_UNIT_TYPES = ALL_UNIT_TYPES

    @staticmethod
    def _resolve_engine_overrides(
        overrides: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, int], int]:
        """Merge a sparse override overlay over the module engine constants.

        Returns ``(unit_data, income_rates, starting_gold)`` fully resolved.
        ``unit_data`` is a deep copy of :data:`UNIT_DATA` with per-unit,
        per-field deltas applied (so the shared module dict is never
        mutated). Unknown unit codes / stat fields raise ``KeyError`` /
        ``ValueError`` early -- a typo in a balance sweep should fail loud,
        not silently train on the wrong stats.
        """
        unit_data = copy.deepcopy(UNIT_DATA)
        income_rates = {
            "headquarters": HEADQUARTERS_INCOME,
            "building": BUILDING_INCOME,
            "tower": TOWER_INCOME,
        }
        starting_gold = STARTING_GOLD
        if not overrides:
            return unit_data, income_rates, starting_gold

        if "starting_gold" in overrides:
            starting_gold = int(overrides["starting_gold"])
        for ov_key, rate_key in (
            ("headquarters_income", "headquarters"),
            ("building_income", "building"),
            ("tower_income", "tower"),
        ):
            if ov_key in overrides:
                income_rates[rate_key] = int(overrides[ov_key])

        unit_overrides = overrides.get("unit_data") or {}
        for code, fields in unit_overrides.items():
            if code not in unit_data:
                raise KeyError(f"engine_overrides.unit_data: unknown unit code '{code}'")
            for field, value in fields.items():
                if field not in unit_data[code]:
                    raise ValueError(
                        f"engine_overrides.unit_data['{code}']: unknown stat field "
                        f"'{field}' (valid: {sorted(unit_data[code])})"
                    )
                unit_data[code][field] = value
        return unit_data, income_rates, starting_gold

    @staticmethod
    def _resolve_max_units_per_player(overrides: Dict[str, Any]) -> int:
        """Resolve the per-player unit cap from the engine-override overlay.

        Defaults to :data:`MAX_UNITS_PER_PLAYER`. A positive int is required
        -- a cap <= 0 would forbid all unit creation, which is never the
        intent and should fail loud rather than silently soft-lock a game.

        The cap is a *creation gate*, not a retroactive trim: it blocks new
        ``create_unit`` calls once a player is at the cap but never removes
        existing units, so a scenario that starts a side at or above the cap
        (or a sweep that sets the cap below the starting army) simply can't
        grow until attrition drops the count. It is therefore a soft ceiling
        on growth, not a hard guarantee of ``<= cap`` units at every instant.
        """
        if "max_units_per_player" not in (overrides or {}):
            return MAX_UNITS_PER_PLAYER
        val = int(overrides["max_units_per_player"])
        if val <= 0:
            raise ValueError(f"engine_overrides.max_units_per_player must be a positive int, got {val}")
        return val

    @staticmethod
    def _resolve_damage_model(overrides: Dict[str, Any]) -> str:
        """Resolve the combat damage model from the engine-override overlay.

        ``"flat"`` (default) reproduces legacy HP-independent damage.
        ``"hp_scaled"`` multiplies outgoing damage by the attacker's current
        HP fraction. An unknown value fails loud rather than silently
        training on an unintended combat model.
        """
        model = (overrides or {}).get("damage_model", "flat")
        if model not in ("flat", "hp_scaled"):
            raise ValueError(f"engine_overrides.damage_model must be 'flat' or 'hp_scaled', got {model!r}")
        return model

    # YAML override key -> structure tile-type code. Lets a balance sweep tune
    # capture difficulty (e.g. ``headquarters_health: 30`` halves a Warrior's
    # HQ-capture time) from the config surface instead of editing constants.py.
    _STRUCTURE_HEALTH_KEYS = {
        "tower_health": "t",
        "building_health": "b",
        "headquarters_health": "h",
    }

    @classmethod
    def _resolve_structure_health(cls, overrides: Dict[str, Any]) -> Dict[str, int]:
        """Resolve per-structure max-HP overrides into ``{tile_code: hp}``.

        Only keys present in ``overrides`` appear in the result; absent
        structures keep their ``constants.py`` defaults. Non-positive values
        fail loud (a structure with <=0 HP would be captured on the first
        seize / be nonsensical for regen).
        """
        resolved: Dict[str, int] = {}
        for ov_key, code in cls._STRUCTURE_HEALTH_KEYS.items():
            if ov_key in (overrides or {}):
                val = int(overrides[ov_key])
                if val <= 0:
                    raise ValueError(f"engine_overrides.{ov_key} must be a positive int, got {val}")
                resolved[code] = val
        return resolved

    def _apply_structure_health_overrides(self) -> None:
        """Overlay resolved structure-HP overrides onto the freshly-built grid.

        ``TileGrid`` constructs structure tiles at the ``constants.py`` HP, so
        this runs right after grid creation while every structure is at full
        health -- setting both ``max_health`` and ``health`` keeps the tile
        consistent (regen scales off ``max_health``; capture resets to it).
        """
        if not self.structure_health:
            return
        for row in self.grid.tiles:
            for tile in row:
                override_hp = self.structure_health.get(tile.type)
                if override_hp is not None and tile.is_capturable():
                    tile.max_health = override_hp
                    tile.health = override_hp

    def __init__(
        self,
        map_data,
        num_players: int = 2,
        max_turns: Optional[int] = None,
        enabled_units: Optional[List[str]] = None,
        fog_of_war: bool = False,
        engine_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the game state.

        Args:
            map_data: 2D array containing map information
            num_players: Number of players (must be 2; the kaggle env spec is 2-player)
            max_turns: Maximum turns for the game (None = unlimited)
            enabled_units: List of enabled unit types (default all units enabled)
            fog_of_war: Enable fog of war (default False for backward compatibility)
            engine_overrides: Optional sparse overlay over the non-YAML
                engine constants (``constants.py``), so balance can be
                varied/recorded as config instead of a code edit. Shape::

                    {
                      "starting_gold": int,
                      "headquarters_income": int,
                      "building_income": int,
                      "tower_income": int,
                      "tower_health": int,         # structure max-HP overrides
                      "building_health": int,      #   (capture-difficulty lever)
                      "headquarters_health": int,
                      "damage_model": "flat" | "hp_scaled",  # combat model
                      "max_units_per_player": int,  # per-player unit cap
                      "unit_data": {CODE: {field: value}},  # sparse deltas
                    }

                Every key is optional; absent keys fall back to the module
                constant, so ``None`` / ``{}`` is byte-identical to today.
                The resolved tables (``self.unit_data``, ``self.income_rates``,
                ``self.starting_gold``) are this game's single source of
                truth -- units and income read them, never the global
                constant -- so an override can't leak or be half-applied.
        """
        if num_players != 2:
            raise ValueError(f"GameState (kaggle vendored copy) is 2-player only; got num_players={num_players}")
        self.grid = TileGrid(map_data)
        self.units: List[Unit] = []
        # Monotonic per-game unit-id counter. Stamped on every newly
        # created unit; written into the replay log alongside each
        # action so the v3 replay player can look up units by id
        # rather than by brittle (x, y) position. Restored from saves
        # in ``from_dict`` so post-load creations don't collide.
        self._next_unit_id: int = 0
        self.current_player: int = 1
        self.num_players: int = num_players
        self.engine_overrides: Dict[str, Any] = dict(engine_overrides) if engine_overrides else {}
        (
            self.unit_data,
            self.income_rates,
            self.starting_gold,
        ) = self._resolve_engine_overrides(self.engine_overrides)
        # Combat damage model (engine-side, config-surfaced via engine_overrides
        # so it's snapshotted into config.json like the economy). "flat"
        # (default, legacy) = HP-independent damage; "hp_scaled" = damage
        # multiplied by the attacker's current HP fraction (decisive combat;
        # consistent with seize, which is already HP-scaled).
        self.damage_model: str = self._resolve_damage_model(self.engine_overrides)
        # Per-structure max-HP overrides (capture-difficulty lever). Resolved
        # from engine_overrides and overlaid onto the grid built above; absent
        # keys keep constants.py defaults. Snapshotted into config.json via the
        # verbatim engine_overrides log, same as damage_model / economy.
        self.structure_health: Dict[str, int] = self._resolve_structure_health(self.engine_overrides)
        self._apply_structure_health_overrides()
        # Hard ceiling on units-per-player (action-space + economy guardrail).
        # Enforced in both create_unit and get_legal_actions so the cap shows
        # up in the action mask, not just as a rejected action.
        self.max_units_per_player: int = self._resolve_max_units_per_player(self.engine_overrides)
        self.player_gold: Dict[int, int] = {i: self.starting_gold for i in range(1, num_players + 1)}
        self.game_over: bool = False
        self.winner: Optional[int] = None
        # Why the game ended. Populated alongside ``game_over`` by
        # ``_set_game_over``. Values: ``hq_capture``, ``elimination``,
        # ``max_turns_draw``, ``resign``. Replays surface this so videos
        # and the load-game UI can explain *how* a game finished without
        # re-deriving it from actions[].
        self.end_reason: Optional[str] = None
        # Index into ``action_history`` of the action that flipped
        # ``game_over``. None until the game ends. Lets replay viewers
        # jump to the decisive moment and lets analysis count any
        # post-victory actions a bot may have queued.
        self.game_over_action_index: Optional[int] = None
        self.turn_number: int = 0
        self.mechanics = GameMechanics()

        # Fog of war settings
        self.fog_of_war: bool = fog_of_war
        self.fog_of_war_method: str = "simple_radius" if fog_of_war else "none"
        self.visibility_maps: Dict[int, VisibilityMap] = {}
        if fog_of_war:
            for player in range(1, num_players + 1):
                self.visibility_maps[player] = VisibilityMap(self.grid.width, self.grid.height, player)

        # Enabled unit types (defaults to all if not specified)
        self.enabled_units: List[str] = enabled_units if enabled_units is not None else self.ALL_UNIT_TYPES.copy()

        # Optional map file reference for saving
        self.map_file_used: Optional[str] = None

        # Original map dimensions (before padding)
        self.original_map_width: int = self.grid.width
        self.original_map_height: int = self.grid.height
        self.map_padding_offset_x: int = 0
        self.map_padding_offset_y: int = 0

        # Store initial map data for replays (as 2D list of tile codes)
        if isinstance(map_data, pd.DataFrame):
            self.initial_map_data: List[List[str]] = map_data.values.tolist()
        elif isinstance(map_data, np.ndarray):
            self.initial_map_data: List[List[str]] = map_data.tolist()
        else:
            self.initial_map_data: List[List[str]] = [list(row) for row in map_data]

        # Store original unpadded map data
        self.original_map_data: Optional[List[List[str]]] = None

        # Player configurations (human vs bot)
        self.player_configs: List[Dict[str, Any]] = []

        # Maximum turns for the game (None = unlimited)
        self.max_turns: Optional[int] = max_turns

        # Action history for replay
        self.action_history: List[Dict[str, Any]] = []
        self.game_start_time: datetime = datetime.now()

        # Cached values for performance (separate validity flags to prevent stale cross-reads)
        self._unit_count_cache: Dict[int, int] = {}
        self._unit_count_cache_valid: bool = False
        self._legal_actions_cache: Dict[int, Dict[str, List[Any]]] = {}
        self._legal_actions_cache_valid: bool = False

    def reset(self, map_data) -> None:
        """Reset the game state."""
        self.__init__(
            map_data,
            self.num_players,
            self.max_turns,
            self.enabled_units,
            self.fog_of_war,
            engine_overrides=self.engine_overrides,
        )

    def set_map_metadata(
        self,
        original_width: int,
        original_height: int,
        padding_offset_x: int,
        padding_offset_y: int,
        map_file: Optional[str] = None,
        original_map_data: Optional[List[List[str]]] = None,
    ) -> None:
        """Set metadata about the original map before padding."""
        self.original_map_width = original_width
        self.original_map_height = original_height
        self.map_padding_offset_x = padding_offset_x
        self.map_padding_offset_y = padding_offset_y
        if map_file:
            self.map_file_used = map_file
        if original_map_data:
            self.original_map_data = original_map_data

    def padded_to_original_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert padded map coordinates to original map coordinates."""
        return (x - self.map_padding_offset_x, y - self.map_padding_offset_y)

    def original_to_padded_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert original map coordinates to padded map coordinates."""
        return (x + self.map_padding_offset_x, y + self.map_padding_offset_y)

    def _invalidate_cache(self) -> None:
        """Invalidate cached values."""
        self._unit_count_cache_valid = False
        self._unit_count_cache.clear()
        self._legal_actions_cache_valid = False
        self._legal_actions_cache.clear()

    def _set_game_over(self, winner: Optional[int], end_reason: str) -> None:
        """Single chokepoint for flipping ``game_over``.

        Records the winner, end reason, and the index of the action that
        caused the game to end (or -1 if no action was recorded yet,
        e.g. a max-turns draw that triggers before any new action is
        appended). Idempotent: first call wins; later attempts to set a
        different reason are ignored so a post-game action can't
        overwrite the real cause.
        """
        if self.game_over:
            return
        self.game_over = True
        self.winner = winner
        self.end_reason = end_reason
        self.game_over_action_index = len(self.action_history) - 1 if self.action_history else -1

    def _check_player_eliminated(self, defeated_player: int) -> None:
        """Check if a player has been eliminated and determine winner if appropriate.

        For 2-player games, the other player wins immediately.
        For 3+ player games, a winner is only declared when exactly one player remains.
        """
        remaining_units = [u for u in self.units if u.player == defeated_player]
        if len(remaining_units) == 0:
            if self.num_players == 2:
                self._set_game_over(winner=2 if defeated_player == 1 else 1, end_reason="elimination")
            else:
                active_players = set(u.player for u in self.units)
                if len(active_players) == 1:
                    self._set_game_over(winner=active_players.pop(), end_reason="elimination")

    def update_visibility(self, player: Optional[int] = None) -> None:
        """Update visibility maps for fog of war."""
        if not self.fog_of_war:
            return

        if player is not None:
            if player in self.visibility_maps:
                self.visibility_maps[player].update(self)
                self.visibility_maps[player].clear_stale_unit_memory(max_turns=10, current_turn=self.turn_number)
        else:
            for vis_map in self.visibility_maps.values():
                vis_map.update(self)
                vis_map.clear_stale_unit_memory(max_turns=10, current_turn=self.turn_number)

    def get_visible_units_for_player(self, player: int, include_own: bool = True) -> List[Unit]:
        """Get units visible to a specific player."""
        return get_visible_units(self, player, include_own)

    def is_position_visible(self, x: int, y: int, player: int) -> bool:
        """Check if a position is visible to a player."""
        if not self.fog_of_war:
            return True

        vis_map = self.visibility_maps.get(player)
        if vis_map is None:
            return True

        return vis_map.is_visible(x, y)

    def is_position_explored(self, x: int, y: int, player: int) -> bool:
        """Check if a position has been explored by a player."""
        if not self.fog_of_war:
            return True

        vis_map = self.visibility_maps.get(player)
        if vis_map is None:
            return True

        return vis_map.is_explored(x, y)

    def capture_visible_enemies_for_unit(self, unit: Unit) -> None:
        """Capture which enemy units are currently visible to a unit's owner."""
        if not self.fog_of_war:
            unit.visible_enemies_at_action_start = None
            return

        visible_positions = set()
        for enemy in self.units:
            if enemy.player != unit.player:
                if self.is_position_visible(enemy.x, enemy.y, unit.player):
                    visible_positions.add((enemy.x, enemy.y))

        unit.visible_enemies_at_action_start = visible_positions

    def is_enemy_attackable_by_unit(self, unit: Unit, enemy: Unit) -> bool:
        """Check if an enemy is attackable by a unit considering FOW pre-move snapshot."""
        if not self.fog_of_war:
            return True

        if unit.visible_enemies_at_action_start is None:
            return self.is_position_visible(enemy.x, enemy.y, unit.player)

        return (enemy.x, enemy.y) in unit.visible_enemies_at_action_start

    def is_unit_type_enabled(self, unit_type: str) -> bool:
        """Check if a unit type is enabled for this game."""
        return unit_type in self.enabled_units

    def set_enabled_units(self, enabled_units: List[str]) -> None:
        """Set the list of enabled unit types."""
        self.enabled_units = enabled_units
        self._invalidate_cache()

    def get_unit_count(self, player: int) -> int:
        """Get cached unit count for a player."""
        if not self._unit_count_cache_valid:
            self._unit_count_cache = {}
            for unit in self.units:
                self._unit_count_cache[unit.player] = self._unit_count_cache.get(unit.player, 0) + 1
            self._unit_count_cache_valid = True
        return self._unit_count_cache.get(player, 0)

    def get_unit_at_position(self, x: int, y: int) -> Optional[Unit]:
        """Get the unit at a grid position."""
        for unit in self.units:
            if unit.x == x and unit.y == y:
                return unit
        return None

    def record_action(self, action_type: str, **kwargs) -> None:
        """
        Record an action for replay purposes.

        Automatically converts any coordinate parameters from padded to original coordinates.

        Args:
            action_type: Type of action (move, attack, create_unit, etc.)
            **kwargs: Action-specific parameters (coordinates will be converted)
        """
        # Don't log anything once the game has been decided. Without this,
        # bots that don't break their per-unit loop on game_over append
        # cosmetic moves (and end_turn) after the winning action, which
        # makes len(actions) > winning_action_index + 1 and inflates
        # turn_number past the real end of the game. The single-chokepoint
        # guard here covers all 12 record_action sites in one place.
        if self.game_over:
            return

        # Convert coordinate parameters from padded to original
        converted_kwargs = {}
        for key, value in kwargs.items():
            if key in ["x", "y", "from_x", "from_y", "to_x", "to_y"]:
                if key.endswith("_x"):
                    converted_kwargs[key] = value
                elif key.endswith("_y"):
                    x_key = key.replace("_y", "_x")
                    if x_key in kwargs:
                        orig_x, orig_y = self.padded_to_original_coords(kwargs[x_key], value)
                        converted_kwargs[x_key] = orig_x
                        converted_kwargs[key] = orig_y
                    else:
                        converted_kwargs[key] = value
                elif key == "x":
                    converted_kwargs[key] = value
                elif key == "y":
                    if "x" in kwargs:
                        orig_x, orig_y = self.padded_to_original_coords(kwargs["x"], value)
                        converted_kwargs["x"] = orig_x
                        converted_kwargs[key] = orig_y
                    else:
                        converted_kwargs[key] = value
            elif key in ["position", "attacker_pos", "target_pos", "healer_pos", "paralyzer_pos", "curer_pos"]:
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    orig_x, orig_y = self.padded_to_original_coords(value[0], value[1])
                    converted_kwargs[key] = (orig_x, orig_y)
                else:
                    converted_kwargs[key] = value
            else:
                converted_kwargs[key] = value

        action_record = {
            "turn": self.turn_number,
            "player": self.current_player,
            "type": action_type,
            "timestamp": datetime.now().isoformat(),
            **converted_kwargs,
        }
        self.action_history.append(action_record)

    def create_unit(self, unit_type: str, x: int, y: int, player: Optional[int] = None) -> Optional[Unit]:
        """Create a unit at the specified position."""
        if player is None:
            player = self.current_player

        # Enforce the per-player unit cap. Mirrored in get_legal_actions so
        # the RL action mask hides create_unit at the cap rather than the
        # agent issuing a rejected action and eating the invalid_action
        # penalty.
        if sum(1 for u in self.units if u.player == player) >= self.max_units_per_player:
            logger.debug(f"Cannot create unit: player {player} at unit cap ({self.max_units_per_player})")
            return None

        # Check if position is occupied
        if self.get_unit_at_position(x, y):
            return None

        # Check if player can afford
        if unit_type not in self.unit_data:
            logger.warning(f"Unknown unit type: {unit_type}")
            return None

        cost = self.unit_data[unit_type]["cost"]
        if self.player_gold[player] < cost:
            return None

        self.player_gold[player] -= cost
        unit = Unit(unit_type, x, y, player, stats=self.unit_data[unit_type])
        unit.unit_id = self._next_unit_id
        self._next_unit_id += 1
        self.units.append(unit)
        self._invalidate_cache()

        # Record action. unit_id lets the replay player rebuild its
        # id -> Unit map on the fly (v3 schema), so subsequent
        # actions can find this unit even after it moves.
        self.record_action("create_unit", unit_type=unit_type, x=x, y=y, player=player, unit_id=unit.unit_id)

        logger.debug(f"Player {player} created {unit_type} at ({x}, {y})")
        return unit

    def move_unit(self, unit: Unit, to_x: int, to_y: int) -> bool:
        """Move a unit to a new position."""
        from_x, from_y = unit.x, unit.y

        # Reject duplicate moves: bot/RL/LLM call sites don't all gate on
        # ``unit.can_move`` before calling, and ``get_reachable_positions``
        # ignores it too, so without this check a unit could be moved more
        # than once per turn (producing duplicate "move" events in replays
        # and illegal positioning in-game).
        if not unit.can_move:
            logger.debug(f"Cannot move {unit.type} at ({unit.x}, {unit.y}): can_move is False")
            return False

        # Stale-reference guard: bots iterate over their own units once
        # per turn, but a unit can die mid-loop from a counter-attack on
        # an earlier action. The bot still holds the Python reference and
        # will keep calling APIs on the dead unit; without this guard
        # the engine moves it, logs the event, and the replay player
        # (which only sees self.units) can't reproduce it -- the action
        # silently no-ops and state diverges. See PR #360 audit.
        if unit not in self.units:
            return False

        # Check if move is valid
        reachable = unit.get_reachable_positions(
            self.grid.width,
            self.grid.height,
            lambda x, y: self.mechanics.can_move_to_position(
                x, y, self.grid, self.units, moving_unit=unit, is_destination=False
            ),
        )

        if (to_x, to_y) not in reachable:
            return False

        if not self.mechanics.can_move_to_position(to_x, to_y, self.grid, self.units, moving_unit=unit, is_destination=True):
            return False

        # FOW: Snapshot pre-move enemy visibility so the unit cannot attack
        # enemies it discovers by moving. The UI's input_handler captures this
        # at unit-selection time; for RL/LLM/bot code paths that drive
        # move_unit directly, capture lazily here just before the move.
        if self.fog_of_war and unit.visible_enemies_at_action_start is None:
            self.capture_visible_enemies_for_unit(unit)

        # Execute move
        unit.move_to(to_x, to_y)
        unit.can_move = False  # Consume move action

        # Record action
        self.record_action(
            "move",
            unit_type=unit.type,
            from_x=from_x,
            from_y=from_y,
            to_x=to_x,
            to_y=to_y,
            player=unit.player,
            actor_unit_id=unit.unit_id,
        )

        self.record_action("move", unit_type=unit.type, from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y, player=unit.player)
        self._invalidate_cache()
        self.update_visibility(unit.player)

        return True

    def attack(self, attacker: Unit, target: Unit) -> Dict[str, Any]:
        """
        Execute an attack.

        Args:
            attacker: Attacking unit
            target: Target unit

        Returns:
            dict: Attack results
        """
        # Stale-reference guard (see move_unit for full rationale).
        # Returns the same shape as a clean no-op attack so callers
        # that index into the result dict don't KeyError.
        if attacker not in self.units or target not in self.units:
            return {
                "attacker_alive": True,
                "target_alive": True,
                "damage": 0,
                "counter_damage": 0,
                "charge_bonus": False,
                "flank_bonus": False,
                "evade": False,
                "attack_buff": False,
                "defence_buff": False,
            }

        result = self.mechanics.attack_unit(attacker, target, self.grid, self.units, damage_model=self.damage_model)

        # Record action. The extra fields (attacker_killed, counter_damage,
        # *_hp_after, evade, *_bonus) are what makes the replay
        # self-describing -- without them the replay player would have
        # to re-call mechanics.attack_unit, which re-rolls Rogue evade
        # RNG (mechanics.py: ``random.random() < evade_chance``) and
        # recomputes damage against the replay's potentially-diverged
        # unit HP. Recording the outcome lets replay apply it directly.
        self.record_action(
            "attack",
            attacker_type=attacker.type,
            attacker_pos=(attacker.x, attacker.y),
            target_type=target.type,
            target_pos=(target.x, target.y),
            damage=result["damage"],
            target_killed=not result["target_alive"],
            attacker_killed=not result["attacker_alive"],
            counter_damage=result["counter_damage"],
            attacker_hp_after=attacker.health if result["attacker_alive"] else 0,
            target_hp_after=target.health if result["target_alive"] else 0,
            evade=result["evade"],
            charge_bonus=result["charge_bonus"],
            flank_bonus=result["flank_bonus"],
            attack_buff=result["attack_buff"],
            defence_buff=result["defence_buff"],
            player=attacker.player,
            attacker_unit_id=attacker.unit_id,
            target_unit_id=target.unit_id,
        )

        if not result["target_alive"]:
            target_tile = self.grid.get_tile(target.x, target.y)
            if target_tile.is_capturable() and target_tile.health < target_tile.max_health:
                target_tile.regenerating = True
            defeated_player = target.player
            self.units.remove(target)
            self._invalidate_cache()
            self._check_player_eliminated(defeated_player)

        if not result["attacker_alive"]:
            attacker_tile = self.grid.get_tile(attacker.x, attacker.y)
            if attacker_tile.is_capturable() and attacker_tile.health < attacker_tile.max_health:
                attacker_tile.regenerating = True
            defeated_player = attacker.player
            if attacker in self.units:
                self.units.remove(attacker)
            self._invalidate_cache()
            self._check_player_eliminated(defeated_player)

        if result["attacker_alive"]:
            attacker.can_move = False
            attacker.can_attack = False
        self._invalidate_cache()

        return result

    def paralyze(self, paralyzer: Unit, target: Unit) -> bool:
        """Paralyze a target unit."""
        if paralyzer not in self.units or target not in self.units:
            return False
        result = self.mechanics.paralyze_unit(paralyzer, target)
        if result:
            paralyzer.can_move = False
            paralyzer.can_attack = False
            self.record_action(
                "paralyze",
                paralyzer_pos=(paralyzer.x, paralyzer.y),
                target_pos=(target.x, target.y),
                player=paralyzer.player,
                actor_unit_id=paralyzer.unit_id,
                target_unit_id=target.unit_id,
            )
            self._invalidate_cache()
        return result

    def heal(self, healer: Unit, target: Unit) -> int:
        """Heal a target unit."""
        if healer not in self.units or target not in self.units:
            return 0
        amount = self.mechanics.heal_unit(healer, target)
        if amount > 0:
            healer.can_move = False
            healer.can_attack = False
            # target_hp_after lets the replay player set HP directly
            # instead of re-calling mechanics.heal_unit (the only path
            # today that could observe HEAL_AMOUNT drift between save
            # and replay).
            self.record_action(
                "heal",
                healer_pos=(healer.x, healer.y),
                target_pos=(target.x, target.y),
                amount=amount,
                target_hp_after=target.health,
                player=healer.player,
                actor_unit_id=healer.unit_id,
                target_unit_id=target.unit_id,
            )
            self._invalidate_cache()
        return amount

    def cure(self, curer: Unit, target: Unit) -> bool:
        """Cure a target unit's paralysis."""
        if curer not in self.units or target not in self.units:
            return False
        result = self.mechanics.cure_unit(curer, target)
        if result:
            curer.can_move = False
            curer.can_attack = False
            self.record_action(
                "cure",
                curer_pos=(curer.x, curer.y),
                target_pos=(target.x, target.y),
                player=curer.player,
                actor_unit_id=curer.unit_id,
                target_unit_id=target.unit_id,
            )
            self._invalidate_cache()
        return result

    def haste(self, sorcerer: Unit, target: Unit) -> bool:
        """
        Sorcerer grants Haste to a target unit.

        Args:
            sorcerer: The Sorcerer unit using Haste
            target: The target friendly unit

        Returns:
            bool: True if Haste was successfully applied
        """
        if sorcerer not in self.units or target not in self.units:
            return False
        result = self.mechanics.haste_unit(sorcerer, target)
        if result:
            sorcerer.can_move = False
            sorcerer.can_attack = False
            self.record_action(
                "haste",
                sorcerer_pos=(sorcerer.x, sorcerer.y),
                target_pos=(target.x, target.y),
                target_type=target.type,
                player=sorcerer.player,
                actor_unit_id=sorcerer.unit_id,
                target_unit_id=target.unit_id,
            )
            self._invalidate_cache()
        return result

    def defence_buff(self, sorcerer: Unit, target: Unit) -> bool:
        """
        Sorcerer grants Defence Buff to a target unit.

        Args:
            sorcerer: The Sorcerer unit using Defence Buff
            target: The target friendly unit

        Returns:
            bool: True if Defence Buff was successfully applied
        """
        if sorcerer not in self.units or target not in self.units:
            return False
        result = self.mechanics.defence_buff_unit(sorcerer, target)
        if result:
            sorcerer.can_move = False
            sorcerer.can_attack = False
            self.record_action(
                "defence_buff",
                sorcerer_pos=(sorcerer.x, sorcerer.y),
                target_pos=(target.x, target.y),
                target_type=target.type,
                player=sorcerer.player,
                actor_unit_id=sorcerer.unit_id,
                target_unit_id=target.unit_id,
            )
            self._invalidate_cache()
        return result

    def attack_buff(self, sorcerer: Unit, target: Unit) -> bool:
        """
        Sorcerer grants Attack Buff to a target unit.

        Args:
            sorcerer: The Sorcerer unit using Attack Buff
            target: The target friendly unit

        Returns:
            bool: True if Attack Buff was successfully applied
        """
        if sorcerer not in self.units or target not in self.units:
            return False
        result = self.mechanics.attack_buff_unit(sorcerer, target)
        if result:
            sorcerer.can_move = False
            sorcerer.can_attack = False
            self.record_action(
                "attack_buff",
                sorcerer_pos=(sorcerer.x, sorcerer.y),
                target_pos=(target.x, target.y),
                target_type=target.type,
                player=sorcerer.player,
                actor_unit_id=sorcerer.unit_id,
                target_unit_id=target.unit_id,
            )
            self._invalidate_cache()
        return result

    def seize(self, unit: Unit) -> Dict[str, Any]:
        """Seize the structure the unit is on."""
        if unit not in self.units:
            tile = self.grid.get_tile(unit.x, unit.y)
            return {"captured": False, "game_over": False, "structure_type": tile.type}
        tile = self.grid.get_tile(unit.x, unit.y)
        # Seize consumes the unit's turn action, so block repeats and
        # block units that have already acted (or were just spawned, since
        # spawn defaults can_attack=False). Without this a unit dropped on
        # an enemy HQ could issue 5 consecutive seize actions in one turn
        # and instantly win.
        if not unit.can_attack:
            return {"captured": False, "game_over": False, "structure_type": tile.type}
        result = self.mechanics.seize_structure(unit, tile)

        # Record action. tile_hp_after / tile_owner_after let the v2
        # replay player set tile state directly instead of re-calling
        # mechanics.seize_structure -- which would decrement by
        # unit.health and only match the original if every prior
        # action's HP-mutation reproduced exactly. Without this,
        # any replay that starts from a partially-damaged tile (e.g.
        # a unit-test that pokes tile.health) desyncs immediately.
        # mechanics.seize_structure also clears tile.regenerating on
        # any successful call, so we don't need to record it separately.
        self.record_action(
            "seize",
            unit_type=unit.type,
            position=(unit.x, unit.y),
            structure_type=tile.type,
            captured=result["captured"],
            tile_hp_after=tile.health,
            tile_owner_after=tile.player,
            player=unit.player,
            actor_unit_id=unit.unit_id,
        )

        if result["game_over"]:
            self._set_game_over(winner=unit.player, end_reason="hq_capture")

        unit.can_move = False
        unit.can_attack = False
        self._invalidate_cache()

        return result

    def heal_units_on_structures(self, player: int) -> Dict[str, Any]:
        """Heal units on owned structures at the start of their turn."""
        stats: Dict[str, Any] = {"total_healed": 0, "total_cost": 0, "units_healed": []}

        enemy_hq_pos = None
        for row in self.grid.tiles:
            for tile in row:
                if tile.type == TileType.HEADQUARTERS.value and tile.player and tile.player != player:
                    enemy_hq_pos = (tile.x, tile.y)
                    break
            if enemy_hq_pos:
                break

        units_to_heal = []

        for unit in self.units:
            if unit.player != player:
                continue
            if unit.health >= unit.max_health:
                continue

            tile = self.grid.get_tile(unit.x, unit.y)
            if not tile or tile.player != player:
                continue

            heal_amount = 0
            structure_name = ""

            if tile.type == TileType.TOWER.value:
                heal_amount = 1
                structure_name = "Tower"
            elif tile.type == TileType.HEADQUARTERS.value:
                heal_amount = 2
                structure_name = "Headquarters"
            elif tile.type == TileType.BUILDING.value:
                heal_amount = 2
                structure_name = "Building"

            if heal_amount > 0:
                distance = float("inf")
                if enemy_hq_pos:
                    distance = abs(unit.x - enemy_hq_pos[0]) + abs(unit.y - enemy_hq_pos[1])

                units_to_heal.append(
                    {"unit": unit, "heal_amount": heal_amount, "structure_name": structure_name, "distance": distance}
                )

        units_to_heal.sort(key=lambda x: x["distance"])

        for heal_data in units_to_heal:
            unit = heal_data["unit"]
            requested_heal = heal_data["heal_amount"]
            structure_name = heal_data["structure_name"]

            max_possible_heal = unit.max_health - unit.health
            desired_heal = min(requested_heal, max_possible_heal)

            unit_cost = self.unit_data[unit.type]["cost"]
            cost_per_hp = unit_cost / unit.max_health

            actual_heal = 0
            actual_cost = 0

            if structure_name == "Tower":
                total_cost = round(cost_per_hp * desired_heal)
                if self.player_gold[player] >= total_cost:
                    actual_heal = desired_heal
                    actual_cost = total_cost
            else:  # HQ or Building - allow partial healing
                for hp in range(desired_heal, 0, -1):
                    cost = round(cost_per_hp * hp)
                    if self.player_gold[player] >= cost:
                        actual_heal = hp
                        actual_cost = cost
                        break

            if actual_heal > 0:
                old_health = unit.health
                unit.health = min(unit.health + actual_heal, unit.max_health)
                self.player_gold[player] -= actual_cost

                stats["total_healed"] += actual_heal
                stats["total_cost"] += actual_cost
                stats["units_healed"].append(
                    {
                        "unit_type": unit.type,
                        "position": (unit.x, unit.y),
                        "structure": structure_name,
                        "healed": actual_heal,
                        "cost": actual_cost,
                        "old_health": old_health,
                        "new_health": unit.health,
                    }
                )

        return stats

    def end_turn(self) -> Dict[str, Any]:
        """End the current player's turn and pass to the next player."""
        # No-op once the game has ended. Prevents turn_number from being
        # bumped past the winning turn (which would otherwise make
        # game_info.turns disagree with max(action.turn) in the replay)
        # and avoids running structure regen / paralysis decrement /
        # max_turns checks on a finished game.
        if self.game_over:
            return {"total": 0, "healing": {"total_healed": 0, "total_cost": 0, "units_healed": []}}

        # Record action
        self.record_action("end_turn", player=self.current_player)

        for unit in self.units:
            if unit.player == self.current_player and unit.has_moved:
                old_tile = self.grid.get_tile(unit.original_x, unit.original_y)
                if (unit.x, unit.y) != (unit.original_x, unit.original_y):
                    self.mechanics.reset_structure_if_vacated(old_tile, self.units)

        self.mechanics.regenerate_structures(self.grid, self.units)

        self.current_player += 1
        if self.current_player > self.num_players:
            self.current_player = 1
            self.turn_number += 1

            # Once all players have moved, check the turn limit (draw on cap).
            if self.max_turns is not None and self.turn_number >= self.max_turns:
                self._set_game_over(winner=None, end_reason="max_turns_draw")
                return {"total": 0, "healing": {"total_healed": 0, "total_cost": 0, "units_healed": []}}

        self.mechanics.decrement_paralysis(self.units, self.current_player)
        self.mechanics.decrement_paralyze_cooldowns(self.units, self.current_player)
        self.mechanics.decrement_haste_cooldowns(self.units, self.current_player)
        self.mechanics.decrement_buff_cooldowns(self.units, self.current_player)
        self.mechanics.decrement_buff_durations(self.units, self.current_player)

        for unit in self.units:
            if unit.player == self.current_player:
                if not unit.is_paralyzed():
                    unit.can_move = True
                    unit.can_attack = True
                else:
                    unit.can_move = False
                    unit.can_attack = False

                unit.original_x = unit.x
                unit.original_y = unit.y
                unit.has_moved = False
                unit.distance_moved = 0
                unit.is_hasted = False
                # FOW: Clear stale snapshot so it gets recaptured before
                # this unit's next move (see move_unit lazy capture).
                unit.visible_enemies_at_action_start = None
            unit.selected = False

        # Calculate and apply income
        income_data = self.mechanics.calculate_income(self.current_player, self.grid, self.income_rates)
        self.player_gold[self.current_player] += income_data["total"]

        healing_stats = self.heal_units_on_structures(self.current_player)
        income_data["healing"] = healing_stats

        self.update_visibility(self.current_player)

        return income_data

    def resign(self, player: Optional[int] = None) -> None:
        """Player resigns."""
        if player is None:
            player = self.current_player

        self.record_action("resign", player=player)

        # Remove the resigning player's units (matches the upstream engine).
        self.units = [u for u in self.units if u.player != player]
        self._invalidate_cache()

        if self.num_players == 2:
            self._set_game_over(winner=2 if player == 1 else 1, end_reason="resign")
        else:
            active_players = set(u.player for u in self.units)
            if len(active_players) <= 1:
                self._set_game_over(
                    winner=active_players.pop() if active_players else None,
                    end_reason="resign",
                )

    def get_legal_actions(self, player: Optional[int] = None) -> Dict[str, List[Any]]:
        """Get all legal actions for the current player."""
        if player is None:
            player = self.current_player

        if self._legal_actions_cache_valid and player in self._legal_actions_cache:
            return self._legal_actions_cache[player]

        legal_actions: Dict[str, Any] = {
            "create_unit": [],
            "move": [],
            "attack": [],
            "paralyze": [],
            "heal": [],
            "cure": [],
            "haste": [],
            "defence_buff": [],
            "attack_buff": [],
            "seize": [],
            "end_turn": True,
        }

        # Building units (only at Buildings, not HQ)
        # Only include enabled unit types. Suppressed entirely once the player
        # is at the unit cap so the action mask matches create_unit's own
        # enforcement (no offered-then-rejected create actions).
        if sum(1 for u in self.units if u.player == player) < self.max_units_per_player:
            for tile in self.grid.get_capturable_tiles(player):
                if tile.type == TileType.BUILDING.value and not self.get_unit_at_position(tile.x, tile.y):
                    for unit_type in self.enabled_units:
                        if self.player_gold[player] >= self.unit_data[unit_type]["cost"]:
                            legal_actions["create_unit"].append({"unit_type": unit_type, "x": tile.x, "y": tile.y})

        for unit in self.units:
            # Guard on health: dead units are normally removed synchronously
            # by ``attack`` (see self.units.remove), but the helpers below all
            # filter on ``health > 0`` defensively -- mirror that here so a
            # corpse left in ``self.units`` by any future deferred-removal path
            # (AoE, end-of-turn DoT, status damage) can't emit phantom actions.
            if unit.player == player and unit.health > 0 and not unit.is_paralyzed():
                # Movement
                if unit.can_move:
                    reachable = unit.get_reachable_positions(
                        self.grid.width,
                        self.grid.height,
                        lambda x, y, _u=unit: self.mechanics.can_move_to_position(x, y, self.grid, self.units, moving_unit=_u),
                    )
                    for pos in reachable:
                        # Only include positions valid as final destinations (not occupied).
                        if self.mechanics.can_move_to_position(
                            pos[0], pos[1], self.grid, self.units, moving_unit=unit, is_destination=True
                        ):
                            legal_actions["move"].append(
                                {"unit": unit, "from_x": unit.x, "from_y": unit.y, "to_x": pos[0], "to_y": pos[1]}
                            )

                if unit.can_attack:
                    if unit.type in ["M", "A", "S"]:
                        unit_tile = self.grid.get_tile(unit.x, unit.y)
                        on_mountain = unit_tile.type == "m"

                        for enemy in self.units:
                            if enemy.player != player and enemy.health > 0:
                                # FOW: Skip enemies not attackable (checks pre-move snapshot)
                                if self.fog_of_war and not self.is_enemy_attackable_by_unit(unit, enemy):
                                    continue

                                damage = unit.get_attack_damage(enemy.x, enemy.y, on_mountain)
                                if damage > 0:
                                    legal_actions["attack"].append({"attacker": unit, "target": enemy})

                                    # Paralyze: skip already-paralyzed targets.
                                    # Re-casting only refreshes the status (a
                                    # near no-op) and inflates the action space,
                                    # unlike heal/cure/buffs which all guard
                                    # against re-applying to an already-affected
                                    # ally.
                                    if unit.type == "M" and unit.can_use_paralyze() and not enemy.is_paralyzed():
                                        # Mages can also paralyze at range (if not on cooldown)
                                        distance = abs(unit.x - enemy.x) + abs(unit.y - enemy.y)
                                        if distance <= 2:
                                            legal_actions["paralyze"].append({"paralyzer": unit, "target": enemy})
                    else:
                        adjacent_enemies = self.mechanics.get_adjacent_enemies(unit, self.units)
                        for enemy in adjacent_enemies:
                            if self.fog_of_war and not self.is_enemy_attackable_by_unit(unit, enemy):
                                continue

                            legal_actions["attack"].append({"attacker": unit, "target": enemy})

                    if unit.type == "C":
                        healable_allies = self.mechanics.get_healable_allies(unit, self.units)
                        for ally in healable_allies:
                            legal_actions["heal"].append({"healer": unit, "target": ally})

                        curable_allies = self.mechanics.get_curable_allies(unit, self.units)
                        for ally in curable_allies:
                            legal_actions["cure"].append({"curer": unit, "target": ally})

                    if unit.type == "S" and unit.can_use_haste():
                        hasteable_allies = self.mechanics.get_hasteable_allies(unit, self.units)
                        for ally in hasteable_allies:
                            legal_actions["haste"].append({"sorcerer": unit, "target": ally})

                    if unit.type == "S" and unit.can_use_defence_buff():
                        buffable_allies = self.mechanics.get_defence_buffable_allies(unit, self.units)
                        for ally in buffable_allies:
                            legal_actions["defence_buff"].append({"sorcerer": unit, "target": ally})

                    if unit.type == "S" and unit.can_use_attack_buff():
                        buffable_allies = self.mechanics.get_attack_buffable_allies(unit, self.units)
                        for ally in buffable_allies:
                            legal_actions["attack_buff"].append({"sorcerer": unit, "target": ally})

                    if unit.can_attack:
                        tile = self.grid.get_tile(unit.x, unit.y)
                        if tile.is_capturable() and tile.player != player:
                            legal_actions["seize"].append({"unit": unit, "tile": tile})

        if self._legal_actions_cache_valid:
            self._legal_actions_cache[player] = legal_actions

        return legal_actions

    def to_dict(self) -> Dict[str, Any]:
        """Convert game state to dictionary for serialization."""
        return {
            "timestamp": self.game_start_time.strftime("%Y-%m-%d %H-%M-%S"),
            "current_player": self.current_player,
            "num_players": self.num_players,
            "player_gold": self.player_gold,
            "turn_number": self.turn_number,
            "max_turns": self.max_turns,
            "game_over": self.game_over,
            "winner": self.winner,
            "map_file": self.map_file_used,
            "player_configs": self.player_configs,
            "enabled_units": self.enabled_units,
            "fog_of_war": self.fog_of_war,
            "fog_of_war_method": self.fog_of_war_method,
            # Persist the engine-constant overlay so a reloaded game runs under
            # the same balance (damage_model, structure HP, economy, unit cap)
            # it was saved under. Absent in pre-0.3.3 saves -> from_dict falls
            # back to {} (== module defaults), preserving backward-compat.
            "engine_overrides": self.engine_overrides,
            "units": [unit.to_dict() for unit in self.units],
            "tiles": self.grid.to_dict()["tiles"],
            "action_history": self.action_history,
            # Restore the per-game unit-id counter on reload so newly
            # created units after load don't reuse retired ids
            # (which would let the replay v3 dispatch route an action
            # to the wrong unit -- exactly the brittleness this whole
            # schema bump is meant to eliminate).
            "next_unit_id": self._next_unit_id,
        }

    def to_numpy(self, for_player: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Convert game state to numpy arrays for RL."""
        grid_state = self.grid.to_numpy()

        # Unit representation (height x width x 8)
        #   [..., 0] = unit_type int (0 = empty, 1..8 = ALL_UNIT_TYPES)
        #   [..., 1] = absolute owner (0 = empty cell, else player number)
        #   [..., 2] = unit HP percentage in [0, 100]
        #   [..., 3] = exhausted flag (1.0 if the unit has no actions left
        #             this turn, 0.0 otherwise). Defined as
        #             ``not (can_move or can_attack)`` so it captures every
        #             way a unit spends its turn -- moving, attacking,
        #             seizing, healing, or casting in place -- not just
        #             movement. (A unit that moved but can still attack reads
        #             0.0; a unit that attacked without moving reads 1.0.)
        #             Consumed by build_observation as a per-unit "exhausted"
        #             signal for the policy.
        #   [..., 4] = paralyzed_turns (0..PARALYZE_DURATION). Surfaces the
        #             Mage paralyze debuff so the policy can value attacking /
        #             defending paralyzed targets correctly.
        #   [..., 5] = is_hasted (0.0 / 1.0). Surfaces the Sorcerer haste
        #             buff (extra-action-this-turn) so the policy can see
        #             which units still have an action left.
        #   [..., 6] = defence_buff_turns (0..SORCERER_BUFF_DURATION).
        #             Surfaces the Sorcerer defence buff so the policy can
        #             account for the +50% damage reduction on the unit.
        #   [..., 7] = attack_buff_turns (0..SORCERER_BUFF_DURATION).
        #             Surfaces the Sorcerer attack buff so the policy can
        #             account for the +50% damage bonus on the unit.
        unit_state = np.zeros((self.grid.height, self.grid.width, 8), dtype=np.float32)

        # Encoding for all 8 unit types: W, M, C, A, K, R, S, B
        unit_type_encoding = {"W": 1, "M": 2, "C": 3, "A": 4, "K": 5, "R": 6, "S": 7, "B": 8}

        visibility_state = np.full((self.grid.height, self.grid.width), VISIBLE, dtype=np.uint8)

        if self.fog_of_war and for_player is not None:
            vis_map = self.visibility_maps.get(for_player)
            if vis_map is not None:
                visibility_state = vis_map.to_numpy()

        for unit in self.units:
            if self.fog_of_war and for_player is not None:
                if unit.player != for_player and visibility_state[unit.y, unit.x] != VISIBLE:
                    continue

            unit_state[unit.y, unit.x, 0] = unit_type_encoding.get(unit.type, 0)
            unit_state[unit.y, unit.x, 1] = unit.player
            unit_state[unit.y, unit.x, 2] = (unit.health / unit.max_health) * 100
            unit_state[unit.y, unit.x, 3] = 0.0 if (unit.can_move or unit.can_attack) else 1.0
            # Status effects (raw turn counts; observation.py normalises
            # by their respective max durations to land in [0, 1]).
            unit_state[unit.y, unit.x, 4] = float(getattr(unit, "paralyzed_turns", 0))
            unit_state[unit.y, unit.x, 5] = 1.0 if getattr(unit, "is_hasted", False) else 0.0
            unit_state[unit.y, unit.x, 6] = float(getattr(unit, "defence_buff_turns", 0))
            unit_state[unit.y, unit.x, 7] = float(getattr(unit, "attack_buff_turns", 0))

        if self.fog_of_war and for_player is not None:
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    if visibility_state[y, x] != VISIBLE:
                        if visibility_state[y, x] == 0:  # UNEXPLORED
                            grid_state[y, x, 0] = 0
                            grid_state[y, x, 1] = 0
                            grid_state[y, x, 2] = 0

        result: Dict[str, Any] = {
            "grid": grid_state,
            "units": unit_state,
            "gold": np.array([self.player_gold[i] for i in range(1, self.num_players + 1)], dtype=np.float32),
            "current_player": self.current_player,
            "turn_number": self.turn_number,
        }

        if self.fog_of_war:
            result["visibility"] = visibility_state

        return result

    def save_to_file(self, filepath: Optional[str] = None) -> Optional[str]:
        """Save game state to file (not available in vendored Kaggle build)."""
        raise NotImplementedError(
            "save_to_file is not available in the vendored Kaggle engine. "
            "Use the full reinforcetactics package for save/load functionality."
        )

    def save_replay_to_file(self, filepath: Optional[str] = None) -> Optional[str]:
        """Save replay to file (not available in vendored Kaggle build)."""
        raise NotImplementedError(
            "save_replay_to_file is not available in the vendored Kaggle engine. "
            "Use the full reinforcetactics package for replay functionality."
        )

    @classmethod
    def from_dict(cls, save_data: Dict[str, Any], map_data) -> "GameState":
        """Restore game state from dictionary."""
        enabled_units = save_data.get("enabled_units", cls.ALL_UNIT_TYPES)
        fog_of_war = save_data.get("fog_of_war", False)
        fog_of_war_method = save_data.get("fog_of_war_method", "simple_radius" if fog_of_war else "none")

        max_turns = save_data.get("max_turns")
        # Restore the engine-constant overlay (damage_model / structure HP /
        # economy / unit cap). Absent in pre-0.3.3 saves -> {} == module
        # defaults, byte-identical to the old load behaviour.
        engine_overrides = save_data.get("engine_overrides") or {}
        game = cls(
            map_data,
            save_data.get("num_players", 2),
            max_turns=max_turns,
            enabled_units=enabled_units,
            fog_of_war=fog_of_war,
            engine_overrides=engine_overrides,
        )
        game.fog_of_war_method = fog_of_war_method

        game.current_player = save_data.get("current_player", 1)
        game.turn_number = save_data.get("turn_number", 0)
        game.game_over = save_data.get("game_over", False)
        game.winner = save_data.get("winner")
        game.end_reason = save_data.get("end_reason")
        game.game_over_action_index = save_data.get("winning_action_index")
        # Restore unit-id counter. Old saves predate this field; ``from_dict``
        # for the units themselves leaves ``unit.unit_id = None`` in that
        # case and ``find_unit_by_id`` falls back to position-based lookup.
        game._next_unit_id = save_data.get("next_unit_id", 0)

        saved_gold = save_data.get("player_gold", {})
        game.player_gold = {int(k): v for k, v in saved_gold.items()}

        game.map_file_used = save_data.get("map_file")
        game.player_configs = save_data.get("player_configs", [])
        game.action_history = save_data.get("action_history", [])

        game.units = []
        for unit_data in save_data.get("units", []):
            unit = Unit.from_dict(unit_data)
            game.units.append(unit)

        for tile_data in save_data.get("tiles", []):
            x, y = tile_data["x"], tile_data["y"]
            if 0 <= x < game.grid.width and 0 <= y < game.grid.height:
                tile = game.grid.tiles[y][x]
                if tile_data.get("player"):
                    tile.player = tile_data["player"]
                if tile_data.get("health") is not None:
                    tile.health = tile_data["health"]
                if tile_data.get("regenerating") is not None:
                    tile.regenerating = tile_data["regenerating"]

        game._invalidate_cache()
        return game
