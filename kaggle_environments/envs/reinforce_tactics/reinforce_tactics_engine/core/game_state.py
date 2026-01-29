"""
Core game state management without rendering dependencies.

Vendored copy for Kaggle Environments — save/replay features that depend
on ``reinforcetactics.utils.file_io`` are stubbed out.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from .unit import Unit
from .grid import TileGrid
from .visibility import VisibilityMap, get_visible_units, VISIBLE
from ..game.mechanics import GameMechanics
from ..constants import STARTING_GOLD, UNIT_DATA, TileType

# Configure logging
logger = logging.getLogger(__name__)


class GameState:
    """Manages the core game state without rendering."""

    # All available unit types
    ALL_UNIT_TYPES = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']

    def __init__(self, map_data, num_players: int = 2, max_turns: Optional[int] = None,
                 enabled_units: Optional[List[str]] = None,
                 fog_of_war: bool = False) -> None:
        """
        Initialize the game state.

        Args:
            map_data: 2D array containing map information
            num_players: Number of players (default 2)
            max_turns: Maximum turns for the game (None = unlimited)
            enabled_units: List of enabled unit types (default all units enabled)
            fog_of_war: Enable fog of war (default False for backward compatibility)
        """
        self.grid = TileGrid(map_data)
        self.units: List[Unit] = []
        self.current_player: int = 1
        self.num_players: int = num_players
        self.player_gold: Dict[int, int] = {i: STARTING_GOLD for i in range(1, num_players + 1)}
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.turn_number: int = 0
        self.mechanics = GameMechanics()

        # Fog of war settings
        self.fog_of_war: bool = fog_of_war
        self.fog_of_war_method: str = 'simple_radius' if fog_of_war else 'none'
        self.visibility_maps: Dict[int, VisibilityMap] = {}
        if fog_of_war:
            for player in range(1, num_players + 1):
                self.visibility_maps[player] = VisibilityMap(
                    self.grid.width, self.grid.height, player
                )

        # Enabled unit types (defaults to all if not specified)
        self.enabled_units: List[str] = (
            enabled_units if enabled_units is not None else self.ALL_UNIT_TYPES.copy()
        )

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

        # Cached values for performance
        self._unit_count_cache: Dict[int, int] = {}
        self._legal_actions_cache: Dict[int, Dict[str, List[Any]]] = {}
        self._cache_valid: bool = False

    def reset(self, map_data) -> None:
        """Reset the game state."""
        self.__init__(map_data, self.num_players, self.max_turns,
                      self.enabled_units, self.fog_of_war)

    def set_map_metadata(self, original_width: int, original_height: int,
                         padding_offset_x: int, padding_offset_y: int,
                         map_file: Optional[str] = None,
                         original_map_data: Optional[List[List[str]]] = None) -> None:
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
        self._cache_valid = False
        self._unit_count_cache.clear()
        self._legal_actions_cache.clear()

    def update_visibility(self, player: Optional[int] = None) -> None:
        """Update visibility maps for fog of war."""
        if not self.fog_of_war:
            return

        if player is not None:
            if player in self.visibility_maps:
                self.visibility_maps[player].update(self)
        else:
            for vis_map in self.visibility_maps.values():
                vis_map.update(self)

    def get_visible_units_for_player(self, player: int,
                                     include_own: bool = True) -> List[Unit]:
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
        if not self._cache_valid:
            self._unit_count_cache = {}
            for unit in self.units:
                self._unit_count_cache[unit.player] = (
                    self._unit_count_cache.get(unit.player, 0) + 1
                )
            self._cache_valid = True
        return self._unit_count_cache.get(player, 0)

    def get_unit_at_position(self, x: int, y: int) -> Optional[Unit]:
        """Get the unit at a grid position."""
        for unit in self.units:
            if unit.x == x and unit.y == y:
                return unit
        return None

    def record_action(self, action_type: str, **kwargs) -> None:
        """Record an action for replay purposes."""
        converted_kwargs = {}
        for key, value in kwargs.items():
            if key in ['x', 'y', 'from_x', 'from_y', 'to_x', 'to_y']:
                if key.endswith('_x'):
                    converted_kwargs[key] = value
                elif key.endswith('_y'):
                    x_key = key.replace('_y', '_x')
                    if x_key in kwargs:
                        orig_x, orig_y = self.padded_to_original_coords(
                            kwargs[x_key], value
                        )
                        converted_kwargs[x_key] = orig_x
                        converted_kwargs[key] = orig_y
                    else:
                        converted_kwargs[key] = value
                elif key == 'x':
                    converted_kwargs[key] = value
                elif key == 'y':
                    if 'x' in kwargs:
                        orig_x, orig_y = self.padded_to_original_coords(
                            kwargs['x'], value
                        )
                        converted_kwargs['x'] = orig_x
                        converted_kwargs[key] = orig_y
                    else:
                        converted_kwargs[key] = value
            elif key in ['position', 'attacker_pos', 'target_pos',
                         'healer_pos', 'paralyzer_pos', 'curer_pos']:
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    orig_x, orig_y = self.padded_to_original_coords(value[0], value[1])
                    converted_kwargs[key] = (orig_x, orig_y)
                else:
                    converted_kwargs[key] = value
            else:
                converted_kwargs[key] = value

        action_record = {
            'turn': self.turn_number,
            'player': self.current_player,
            'type': action_type,
            'timestamp': datetime.now().isoformat(),
            **converted_kwargs
        }
        self.action_history.append(action_record)

    def create_unit(self, unit_type: str, x: int, y: int,
                    player: Optional[int] = None) -> Optional[Unit]:
        """Create a unit at the specified position."""
        if player is None:
            player = self.current_player

        if self.get_unit_at_position(x, y):
            return None

        if unit_type not in UNIT_DATA:
            return None

        cost = UNIT_DATA[unit_type]['cost']
        if self.player_gold[player] < cost:
            return None

        self.player_gold[player] -= cost
        unit = Unit(unit_type, x, y, player)
        self.units.append(unit)
        self._invalidate_cache()

        self.record_action('create_unit', unit_type=unit_type, x=x, y=y, player=player)
        return unit

    def move_unit(self, unit: Unit, to_x: int, to_y: int) -> bool:
        """Move a unit to a new position."""
        from_x, from_y = unit.x, unit.y

        reachable = unit.get_reachable_positions(
            self.grid.width,
            self.grid.height,
            lambda x, y: self.mechanics.can_move_to_position(
                x, y, self.grid, self.units, moving_unit=unit, is_destination=False
            )
        )

        if (to_x, to_y) not in reachable:
            return False

        if not self.mechanics.can_move_to_position(
            to_x, to_y, self.grid, self.units, moving_unit=unit, is_destination=True
        ):
            return False

        unit.move_to(to_x, to_y)
        unit.can_move = False

        self.record_action('move', unit_type=unit.type, from_x=from_x, from_y=from_y,
                           to_x=to_x, to_y=to_y, player=unit.player)
        self._invalidate_cache()
        self.update_visibility(unit.player)

        return True

    def attack(self, attacker: Unit, target: Unit) -> Dict[str, Any]:
        """Execute an attack."""
        result = self.mechanics.attack_unit(attacker, target, self.grid, self.units)

        self.record_action('attack',
                           attacker_type=attacker.type,
                           attacker_pos=(attacker.x, attacker.y),
                           target_type=target.type,
                           target_pos=(target.x, target.y),
                           damage=result['damage'],
                           target_killed=not result['target_alive'],
                           player=attacker.player)

        if not result['target_alive']:
            target_tile = self.grid.get_tile(target.x, target.y)
            if target_tile.is_capturable() and target_tile.health < target_tile.max_health:
                target_tile.regenerating = True
            defeated_player = target.player
            self.units.remove(target)
            self._invalidate_cache()

            remaining_units = [u for u in self.units if u.player == defeated_player]
            if not remaining_units:
                self.game_over = True
                if self.num_players == 2:
                    self.winner = 2 if defeated_player == 1 else 1
                else:
                    self.winner = (
                        defeated_player + 1
                        if defeated_player < self.num_players
                        else 1
                    )

        if not result['attacker_alive']:
            attacker_tile = self.grid.get_tile(attacker.x, attacker.y)
            if attacker_tile.is_capturable() and attacker_tile.health < attacker_tile.max_health:
                attacker_tile.regenerating = True
            defeated_player = attacker.player
            self.units.remove(attacker)
            self._invalidate_cache()

            remaining_units = [u for u in self.units if u.player == defeated_player]
            if not remaining_units:
                self.game_over = True
                if self.num_players == 2:
                    self.winner = 2 if defeated_player == 1 else 1
                else:
                    self.winner = (
                        defeated_player + 1
                        if defeated_player < self.num_players
                        else 1
                    )

        attacker.can_move = False
        attacker.can_attack = False
        self._invalidate_cache()

        return result

    def paralyze(self, paralyzer: Unit, target: Unit) -> bool:
        """Paralyze a target unit."""
        result = self.mechanics.paralyze_unit(paralyzer, target)
        if result:
            paralyzer.can_move = False
            paralyzer.can_attack = False
            self.record_action('paralyze',
                               paralyzer_pos=(paralyzer.x, paralyzer.y),
                               target_pos=(target.x, target.y),
                               player=paralyzer.player)
            self._invalidate_cache()
        return result

    def heal(self, healer: Unit, target: Unit) -> int:
        """Heal a target unit."""
        amount = self.mechanics.heal_unit(healer, target)
        if amount > 0:
            healer.can_move = False
            healer.can_attack = False
            self.record_action('heal',
                               healer_pos=(healer.x, healer.y),
                               target_pos=(target.x, target.y),
                               amount=amount,
                               player=healer.player)
            self._invalidate_cache()
        return amount

    def cure(self, curer: Unit, target: Unit) -> bool:
        """Cure a target unit's paralysis."""
        result = self.mechanics.cure_unit(curer, target)
        if result:
            curer.can_move = False
            curer.can_attack = False
            self.record_action('cure',
                               curer_pos=(curer.x, curer.y),
                               target_pos=(target.x, target.y),
                               player=curer.player)
            self._invalidate_cache()
        return result

    def haste(self, sorcerer: Unit, target: Unit) -> bool:
        """Sorcerer grants Haste to a target unit."""
        result = self.mechanics.haste_unit(sorcerer, target)
        if result:
            sorcerer.can_move = False
            sorcerer.can_attack = False
            self.record_action('haste',
                               sorcerer_pos=(sorcerer.x, sorcerer.y),
                               target_pos=(target.x, target.y),
                               target_type=target.type,
                               player=sorcerer.player)
            self._invalidate_cache()
        return result

    def defence_buff(self, sorcerer: Unit, target: Unit) -> bool:
        """Sorcerer grants Defence Buff to a target unit."""
        result = self.mechanics.defence_buff_unit(sorcerer, target)
        if result:
            sorcerer.can_move = False
            sorcerer.can_attack = False
            self.record_action('defence_buff',
                               sorcerer_pos=(sorcerer.x, sorcerer.y),
                               target_pos=(target.x, target.y),
                               target_type=target.type,
                               player=sorcerer.player)
            self._invalidate_cache()
        return result

    def attack_buff(self, sorcerer: Unit, target: Unit) -> bool:
        """Sorcerer grants Attack Buff to a target unit."""
        result = self.mechanics.attack_buff_unit(sorcerer, target)
        if result:
            sorcerer.can_move = False
            sorcerer.can_attack = False
            self.record_action('attack_buff',
                               sorcerer_pos=(sorcerer.x, sorcerer.y),
                               target_pos=(target.x, target.y),
                               target_type=target.type,
                               player=sorcerer.player)
            self._invalidate_cache()
        return result

    def seize(self, unit: Unit) -> Dict[str, Any]:
        """Seize the structure the unit is on."""
        tile = self.grid.get_tile(unit.x, unit.y)
        result = self.mechanics.seize_structure(unit, tile)

        self.record_action('seize',
                           unit_type=unit.type,
                           position=(unit.x, unit.y),
                           structure_type=tile.type,
                           captured=result['captured'],
                           player=unit.player)

        if result['game_over']:
            self.game_over = True
            self.winner = unit.player

        unit.can_move = False
        unit.can_attack = False
        self._invalidate_cache()

        return result

    def heal_units_on_structures(self, player: int) -> Dict[str, Any]:
        """Heal units on owned structures at the start of their turn."""
        stats: Dict[str, Any] = {
            'total_healed': 0, 'total_cost': 0, 'units_healed': []
        }

        enemy_hq_pos = None
        for row in self.grid.tiles:
            for tile in row:
                if (tile.type == TileType.HEADQUARTERS.value
                        and tile.player and tile.player != player):
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
                distance = float('inf')
                if enemy_hq_pos:
                    distance = abs(unit.x - enemy_hq_pos[0]) + abs(unit.y - enemy_hq_pos[1])

                units_to_heal.append({
                    'unit': unit,
                    'heal_amount': heal_amount,
                    'structure_name': structure_name,
                    'distance': distance
                })

        units_to_heal.sort(key=lambda x: x['distance'])

        for heal_data in units_to_heal:
            unit = heal_data['unit']
            requested_heal = heal_data['heal_amount']
            structure_name = heal_data['structure_name']

            max_possible_heal = unit.max_health - unit.health
            desired_heal = min(requested_heal, max_possible_heal)

            unit_cost = UNIT_DATA[unit.type]['cost']
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

                stats['total_healed'] += actual_heal
                stats['total_cost'] += actual_cost
                stats['units_healed'].append({
                    'unit_type': unit.type,
                    'position': (unit.x, unit.y),
                    'structure': structure_name,
                    'healed': actual_heal,
                    'cost': actual_cost,
                    'old_health': old_health,
                    'new_health': unit.health
                })

        return stats

    def end_turn(self) -> Dict[str, Any]:
        """End the current player's turn and pass to the next player."""
        self.record_action('end_turn', player=self.current_player)

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
            unit.selected = False

        income_data = self.mechanics.calculate_income(self.current_player, self.grid)
        self.player_gold[self.current_player] += income_data['total']

        healing_stats = self.heal_units_on_structures(self.current_player)
        income_data['healing'] = healing_stats

        self.update_visibility(self.current_player)

        return income_data

    def resign(self, player: Optional[int] = None) -> None:
        """Player resigns."""
        if player is None:
            player = self.current_player

        self.record_action('resign', player=player)

        if self.num_players == 2:
            self.winner = 2 if player == 1 else 1
        else:
            self.winner = player + 1 if player < self.num_players else 1

        self.game_over = True

    def get_legal_actions(self, player: Optional[int] = None) -> Dict[str, List[Any]]:
        """Get all legal actions for the current player."""
        if player is None:
            player = self.current_player

        if self._cache_valid and player in self._legal_actions_cache:
            return self._legal_actions_cache[player]

        legal_actions: Dict[str, Any] = {
            'create_unit': [],
            'move': [],
            'attack': [],
            'paralyze': [],
            'heal': [],
            'cure': [],
            'haste': [],
            'defence_buff': [],
            'attack_buff': [],
            'seize': [],
            'end_turn': True
        }

        for tile in self.grid.get_capturable_tiles(player):
            if (tile.type == TileType.BUILDING.value
                    and not self.get_unit_at_position(tile.x, tile.y)):
                for unit_type in self.enabled_units:
                    if self.player_gold[player] >= UNIT_DATA[unit_type]['cost']:
                        legal_actions['create_unit'].append({
                            'unit_type': unit_type,
                            'x': tile.x,
                            'y': tile.y
                        })

        for unit in self.units:
            if unit.player == player and not unit.is_paralyzed():
                if unit.can_move:
                    reachable = unit.get_reachable_positions(
                        self.grid.width,
                        self.grid.height,
                        lambda x, y: self.mechanics.can_move_to_position(
                            x, y, self.grid, self.units
                        )
                    )
                    for pos in reachable:
                        legal_actions['move'].append({
                            'unit': unit,
                            'from_x': unit.x,
                            'from_y': unit.y,
                            'to_x': pos[0],
                            'to_y': pos[1]
                        })

                if unit.can_attack:
                    if unit.type in ['M', 'A', 'S']:
                        unit_tile = self.grid.get_tile(unit.x, unit.y)
                        on_mountain = (unit_tile.type == 'm')

                        for enemy in self.units:
                            if enemy.player != player:
                                if (self.fog_of_war
                                        and not self.is_enemy_attackable_by_unit(unit, enemy)):
                                    continue

                                damage = unit.get_attack_damage(
                                    enemy.x, enemy.y, on_mountain
                                )
                                if damage > 0:
                                    legal_actions['attack'].append({
                                        'attacker': unit,
                                        'target': enemy
                                    })

                                    if unit.type == 'M' and unit.can_use_paralyze():
                                        distance = (abs(unit.x - enemy.x)
                                                    + abs(unit.y - enemy.y))
                                        if distance <= 2:
                                            legal_actions['paralyze'].append({
                                                'paralyzer': unit,
                                                'target': enemy
                                            })
                    else:
                        adjacent_enemies = self.mechanics.get_adjacent_enemies(
                            unit, self.units
                        )
                        for enemy in adjacent_enemies:
                            if (self.fog_of_war
                                    and not self.is_enemy_attackable_by_unit(unit, enemy)):
                                continue

                            legal_actions['attack'].append({
                                'attacker': unit,
                                'target': enemy
                            })

                    if unit.type == 'C':
                        healable_allies = self.mechanics.get_healable_allies(
                            unit, self.units
                        )
                        for ally in healable_allies:
                            legal_actions['heal'].append({
                                'healer': unit,
                                'target': ally
                            })

                        curable_allies = self.mechanics.get_curable_allies(
                            unit, self.units
                        )
                        for ally in curable_allies:
                            legal_actions['cure'].append({
                                'curer': unit,
                                'target': ally
                            })

                    if unit.type == 'S' and unit.can_use_haste():
                        hasteable_allies = self.mechanics.get_hasteable_allies(
                            unit, self.units
                        )
                        for ally in hasteable_allies:
                            legal_actions['haste'].append({
                                'sorcerer': unit,
                                'target': ally
                            })

                    if unit.type == 'S' and unit.can_use_defence_buff():
                        buffable_allies = self.mechanics.get_defence_buffable_allies(
                            unit, self.units
                        )
                        for ally in buffable_allies:
                            legal_actions['defence_buff'].append({
                                'sorcerer': unit,
                                'target': ally
                            })

                    if unit.type == 'S' and unit.can_use_attack_buff():
                        buffable_allies = self.mechanics.get_attack_buffable_allies(
                            unit, self.units
                        )
                        for ally in buffable_allies:
                            legal_actions['attack_buff'].append({
                                'sorcerer': unit,
                                'target': ally
                            })

                    tile = self.grid.get_tile(unit.x, unit.y)
                    if tile.is_capturable() and tile.player != player:
                        legal_actions['seize'].append({
                            'unit': unit,
                            'tile': tile
                        })

        if self._cache_valid:
            self._legal_actions_cache[player] = legal_actions

        return legal_actions

    def to_dict(self) -> Dict[str, Any]:
        """Convert game state to dictionary for serialization."""
        return {
            'timestamp': self.game_start_time.strftime("%Y-%m-%d %H-%M-%S"),
            'current_player': self.current_player,
            'num_players': self.num_players,
            'player_gold': self.player_gold,
            'turn_number': self.turn_number,
            'game_over': self.game_over,
            'winner': self.winner,
            'map_file': self.map_file_used,
            'player_configs': self.player_configs,
            'enabled_units': self.enabled_units,
            'fog_of_war': self.fog_of_war,
            'fog_of_war_method': self.fog_of_war_method,
            'units': [unit.to_dict() for unit in self.units],
            'tiles': self.grid.to_dict()['tiles']
        }

    def to_numpy(self, for_player: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Convert game state to numpy arrays for RL."""
        grid_state = self.grid.to_numpy()

        unit_state = np.zeros((self.grid.height, self.grid.width, 3), dtype=np.float32)
        unit_type_encoding = {
            'W': 1, 'M': 2, 'C': 3, 'A': 4, 'K': 5, 'R': 6, 'S': 7, 'B': 8
        }

        visibility_state = np.full(
            (self.grid.height, self.grid.width), VISIBLE, dtype=np.uint8
        )

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

        if self.fog_of_war and for_player is not None:
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    if visibility_state[y, x] != VISIBLE:
                        if visibility_state[y, x] == 0:  # UNEXPLORED
                            grid_state[y, x, 0] = 0
                            grid_state[y, x, 1] = 0
                            grid_state[y, x, 2] = 0

        result: Dict[str, Any] = {
            'grid': grid_state,
            'units': unit_state,
            'gold': np.array(
                [self.player_gold[i] for i in range(1, self.num_players + 1)],
                dtype=np.float32
            ),
            'current_player': self.current_player,
            'turn_number': self.turn_number
        }

        if self.fog_of_war:
            result['visibility'] = visibility_state

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
    def from_dict(cls, save_data: Dict[str, Any], map_data) -> 'GameState':
        """Restore game state from dictionary."""
        enabled_units = save_data.get('enabled_units', cls.ALL_UNIT_TYPES)
        fog_of_war = save_data.get('fog_of_war', False)
        fog_of_war_method = save_data.get(
            'fog_of_war_method',
            'simple_radius' if fog_of_war else 'none'
        )

        game = cls(map_data, save_data.get('num_players', 2),
                   enabled_units=enabled_units, fog_of_war=fog_of_war)
        game.fog_of_war_method = fog_of_war_method

        game.current_player = save_data.get('current_player', 1)
        game.turn_number = save_data.get('turn_number', 0)
        game.game_over = save_data.get('game_over', False)
        game.winner = save_data.get('winner')

        saved_gold = save_data.get('player_gold', {})
        game.player_gold = {int(k): v for k, v in saved_gold.items()}

        game.map_file_used = save_data.get('map_file')
        game.player_configs = save_data.get('player_configs', [])

        game.units = []
        for unit_data in save_data.get('units', []):
            unit = Unit.from_dict(unit_data)
            game.units.append(unit)

        for tile_data in save_data.get('tiles', []):
            x, y = tile_data['x'], tile_data['y']
            if 0 <= x < game.grid.width and 0 <= y < game.grid.height:
                tile = game.grid.tiles[y][x]
                if tile_data.get('player'):
                    tile.player = tile_data['player']
                if tile_data.get('health') is not None:
                    tile.health = tile_data['health']
                if tile_data.get('regenerating') is not None:
                    tile.regenerating = tile_data['regenerating']

        game._invalidate_cache()
        return game
