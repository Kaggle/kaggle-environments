"""
Core game mechanics including combat, movement, income, and structure capture.
"""

import random

from ..constants import (
    BUILDING_INCOME,
    CHARGE_BONUS,
    CHARGE_MIN_DISTANCE,
    COUNTER_ATTACK_MULTIPLIER,
    DEFENCE_REDUCTION_PER_POINT,
    FLANK_BONUS,
    HASTE_COOLDOWN,
    HEADQUARTERS_INCOME,
    HEAL_AMOUNT,
    PARALYZE_COOLDOWN,
    PARALYZE_DURATION,
    ROGUE_EVADE_CHANCE,
    ROGUE_FOREST_EVADE_BONUS,
    SORCERER_ATTACK_BUFF_AMOUNT,
    SORCERER_BUFF_COOLDOWN,
    SORCERER_BUFF_DURATION,
    SORCERER_DEFENCE_BUFF_AMOUNT,
    STRUCTURE_REGEN_RATE,
    TOWER_INCOME,
)


class GameMechanics:
    """Handles core game mechanics and rules."""

    @staticmethod
    def can_move_to_position(x, y, grid, units, moving_unit=None, is_destination=False):
        """
        Check if a position is valid for unit movement.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            grid: TileGrid instance
            units: List of Unit instances
            moving_unit: The unit that is moving (optional, for team checking)
            is_destination: If True, blocks all units. If False (pathfinding),
                           only blocks enemy units (default: False)

        Returns:
            True if position is valid for movement
        """
        if not (0 <= x < grid.width and 0 <= y < grid.height):
            return False

        tile = grid.get_tile(x, y)
        if not tile.is_walkable():
            return False

        # Check if another unit is already there
        for unit in units:
            if unit.x == x and unit.y == y:
                # If this is the final destination, block all units
                if is_destination:
                    return False

                # For pathfinding, allow passing through friendly units
                if moving_unit is not None and unit.player == moving_unit.player:
                    continue  # Allow passing through friendly units

                # Block enemy units or if no moving_unit specified (legacy behavior)
                return False

        return True

    @staticmethod
    def get_adjacent_enemies(unit, units):
        """Get list of enemy units adjacent to the given unit."""
        adjacent_enemies = []
        adjacent_positions = [(unit.x, unit.y - 1), (unit.x, unit.y + 1), (unit.x - 1, unit.y), (unit.x + 1, unit.y)]

        for enemy in units:
            if enemy.player != unit.player and enemy.health > 0:
                if (enemy.x, enemy.y) in adjacent_positions:
                    adjacent_enemies.append(enemy)

        return adjacent_enemies

    @staticmethod
    def get_attackable_enemies(unit, units, grid):
        """
        Get list of enemy units within the given unit's attack range.

        Args:
            unit: The unit to check attack range for
            units: List of all units
            grid: TileGrid instance (for checking mountain tiles)

        Returns:
            List of enemy units within attack range
        """
        attackable_enemies = []

        # Check if unit is on a mountain (for Archer range bonus)
        on_mountain = False
        if grid:
            tile = grid.get_tile(unit.x, unit.y)
            on_mountain = tile.type == "m"

        # Get the unit's attack range
        min_range, max_range = unit.get_attack_range(on_mountain)

        # Check all enemies
        for enemy in units:
            if enemy.player != unit.player and enemy.health > 0:
                distance = abs(unit.x - enemy.x) + abs(unit.y - enemy.y)
                if min_range <= distance <= max_range:
                    attackable_enemies.append(enemy)

        return attackable_enemies

    @staticmethod
    def get_adjacent_allies(unit, units):
        """Get list of damaged friendly units adjacent to the given unit."""
        adjacent_allies = []
        adjacent_positions = [(unit.x, unit.y - 1), (unit.x, unit.y + 1), (unit.x - 1, unit.y), (unit.x + 1, unit.y)]

        for ally in units:
            if ally.player == unit.player and ally.health > 0 and ally != unit:
                if (ally.x, ally.y) in adjacent_positions:
                    if ally.health < ally.max_health:
                        adjacent_allies.append(ally)

        return adjacent_allies

    @staticmethod
    def get_healable_allies(cleric, units):
        """
        Get list of damaged friendly units within the Cleric's heal range (1-2 tiles).

        Args:
            cleric: The Cleric unit
            units: List of all units

        Returns:
            List of allied units within range 1-2 that are damaged
        """
        healable = []

        for ally in units:
            if ally.player == cleric.player and ally.health > 0 and ally != cleric:
                # Check distance (range 1-2)
                distance = abs(cleric.x - ally.x) + abs(cleric.y - ally.y)
                if 1 <= distance <= 2 and ally.health < ally.max_health:
                    healable.append(ally)

        return healable

    @staticmethod
    def get_curable_allies(cleric, units):
        """
        Get list of paralyzed friendly units within the Cleric's cure range (1-2 tiles).

        Args:
            cleric: The Cleric unit
            units: List of all units

        Returns:
            List of allied units within range 1-2 that are paralyzed
        """
        curable = []

        for ally in units:
            if ally.player == cleric.player and ally.health > 0 and ally != cleric:
                # Check distance (range 1-2)
                distance = abs(cleric.x - ally.x) + abs(cleric.y - ally.y)
                if 1 <= distance <= 2 and ally.is_paralyzed():
                    curable.append(ally)

        return curable

    @staticmethod
    def get_adjacent_paralyzed_allies(unit, units):
        """Get list of paralyzed friendly units adjacent to the given unit."""
        adjacent_paralyzed = []
        adjacent_positions = [(unit.x, unit.y - 1), (unit.x, unit.y + 1), (unit.x - 1, unit.y), (unit.x + 1, unit.y)]

        for ally in units:
            if ally.player == unit.player and ally.health > 0 and ally != unit:
                if (ally.x, ally.y) in adjacent_positions:
                    if ally.is_paralyzed():
                        adjacent_paralyzed.append(ally)

        return adjacent_paralyzed

    @staticmethod
    def is_enemy_flanked(attacker, target, units):
        """
        Check if the target enemy is flanked (adjacent to at least one of attacker's allies).

        Args:
            attacker: The attacking unit
            target: The target enemy unit
            units: List of all units

        Returns:
            True if target is adjacent to at least one of attacker's allies (excluding attacker)
        """
        adjacent_positions = [
            (target.x, target.y - 1),
            (target.x, target.y + 1),
            (target.x - 1, target.y),
            (target.x + 1, target.y),
        ]

        for unit in units:
            if unit.player == attacker.player and unit != attacker and unit.health > 0:
                if (unit.x, unit.y) in adjacent_positions:
                    return True

        return False

    @staticmethod
    def apply_defence_reduction(base_damage, target_defence):
        """
        Apply defence reduction to damage using percentage reduction.

        Each point of defence reduces damage by 5%.

        Args:
            base_damage: The raw damage before defence
            target_defence: The target's defence stat

        Returns:
            Reduced damage as integer (minimum 1)
        """
        reduction = target_defence * DEFENCE_REDUCTION_PER_POINT
        # Cap reduction at 90% to ensure some damage always gets through
        reduction = min(reduction, 0.9)
        reduced_damage = base_damage * (1 - reduction)
        return max(1, int(reduced_damage))

    @staticmethod
    def get_hasteable_allies(sorcerer, units):
        """
        Get list of friendly units that can receive Haste from the Sorcerer.

        Args:
            sorcerer: The Sorcerer unit
            units: List of all units

        Returns:
            List of allied units (excluding sorcerer) within range 1-2 that haven't been hasted
        """
        hasteable = []

        for unit in units:
            if unit.player == sorcerer.player and unit != sorcerer and unit.health > 0:
                # Haste range is 1-2 tiles
                distance = abs(sorcerer.x - unit.x) + abs(sorcerer.y - unit.y)
                if 1 <= distance <= 2 and not unit.is_hasted:
                    hasteable.append(unit)

        return hasteable

    @staticmethod
    def get_defence_buffable_allies(sorcerer, units):
        """
        Get list of friendly units that can receive Defence Buff from the Sorcerer.

        Args:
            sorcerer: The Sorcerer unit
            units: List of all units

        Returns:
            List of allied units within range 1-2 that don't have defence buff
        """
        buffable = []

        for unit in units:
            if unit.player == sorcerer.player and unit.health > 0:
                # Buff range is 1-2 tiles (can buff self at distance 0)
                distance = abs(sorcerer.x - unit.x) + abs(sorcerer.y - unit.y)
                if distance <= 2 and not unit.has_defence_buff():
                    buffable.append(unit)

        return buffable

    @staticmethod
    def get_attack_buffable_allies(sorcerer, units):
        """
        Get list of friendly units that can receive Attack Buff from the Sorcerer.

        Args:
            sorcerer: The Sorcerer unit
            units: List of all units

        Returns:
            List of allied units within range 1-2 that don't have attack buff
        """
        buffable = []

        for unit in units:
            if unit.player == sorcerer.player and unit.health > 0:
                # Buff range is 1-2 tiles (can buff self at distance 0)
                distance = abs(sorcerer.x - unit.x) + abs(sorcerer.y - unit.y)
                if distance <= 2 and not unit.has_attack_buff():
                    buffable.append(unit)

        return buffable

    @staticmethod
    def _calculate_counter_damage(unit, target_x, target_y, grid):
        """
        Calculate counter-attack damage for a unit.

        Args:
            unit: The unit that would counter-attack
            target_x: X coordinate of the target
            target_y: Y coordinate of the target
            grid: TileGrid instance (optional, for checking mountain tiles)

        Returns:
            Counter-attack damage as integer
        """
        on_mountain = False
        if grid:
            tile = grid.get_tile(unit.x, unit.y)
            on_mountain = tile.type == "m"

        return int(unit.get_attack_damage(target_x, target_y, on_mountain) * COUNTER_ATTACK_MULTIPLIER)

    @staticmethod
    def attack_unit(attacker, target, grid=None, units=None):
        """
        Execute an attack from attacker to target.

        Args:
            attacker: The attacking unit
            target: The target unit
            grid: TileGrid instance (optional, for checking mountain tiles)
            units: List of all units (optional, for flanking checks)

        Returns:
            dict with 'attacker_alive', 'target_alive', 'damage', 'counter_damage',
            and bonus info ('charge_bonus', 'flank_bonus')
        """
        # Check if attacker is on mountain for range calculation
        attacker_on_mountain = False
        if grid:
            attacker_tile = grid.get_tile(attacker.x, attacker.y)
            attacker_on_mountain = attacker_tile.type == "m"

        # Calculate base attack damage
        base_attack_damage = attacker.get_attack_damage(target.x, target.y, attacker_on_mountain)

        # Apply special ability bonuses
        charge_applied = False
        flank_applied = False
        evade_applied = False
        attack_buff_applied = False
        defence_buff_applied = False

        # Knight's Charge: +50% damage if moved 3+ tiles
        if attacker.type == "K" and attacker.distance_moved >= CHARGE_MIN_DISTANCE:
            base_attack_damage = int(base_attack_damage * (1 + CHARGE_BONUS))
            charge_applied = True

        # Rogue's Flank: +50% damage if enemy is adjacent to another friendly unit
        if attacker.type == "R" and units:
            if GameMechanics.is_enemy_flanked(attacker, target, units):
                base_attack_damage = int(base_attack_damage * (1 + FLANK_BONUS))
                flank_applied = True

        # Sorcerer's Attack Buff: +50% damage if attacker has attack buff
        if attacker.has_attack_buff():
            base_attack_damage = int(base_attack_damage * (1 + SORCERER_ATTACK_BUFF_AMOUNT))
            attack_buff_applied = True

        # Apply defence reduction to attack damage
        attack_damage = GameMechanics.apply_defence_reduction(base_attack_damage, target.defence)

        # Sorcerer's Defence Buff: -50% damage taken if target has defence buff
        if target.has_defence_buff():
            attack_damage = max(1, int(attack_damage * (1 - SORCERER_DEFENCE_BUFF_AMOUNT)))
            defence_buff_applied = True

        target_alive = target.take_damage(attack_damage)

        attacker_alive = True
        counter_damage = 0

        # Counter-attack logic with Archer restrictions
        if target_alive and not target.is_paralyzed():
            # Determine if counter-attack is allowed
            can_counter = True

            # If attacker is an Archer, only Archers, Mages, and Sorcerers can counter
            if attacker.type == "A":
                if target.type not in ["A", "M", "S"]:
                    can_counter = False

            # Rogue's Evade: 15% chance to dodge counter-attacks (30% in forest)
            if can_counter and attacker.type == "R":
                evade_chance = ROGUE_EVADE_CHANCE
                # Check if Rogue is in forest for bonus evade chance
                if grid:
                    rogue_tile = grid.get_tile(attacker.x, attacker.y)
                    if rogue_tile.type == "f":  # Forest tile
                        evade_chance += ROGUE_FOREST_EVADE_BONUS
                if random.random() < evade_chance:
                    can_counter = False
                    evade_applied = True

            if can_counter:
                # Calculate base counter damage
                base_counter_damage = GameMechanics._calculate_counter_damage(target, attacker.x, attacker.y, grid)

                # Apply attack buff to counter-attacker (target)
                if target.has_attack_buff():
                    base_counter_damage = int(base_counter_damage * (1 + SORCERER_ATTACK_BUFF_AMOUNT))

                # Apply defence reduction to counter damage
                counter_damage = GameMechanics.apply_defence_reduction(base_counter_damage, attacker.defence)

                # Apply defence buff to attacker receiving counter damage
                if attacker.has_defence_buff():
                    counter_damage = max(1, int(counter_damage * (1 - SORCERER_DEFENCE_BUFF_AMOUNT)))

                if counter_damage > 0:
                    attacker_alive = attacker.take_damage(counter_damage)

        # Calculate counter damage for response (even if 0, or if evaded)
        counter_damage_for_response = 0
        if target_alive and not target.is_paralyzed() and not evade_applied:
            # Adjust counter_damage if Archer attacked melee unit
            if attacker.type == "A" and target.type not in ["A", "M", "S"]:
                counter_damage_for_response = 0
            else:
                base_counter = GameMechanics._calculate_counter_damage(target, attacker.x, attacker.y, grid)

                # Apply attack buff to counter-attacker (target)
                if target.has_attack_buff():
                    base_counter = int(base_counter * (1 + SORCERER_ATTACK_BUFF_AMOUNT))

                counter_damage_for_response = GameMechanics.apply_defence_reduction(base_counter, attacker.defence)

                # Apply defence buff to attacker receiving counter damage
                if attacker.has_defence_buff():
                    counter_damage_for_response = max(
                        1, int(counter_damage_for_response * (1 - SORCERER_DEFENCE_BUFF_AMOUNT))
                    )

        return {
            "attacker_alive": attacker_alive,
            "target_alive": target_alive,
            "damage": attack_damage,
            "counter_damage": counter_damage_for_response,
            "charge_bonus": charge_applied,
            "flank_bonus": flank_applied,
            "evade": evade_applied,
            "attack_buff": attack_buff_applied,
            "defence_buff": defence_buff_applied,
        }

    @staticmethod
    def paralyze_unit(paralyzer, target):
        """Mage paralyzes the target unit."""
        if paralyzer.type != "M":
            return False

        if paralyzer.paralyze_cooldown > 0:
            return False

        if target.player == paralyzer.player:
            return False

        target.paralyzed_turns = PARALYZE_DURATION
        paralyzer.paralyze_cooldown = PARALYZE_COOLDOWN
        return True

    @staticmethod
    def heal_unit(healer, target):
        """
        Healer heals the target unit.

        Args:
            healer: The unit doing the healing (must be Cleric)
            target: The target unit to heal

        Returns:
            int: Actual amount healed, or -1 if heal failed
        """
        if healer.type != "C":
            return -1

        if target.player != healer.player:
            return -1

        # Check distance (range 1-2)
        distance = abs(healer.x - target.x) + abs(healer.y - target.y)
        if distance < 1 or distance > 2:
            return -1

        if target.health >= target.max_health:
            return -1

        old_health = target.health
        target.health = min(target.health + HEAL_AMOUNT, target.max_health)
        return target.health - old_health

    @staticmethod
    def cure_unit(curer, target):
        """Cleric cures the target unit's paralysis."""
        if curer.type != "C":
            return False

        if target.player != curer.player:
            return False

        # Check distance (range 1-2)
        distance = abs(curer.x - target.x) + abs(curer.y - target.y)
        if distance < 1 or distance > 2:
            return False

        if not target.is_paralyzed():
            return False

        target.paralyzed_turns = 0
        target.can_move = True
        target.can_attack = True
        return True

    @staticmethod
    def haste_unit(sorcerer, target):
        """
        Sorcerer grants Haste to target unit, allowing an extra action.

        Args:
            sorcerer: The Sorcerer unit using Haste
            target: The target friendly unit to receive Haste

        Returns:
            bool: True if Haste was successfully applied
        """
        if sorcerer.type != "S":
            return False

        if sorcerer.haste_cooldown > 0:
            return False

        if target.player != sorcerer.player:
            return False

        if target == sorcerer:
            return False

        if target.is_hasted:
            return False

        # Check distance (range 1-2)
        distance = abs(sorcerer.x - target.x) + abs(sorcerer.y - target.y)
        if distance < 1 or distance > 2:
            return False

        # Apply haste to target
        target.is_hasted = True
        target.can_move = True
        target.can_attack = True

        # Set cooldown on sorcerer
        sorcerer.haste_cooldown = HASTE_COOLDOWN

        return True

    @staticmethod
    def defence_buff_unit(sorcerer, target):
        """
        Sorcerer grants Defence Buff to target unit, reducing damage taken by 35%.

        Args:
            sorcerer: The Sorcerer unit using Defence Buff
            target: The target friendly unit to receive the buff

        Returns:
            bool: True if Defence Buff was successfully applied
        """
        if sorcerer.type != "S":
            return False

        if sorcerer.defence_buff_cooldown > 0:
            return False

        if target.player != sorcerer.player:
            return False

        if target.has_defence_buff():
            return False

        # Check distance (range 0-2, can buff self)
        distance = abs(sorcerer.x - target.x) + abs(sorcerer.y - target.y)
        if distance > 2:
            return False

        # Apply defence buff to target
        target.defence_buff_turns = SORCERER_BUFF_DURATION

        # Set cooldown on sorcerer
        sorcerer.defence_buff_cooldown = SORCERER_BUFF_COOLDOWN

        return True

    @staticmethod
    def attack_buff_unit(sorcerer, target):
        """
        Sorcerer grants Attack Buff to target unit, increasing damage dealt by 35%.

        Args:
            sorcerer: The Sorcerer unit using Attack Buff
            target: The target friendly unit to receive the buff

        Returns:
            bool: True if Attack Buff was successfully applied
        """
        if sorcerer.type != "S":
            return False

        if sorcerer.attack_buff_cooldown > 0:
            return False

        if target.player != sorcerer.player:
            return False

        if target.has_attack_buff():
            return False

        # Check distance (range 0-2, can buff self)
        distance = abs(sorcerer.x - target.x) + abs(sorcerer.y - target.y)
        if distance > 2:
            return False

        # Apply attack buff to target
        target.attack_buff_turns = SORCERER_BUFF_DURATION

        # Set cooldown on sorcerer
        sorcerer.attack_buff_cooldown = SORCERER_BUFF_COOLDOWN

        return True

    @staticmethod
    def decrement_paralyze_cooldowns(units, player):
        """
        Decrement paralyze cooldowns for a player's Mages at turn start.

        Args:
            units: List of all units
            player: Player number whose turn is starting

        Returns:
            List of Mages that came off cooldown
        """
        ready = []
        for unit in units:
            if unit.player == player and unit.type == "M" and unit.paralyze_cooldown > 0:
                unit.paralyze_cooldown -= 1
                if unit.paralyze_cooldown == 0:
                    ready.append(unit)
        return ready

    @staticmethod
    def decrement_haste_cooldowns(units, player):
        """
        Decrement haste cooldowns for a player's Sorcerers at turn start.

        Args:
            units: List of all units
            player: Player number whose turn is starting

        Returns:
            List of Sorcerers that came off cooldown
        """
        ready = []
        for unit in units:
            if unit.player == player and unit.type == "S" and unit.haste_cooldown > 0:
                unit.haste_cooldown -= 1
                if unit.haste_cooldown == 0:
                    ready.append(unit)
        return ready

    @staticmethod
    def decrement_buff_cooldowns(units, player):
        """
        Decrement buff cooldowns for a player's Sorcerers at turn start.

        Args:
            units: List of all units
            player: Player number whose turn is starting

        Returns:
            Dict with lists of Sorcerers that came off defence/attack buff cooldown
        """
        defence_ready = []
        attack_ready = []
        for unit in units:
            if unit.player == player and unit.type == "S":
                if unit.defence_buff_cooldown > 0:
                    unit.defence_buff_cooldown -= 1
                    if unit.defence_buff_cooldown == 0:
                        defence_ready.append(unit)
                if unit.attack_buff_cooldown > 0:
                    unit.attack_buff_cooldown -= 1
                    if unit.attack_buff_cooldown == 0:
                        attack_ready.append(unit)
        return {"defence_ready": defence_ready, "attack_ready": attack_ready}

    @staticmethod
    def decrement_buff_durations(units, player):
        """
        Decrement buff durations for a player's units at turn start.

        Args:
            units: List of all units
            player: Player number whose turn is starting

        Returns:
            Dict with lists of units that lost defence/attack buff
        """
        defence_expired = []
        attack_expired = []
        for unit in units:
            if unit.player == player:
                if unit.defence_buff_turns > 0:
                    unit.defence_buff_turns -= 1
                    if unit.defence_buff_turns == 0:
                        defence_expired.append(unit)
                if unit.attack_buff_turns > 0:
                    unit.attack_buff_turns -= 1
                    if unit.attack_buff_turns == 0:
                        attack_expired.append(unit)
        return {"defence_expired": defence_expired, "attack_expired": attack_expired}

    @staticmethod
    def seize_structure(unit, tile):
        """
        Unit seizes a structure (tower, building, or HQ).

        Args:
            unit: The unit seizing
            tile: The structure tile

        Returns:
            dict with 'captured' boolean and 'game_over' boolean
        """
        if not tile.is_capturable():
            return {"captured": False, "game_over": False}

        if tile.player == unit.player:
            return {"captured": False, "game_over": False}

        if tile.regenerating:
            tile.regenerating = False

        damage = unit.health
        tile.health -= damage

        captured = False
        game_over = False

        if tile.health <= 0:
            tile.health = tile.max_health
            tile.player = unit.player
            tile.regenerating = False
            captured = True

            if tile.type == "h":
                game_over = True

        return {"captured": captured, "game_over": game_over, "damage": damage, "remaining_hp": tile.health}

    @staticmethod
    def reset_structure_if_vacated(tile, units):
        """Reset structure HP if no unit is on it."""
        if not tile.is_capturable():
            return False

        # Check if any unit is on this tile
        for unit in units:
            if unit.x == tile.x and unit.y == tile.y:
                return False

        if tile.health < tile.max_health:
            tile.health = tile.max_health
            tile.regenerating = False
            return True

        return False

    @staticmethod
    def regenerate_structures(grid, units):
        """Regenerate HP for structures that are marked for regeneration."""
        regenerated = []
        for row in grid.tiles:
            for tile in row:
                if tile.is_capturable() and tile.regenerating:
                    # Check if there's a unit on this tile
                    unit_on_tile = False
                    for unit in units:
                        if unit.x == tile.x and unit.y == tile.y:
                            unit_on_tile = True
                            tile.regenerating = False
                            break

                    if not unit_on_tile:
                        regen_amount = int(tile.max_health * STRUCTURE_REGEN_RATE)
                        old_health = tile.health
                        tile.health = min(tile.health + regen_amount, tile.max_health)

                        if tile.health >= tile.max_health:
                            tile.regenerating = False

                        regenerated.append({"tile": tile, "amount": tile.health - old_health})

        return regenerated

    @staticmethod
    def calculate_income(player, grid):
        """Calculate income for a player based on controlled structures."""
        headquarters_count = 0
        building_count = 0
        tower_count = 0

        for row in grid.tiles:
            for tile in row:
                if tile.player == player:
                    if tile.type == "h":
                        headquarters_count += 1
                    elif tile.type == "b":
                        building_count += 1
                    elif tile.type == "t":
                        tower_count += 1

        total_income = (
            headquarters_count * HEADQUARTERS_INCOME + building_count * BUILDING_INCOME + tower_count * TOWER_INCOME
        )

        return {
            "total": total_income,
            "headquarters": headquarters_count,
            "buildings": building_count,
            "towers": tower_count,
        }

    @staticmethod
    def decrement_paralysis(units, player):
        """Decrement paralysis counters for a player's units at turn start."""
        cured = []
        for unit in units:
            if unit.player == player and unit.is_paralyzed():
                unit.paralyzed_turns -= 1
                if unit.paralyzed_turns <= 0:
                    cured.append(unit)
        return cured
