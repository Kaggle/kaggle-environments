"""
Unit class representing a game unit.
"""

from collections import deque

from ..constants import UNIT_DATA


class Unit:
    """Represents a unit on the map."""

    def __init__(self, unit_type, x, y, player):
        """
        Initialize a unit.

        Args:
            unit_type: 'W' (Warrior), 'M' (Mage), 'C' (Cleric), 'A' (Archer),
                       'K' (Knight), 'R' (Rogue), 'S' (Sorcerer), 'B' (Barbarian)
            x: X coordinate on grid
            y: Y coordinate on grid
            player: Player number who owns this unit
        """
        self.type = unit_type
        self.x = x
        self.y = y
        self.original_x = x
        self.original_y = y
        self.player = player
        self.can_move = False
        self.can_attack = False
        self.selected = False
        self.has_moved = False
        self.movement_range = UNIT_DATA[unit_type]["movement"]
        self.max_health = UNIT_DATA[unit_type]["health"]
        self.health = self.max_health
        self.attack_data = UNIT_DATA[unit_type]["attack"]
        self.defence = UNIT_DATA[unit_type]["defence"]
        self.paralyzed_turns = 0

        # Knight charge tracking
        self.distance_moved = 0

        # Mage paralyze ability tracking
        self.paralyze_cooldown = 0  # Turns remaining before can use Paralyze again

        # Sorcerer haste ability tracking
        self.haste_cooldown = 0  # Turns remaining before can use Haste again

        # Haste buff tracking (for any unit that receives Haste)
        self.is_hasted = False  # True if unit has extra action this turn

        # Sorcerer buff ability tracking (cooldowns for the Sorcerer)
        self.defence_buff_cooldown = 0  # Turns remaining before can use Defence Buff again
        self.attack_buff_cooldown = 0  # Turns remaining before can use Attack Buff again

        # Buff status tracking (for any unit that receives buffs)
        self.defence_buff_turns = 0  # Turns remaining with defence buff active
        self.attack_buff_turns = 0  # Turns remaining with attack buff active

        # Fog of war: Track which enemy positions were visible when this unit started its action
        # This prevents "move to discover, then attack" exploitation
        self.visible_enemies_at_action_start = None  # Set of (x, y) tuples, or None if not captured

    def get_attack_damage(self, target_x, target_y, on_mountain=False):
        """
        Calculate attack damage based on distance to target.

        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            on_mountain: Whether the unit is on a mountain tile (for Archer range bonus)

        Returns:
            Attack damage value
        """
        distance = abs(self.x - target_x) + abs(self.y - target_y)

        if self.type in ["M", "S"]:
            # Mage and Sorcerer have ranged attacks
            if distance == 1:
                return self.attack_data["adjacent"]
            if distance == 2:
                return self.attack_data["range"]
            return 0
        if self.type == "A":
            # Archer has range 2-3 normally, 2-4 on mountain (cannot attack at distance 1)
            max_range = 4 if on_mountain else 3
            if 2 <= distance <= max_range:
                return self.attack_data
            return 0
        if distance == 1:
            return self.attack_data
        return 0

    def get_attack_range(self, on_mountain=False):
        """
        Get the min and max attack range for this unit.

        Args:
            on_mountain: Whether the unit is on a mountain tile (for Archer range bonus)

        Returns:
            Tuple of (min_range, max_range)
        """
        if self.type in ["M", "S"]:
            # Mage and Sorcerer: distance 1-2
            return (1, 2)
        if self.type == "A":
            # Archer: distance 2-3, or 2-4 on mountain
            max_range = 4 if on_mountain else 3
            return (2, max_range)
        # Warrior, Cleric, Barbarian, Knight, Rogue: distance 1 only
        return (1, 1)

    def take_damage(self, damage):
        """
        Apply damage to the unit.

        Args:
            damage: Amount of damage to take

        Returns:
            True if unit is still alive, False if dead
        """
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            return False
        return True

    def is_paralyzed(self):
        """Check if this unit is currently paralyzed."""
        return self.paralyzed_turns > 0

    def get_reachable_positions(self, grid_width, grid_height, can_move_to_func):
        """
        Get all positions reachable within movement range using BFS.

        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid
            can_move_to_func: Function to check if a position is valid for movement

        Returns:
            List of (x, y) tuples for all reachable positions
        """
        reachable = []
        visited = set()
        queue = deque([(self.x, self.y, 0)])
        visited.add((self.x, self.y))

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while queue:
            curr_x, curr_y, distance = queue.popleft()

            if distance > 0:
                reachable.append((curr_x, curr_y))

            if distance < self.movement_range:
                for dx, dy in directions:
                    new_x = curr_x + dx
                    new_y = curr_y + dy

                    if (new_x, new_y) not in visited:
                        if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
                            if can_move_to_func(new_x, new_y):
                                visited.add((new_x, new_y))
                                queue.append((new_x, new_y, distance + 1))

        return reachable

    def move_to(self, x, y):
        """Move the unit to a new position."""
        # Calculate Manhattan distance moved (for Knight's Charge ability)
        self.distance_moved = abs(x - self.original_x) + abs(y - self.original_y)
        self.x = x
        self.y = y
        self.has_moved = True
        self.selected = False

    def cancel_move(self):
        """Cancel the unit's movement and return to original position.

        Also resets can_move to True so the unit can move again.
        """
        if self.has_moved:
            self.x = self.original_x
            self.y = self.original_y
            self.has_moved = False
            self.can_move = True  # Allow unit to move again after cancel
            self.distance_moved = 0  # Reset distance for Knight's Charge
            return True
        return False

    def end_unit_turn(self, force_end=False):
        """End this unit's turn.

        If the unit has haste (is_hasted=True) and force_end is False,
        the haste is consumed and the unit gets another full action instead
        of ending its turn.

        Args:
            force_end: If True, always end the turn even if hasted

        Returns:
            bool: True if the unit can still act (haste was consumed),
                  False if the turn actually ended
        """
        # If hasted and not forcing end, consume haste and refresh for another action
        if self.is_hasted and not force_end:
            self.is_hasted = False
            self.can_move = True
            self.can_attack = True
            self.has_moved = False
            self.original_x = self.x
            self.original_y = self.y
            self.distance_moved = 0
            self.selected = False
            self.visible_enemies_at_action_start = None  # Clear FOW snapshot for new action
            return True  # Unit can still act

        # Normal turn end
        self.can_move = False
        self.can_attack = False
        self.selected = False
        self.has_moved = False
        self.original_x = self.x
        self.original_y = self.y
        self.distance_moved = 0
        self.is_hasted = False
        self.visible_enemies_at_action_start = None  # Clear FOW snapshot
        return False  # Turn ended

    def can_use_paralyze(self):
        """Check if this Mage can use Paralyze ability."""
        return self.type == "M" and self.paralyze_cooldown == 0

    def can_use_haste(self):
        """Check if this Sorcerer can use Haste ability."""
        return self.type == "S" and self.haste_cooldown == 0

    def can_use_defence_buff(self):
        """Check if this Sorcerer can use Defence Buff ability."""
        return self.type == "S" and self.defence_buff_cooldown == 0

    def can_use_attack_buff(self):
        """Check if this Sorcerer can use Attack Buff ability."""
        return self.type == "S" and self.attack_buff_cooldown == 0

    def has_defence_buff(self):
        """Check if this unit has an active defence buff."""
        return self.defence_buff_turns > 0

    def has_attack_buff(self):
        """Check if this unit has an active attack buff."""
        return self.attack_buff_turns > 0

    def to_dict(self):
        """Convert unit to dictionary for serialization."""
        return {
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "player": self.player,
            "health": self.health,
            "paralyzed_turns": self.paralyzed_turns,
            "paralyze_cooldown": self.paralyze_cooldown,
            "can_move": self.can_move,
            "can_attack": self.can_attack,
            "haste_cooldown": self.haste_cooldown,
            "is_hasted": self.is_hasted,
            "distance_moved": self.distance_moved,
            "defence_buff_cooldown": self.defence_buff_cooldown,
            "attack_buff_cooldown": self.attack_buff_cooldown,
            "defence_buff_turns": self.defence_buff_turns,
            "attack_buff_turns": self.attack_buff_turns,
        }

    @classmethod
    def from_dict(cls, data):
        """Create unit from dictionary."""
        unit = cls(data["type"], data["x"], data["y"], data["player"])
        unit.health = data["health"]
        unit.paralyzed_turns = data.get("paralyzed_turns", 0)
        unit.paralyze_cooldown = data.get("paralyze_cooldown", 0)
        unit.can_move = data.get("can_move", True)
        unit.can_attack = data.get("can_attack", True)
        unit.haste_cooldown = data.get("haste_cooldown", 0)
        unit.is_hasted = data.get("is_hasted", False)
        unit.distance_moved = data.get("distance_moved", 0)
        unit.defence_buff_cooldown = data.get("defence_buff_cooldown", 0)
        unit.attack_buff_cooldown = data.get("attack_buff_cooldown", 0)
        unit.defence_buff_turns = data.get("defence_buff_turns", 0)
        unit.attack_buff_turns = data.get("attack_buff_turns", 0)
        unit.original_x = unit.x
        unit.original_y = unit.y
        return unit
