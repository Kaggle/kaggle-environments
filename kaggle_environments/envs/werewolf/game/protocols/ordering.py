import random
from typing import List, Set

from kaggle_environments.envs.werewolf.game.consts import StrEnum


class FirstPlayerStrategy(StrEnum):
    FIXED = "fixed"
    ROTATE = "rotate"
    RANDOM = "random"


class PivotSelector:
    def __init__(self, strategy: str = FirstPlayerStrategy.ROTATE):
        self.strategy = FirstPlayerStrategy(strategy)
        self._current_cursor = 0

    def get_pivot(self, all_ids: List[str], alive_ids: Set[str]) -> int:
        """
        Determines the starting pivot index based on the strategy.
        Updates internal state (_current_cursor) if strategy is ROTATE.
        """
        num_players = len(all_ids)
        if num_players == 0:
            return 0

        pivot = 0

        if self.strategy == FirstPlayerStrategy.FIXED:
            pivot = 0
        elif self.strategy == FirstPlayerStrategy.RANDOM:
            pivot = random.randrange(num_players)
        elif self.strategy == FirstPlayerStrategy.ROTATE:
            pivot_found = False
            for i in range(num_players):
                idx = (self._current_cursor + i) % num_players
                if all_ids[idx] in alive_ids:
                    pivot = idx
                    self._current_cursor = (pivot + 1) % num_players
                    pivot_found = True
                    break
            if not pivot_found:
                pivot = 0

        return pivot

    @staticmethod
    def get_ordered_ids(all_ids: List[str], pivot: int) -> List[str]:
        """Returns a list of player IDs ordered starting from the pivot index."""
        num_players = len(all_ids)
        if num_players == 0:
            return []
        return [all_ids[(pivot + i) % num_players] for i in range(num_players)]
