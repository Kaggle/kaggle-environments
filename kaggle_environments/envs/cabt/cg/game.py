import ctypes
import json

from .sim import lib, Battle

def _get_battle_data() -> dict:
    """Retrieve the current state.

    Returns:
        dict: Current observation.
    """
    sd = lib.GetBattleData(Battle.battle_ptr)
    Battle.obs = json.loads(sd.json.decode())
    Battle.obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode('ascii')
    return Battle.obs

def battle_start(deck0: list[int], deck1: list[int]) -> dict:
    """Start the battle.
    
    Args:
        deck0: List of card IDs included in the first player’s deck.
        deck1: List of card IDs included in the second player’s deck.

    Returns:
        dict: First observation.
    """
    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("The deck must contain 60 cards.")
    cards = deck0 + deck1
    arg = (ctypes.c_int*len(cards))(*cards)
    Battle.battle_ptr = lib.BattleStart(arg)
    if Battle.battle_ptr == 0:
        raise ValueError("Invalid deck.")
    return _get_battle_data()

def battle_finish():
    """End the battle and free the memory used during it."""
    lib.BattleFinish(Battle.battle_ptr)

def battle_select(select_list: list[int]) -> dict:
    """Select option.
    
    Args:
        select_list: 

    Returns:
        dict: Next observation.
    """
    if not isinstance(select_list, list) or not all(isinstance(i, int) for i in select_list):
        raise ValueError("select_list is not list[int]")
    arg = (ctypes.c_int*len(select_list))(*select_list)
    err = lib.Select(Battle.battle_ptr, arg, len(select_list))
    if err != 0:
        if err == 30:
            raise ValueError("battle_ptr broken.")
        else:
            raise IndexError()
    return _get_battle_data()

def visualize_data() -> str:
    """Retrieve the data to be used by the visualizer.

    Returns:
        str: The data to be used by the visualizer.
    """
    return lib.VisualizeData(Battle.battle_ptr).decode()
