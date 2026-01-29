"""
Random agent for Reinforce Tactics Kaggle environment.

This agent creates random units at available buildings and ends its turn.
It serves as a minimal baseline for competition participants.

Usage as a Kaggle submission:
    Copy this file and submit it. The ``agent`` function will be called
    each turn with the current observation and configuration.
"""
import random


# Unit costs for budget tracking
UNIT_COSTS = {
    "W": 200, "M": 300, "C": 200, "A": 250,
    "K": 350, "R": 350, "S": 400, "B": 400,
}


def agent(observation, configuration):
    """
    Random agent that creates a random affordable unit each turn.

    Args:
        observation: Kaggle observation struct with fields:
            - board: 2D array of terrain codes
            - structures: list of structure dicts
            - units: list of unit dicts
            - gold: [p1_gold, p2_gold]
            - player: agent's player index (0 or 1)
            - turnNumber: current turn
            - mapWidth, mapHeight: map dimensions
        configuration: Kaggle configuration struct

    Returns:
        list of action dicts
    """
    actions = []
    player_idx = observation.player
    player = player_idx + 1  # Game uses 1-indexed players
    gold = observation.gold[player_idx]

    # Get units list
    units = observation.units if hasattr(observation, "units") else []
    structures = observation.structures if hasattr(observation, "structures") else []

    # Find buildings we own that are unoccupied
    occupied = {(u["x"], u["y"]) for u in units}
    my_buildings = [
        s for s in structures
        if s["owner"] == player and s["type"] == "b"
        and (s["x"], s["y"]) not in occupied
    ]

    # Parse enabled units from configuration
    enabled_str = getattr(configuration, "enabledUnits", "W,M,C,A,K,R,S,B")
    enabled_units = [u.strip() for u in enabled_str.split(",") if u.strip()]

    # Try to create a random unit at each available building
    for bldg in my_buildings:
        affordable = [
            ut for ut in enabled_units
            if ut in UNIT_COSTS and UNIT_COSTS[ut] <= gold
        ]
        if affordable:
            unit_type = random.choice(affordable)
            actions.append({
                "type": "create_unit",
                "unit_type": unit_type,
                "x": bldg["x"],
                "y": bldg["y"],
            })
            gold -= UNIT_COSTS[unit_type]

    actions.append({"type": "end_turn"})
    return actions
