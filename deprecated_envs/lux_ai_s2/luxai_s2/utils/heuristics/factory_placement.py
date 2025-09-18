import numpy as np
from luxai_s2.unit import FactoryPlacementActionType
from luxai_s2.state import ObservationStateDict

def random_factory_placement(player, obs: ObservationStateDict) -> FactoryPlacementActionType:
    """
    This policy places factories with 150 water and metal at random locations
    """
    # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
    potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
    spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
    return dict(spawn=spawn_loc, metal=150, water=150)

def place_near_random_ice(player, obs: ObservationStateDict):
    if obs["teams"][player]["metal"] == 0:
        return dict()
    potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
    potential_spawns_set = set(potential_spawns)
    done_search = False
    # if player == "player_1":
    ice_diff = np.diff(obs["board"]["ice"])
    pot_ice_spots = np.argwhere(ice_diff == 1)
    if len(pot_ice_spots) == 0:
        pot_ice_spots = potential_spawns
    trials = 5
    while trials > 0:
        pos_idx = np.random.randint(0, len(pot_ice_spots))
        pos = pot_ice_spots[pos_idx]

        area = 3
        for x in range(area):
            for y in range(area):
                check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                if tuple(check_pos) in potential_spawns_set:
                    done_search = True
                    pos = check_pos
                    break
            if done_search:
                break
        if done_search:
            break
        trials -= 1
    spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
    if not done_search:
        pos = spawn_loc

    metal = obs["teams"][player]["metal"]
    return dict(spawn=pos, metal=metal, water=metal)
