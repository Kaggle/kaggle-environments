from dataclasses import dataclass, field
from typing import Dict
import numpy as np
if __package__ == "":
    from lux.cargo import UnitCargo
    from lux.config import EnvConfig
    from lux.team import Team, FactionTypes
    from lux.unit import Unit
    from lux.factory import Factory
else:
    from .cargo import UnitCargo
    from .config import EnvConfig
    from .team import Team, FactionTypes
    from .unit import Unit
    from .factory import Factory
def process_action(action):
    return to_json(action)
def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(s) for s in obj]
    elif isinstance(obj, dict):
        out = {}
        for k in obj:
            out[k] = to_json(obj[k])
        return out
    else:
        return obj
def from_json(state):
    if isinstance(state, list):
        return np.array(state)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state 

def process_obs(player, game_state, step, obs):
    if step == 0:
        # at step 0 we get the entire map information
        game_state = from_json(obs)
    else:
        # use delta changes to board to update game state
        obs = from_json(obs)
        for k in obs:
            if k != 'board':
                game_state[k] = obs[k]
            else:
                if "valid_spawns_mask" in obs[k]:
                    game_state["board"]["valid_spawns_mask"] = obs[k]["valid_spawns_mask"]
        for item in ["rubble", "lichen", "lichen_strains"]:
            for k, v in obs["board"][item].items():
                k = k.split(",")
                x, y = int(k[0]), int(k[1])
                game_state["board"][item][x, y] = v
    return game_state

def obs_to_game_state(step, env_cfg: EnvConfig, obs):
    
    units = dict()
    for agent in obs["units"]:
        units[agent] = dict()
        for unit_id in obs["units"][agent]:
            unit_data = obs["units"][agent][unit_id]
            cargo = UnitCargo(**unit_data["cargo"])
            unit = Unit(
                **unit_data,
                unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]],
                env_cfg=env_cfg
            )
            unit.cargo = cargo
            units[agent][unit_id] = unit
            

    factory_occupancy_map = np.ones_like(obs["board"]["rubble"], dtype=int) * -1
    factories = dict()
    for agent in obs["factories"]:
        factories[agent] = dict()
        for unit_id in obs["factories"][agent]:
            f_data = obs["factories"][agent][unit_id]
            cargo = UnitCargo(**f_data["cargo"])
            factory = Factory(
                **f_data,
                env_cfg=env_cfg
            )
            factory.cargo = cargo
            factories[agent][unit_id] = factory
            factory_occupancy_map[factory.pos_slice] = factory.strain_id
    teams = dict()
    for agent in obs["teams"]:
        team_data = obs["teams"][agent]
        faction = FactionTypes[team_data["faction"]]
        teams[agent] = Team(**team_data, agent=agent)

    return GameState(
        env_cfg=env_cfg,
        env_steps=step,
        board=Board(
            rubble=obs["board"]["rubble"],
            ice=obs["board"]["ice"],
            ore=obs["board"]["ore"],
            lichen=obs["board"]["lichen"],
            lichen_strains=obs["board"]["lichen_strains"],
            factory_occupancy_map=factory_occupancy_map,
            factories_per_team=obs["board"]["factories_per_team"],
            valid_spawns_mask=obs["board"]["valid_spawns_mask"]
        ),
        units=units,
        factories=factories,
        teams=teams

    )

@dataclass
class Board:
    rubble: np.ndarray
    ice: np.ndarray
    ore: np.ndarray
    lichen: np.ndarray
    lichen_strains: np.ndarray
    factory_occupancy_map: np.ndarray
    factories_per_team: int
    valid_spawns_mask: np.ndarray
@dataclass
class GameState:
    """
    A GameState object at step env_steps. Copied from luxai_s2/state/state.py
    """
    env_steps: int
    env_cfg: dict
    board: Board
    units: Dict[str, Dict[str, Unit]] = field(default_factory=dict)
    factories: Dict[str, Dict[str, Factory]] = field(default_factory=dict)
    teams: Dict[str, Team] = field(default_factory=dict)
    @property
    def real_env_steps(self):
        """
        the actual env step in the environment, which subtracts the time spent bidding and placing factories
        """
        if self.env_cfg.BIDDING_SYSTEM:
            # + 1 for extra factory placement and + 1 for bidding step
            return self.env_steps - (self.board.factories_per_team * 2 + 1)
        else:
            return self.env_steps


    # various utility functions
    def is_day(self):
        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH

