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
def to_json(state):
    if isinstance(state, np.ndarray):
        return state.tolist()
    elif isinstance(state, np.int64):
        return state.tolist()
    elif isinstance(state, list):
        return [to_json(s) for s in state]
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = to_json(state[k])
        return out
    else:
        return state  
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
        for item in ["rubble", "lichen", "lichen_strains"]:
            for k, v in obs["board"][item].items():
                k = k.split(",")
                x, y = int(k[0]), int(k[1])
                game_state["board"][item][y, x] = v
    return game_state

def obs_to_game_state(step, env_cfg: EnvConfig, obs):
    
    units = dict()
    for agent in obs["units"]:
        units[agent] = dict()
        for unit_id in obs["units"][agent]:
            unit_data = obs["units"][agent][unit_id]
            cargo = UnitCargo(**unit_data["cargo"])
            del unit_data["cargo"]
            units[agent][unit_id] = Unit(
                **unit_data,
                cargo=cargo,
                unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]],
                env_cfg=env_cfg
            )

    factory_occupancy_map = np.ones_like(obs["board"]["rubble"], dtype=int) * -1
    factories = dict()
    for agent in obs["factories"]:
        factories[agent] = dict()
        for unit_id in obs["factories"][agent]:
            f_data = obs["factories"][agent][unit_id]
            cargo = UnitCargo(**f_data["cargo"])
            del f_data["cargo"]
            factory = Factory(
                **f_data,
                cargo=cargo,
                env_cfg=env_cfg
            )
            factories[agent][unit_id] = factory
            factory_occupancy_map[factory.pos[1] - 1:factory.pos[1] + 1, factory.pos[0] - 1:factory.pos[0] + 1] = factory.team_id
    teams = dict()
    for agent in obs["teams"]:
        team_data = obs["teams"][agent]
        faction = FactionTypes[team_data["faction"]]
        del team_data["faction"]
        teams[agent] = Team(**team_data, faction=faction, agent=agent)

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
            factories_per_team=obs["board"]["factories_per_team"]
        ),
        weather_schedule=obs["weather_schedule"],
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
@dataclass
class GameState:
    """
    A GameState object at step env_steps. Copied from luxai2022/state/state.py
    """
    env_steps: int
    env_cfg: dict
    board: Board
    weather_schedule: np.ndarray = None
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
            return self.env_steps - (self.board.factories_per_team + 1 + 1)
        else:
            return self.env_steps


    # various utility functions
    def is_day(self):
        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH