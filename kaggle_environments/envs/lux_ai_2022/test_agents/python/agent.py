import json
import sys
from typing import Dict
if __package__ == "":
    from lux.config import EnvConfig
    from lux.kit import GameState, process_obs, to_json, from_json, process_action, obs_to_game_state
else:
    from .lux.config import EnvConfig
    from .lux.kit import GameState, process_obs, to_json, from_json, process_action, obs_to_game_state
import numpy as np
class Agent():
    def __init__(self, player: str) -> None:
        self.player = player
        self.opp_player = ""
        self.game_state: GameState = None
        if self.player == "player_0":
            self.opp_player = "player_1"
        else:
            self.opp_player = "player_0"
        np.random.seed(0)
        self.step = -1
        self.factories_owned = 0
        self.init_metal_left = 0
        self.init_water_left = 0
        self.env_cfg: EnvConfig = None

    def early_setup(self, step, obs, remainingOverageTime: int):
        """
        Logic here to make actions in the early game. Select faction, bid for an extra factory, and place factories
        """
        # various maps to help aid in decision making over factory placement
        rubble = obs["board"]["rubble"]
        # if ice[y][x] > 0, then there is an ice tile at (x, y)
        ice = obs["board"]["ice"]
        # if ore[y][x] > 0, then there is an ore tile at (x, y)
        ore = obs["board"]["ore"]

        env_cfg = self.env_cfg # the current env configuration for the episode
        
        if step == 0:
            # decide on a faction, and make a bid for the extra factory. 
            # Each unit of bid removes one unit of water and metal from your initial pool
            faction = "MotherMars"
            if self.player == "player_1":
                faction = "AlphaStrike"
            return dict(faction=faction, bid=10)
        else:
            # decide on where to spawn the next factory. Returning an empty dict() will skip your factory placement

            # how much water and metal you have in your starting pool to give to new factories
            water_left = obs["teams"][self.player]["water"]
            metal_left = obs["teams"][self.player]["metal"]
            # how many factories you have left to place
            factories_to_place = obs["teams"][self.player]["factories_to_place"]
            if step == 1:
                # first step of factory placement, we save our initial pool of water and factory amount
                self.factories_owned = factories_to_place
                self.init_metal_left = metal_left
                self.init_water_left = water_left
            # obs["teams"][self.opp_player] has the same information but for the other team
            # potential spawnable locations in your half of the map
            potential_spawns = obs["board"]["spawns"][self.player]

            # as a naive approach we randomly select a spawn location and spawn a factory there
            spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
            return dict(spawn=spawn_loc, metal=self.init_metal_left // self.factories_owned, water=self.init_water_left // self.factories_owned)

    def act(self, step: int, obs: Dict, remainingOverageTime: int):
        """
        Logic here to make actions for the rest of the game.

        Parameters
        ----------
        step - the current environment step. This is the "real" time step, starts from 0
            after all the bidding and factory placement is done.

        obs - the env observation. This is the raw dictionary based observation and it's usage is detailed below.
            The more raw format is suitable for those who may need it for ML based solutions. For programmed solutions
            an API is provided that interfaces with the observation more nicely, in addition to providing
            functions to generate action vectors

        remainingOverageTime - amount of time in seconds left to make an action in addition to the 
            2s timer per action. Time spent > 2s eats into the remainingOverageTime pool for next turn


        NOTE that step is set back to 0 in this part of the code. This act function is executed max_episode_length times.
        The code ignores the time spent bidding and placing factories
        """
        actions = dict()

        env_cfg: EnvConfig = self.env_cfg # the current env configuration for the episode
        game_state = obs_to_game_state(step=self.step, env_cfg=env_cfg, obs=obs)

        # the weather schedule, a sequence of values representing what the weather is at each real time step
        # 0 = Nothing, 1 = MARS_QUAKE, 2 = COLD_SNAP, 3 = DUST_STORM, 4 = SOLAR_FLARE
        weather_schedule = game_state.weather_schedule
        current_weather = game_state.weather_schedule[step]

        # various maps to help aid in decision making
        rubble = game_state.board.rubble

        # if ice[y][x] > 0, then there is an ice tile at (x, y)
        ice = game_state.board.ice
        # if ore[y][x] > 0, then there is an ore tile at (x, y)
        ore = game_state.board.ore

        # lichen[y][x] = amount of lichen at tile (x, y)
        lichen = game_state.board.lichen
        # lichenStrains[y][x] = the strain id of the lichen at tile (x, y). Each strain id is
        # associated with a single factory and cannot mix with other strains. 
        # factory.strain_id defines the factory's strain id
        lichen_strains = game_state.board.lichen_strains

        # units and factories for your team and the opposition team
        units = game_state.units[self.player]
        opp_units = game_state.units[self.opp_player]
        factories = game_state.factories[self.player]
        opp_factories = game_state.factories[self.opp_player]
        
        # iterate over all active factories
        for unit_id, factory in factories.items():
            if step % 4 == 0 and step > 1:
                p = np.random.uniform()
                if p < 0.75:
                    if factory.can_build_light(game_state):
                        # check if factory can build a light unit, and submit the action if so
                        actions[unit_id] = factory.build_light()
                else:
                    if factory.can_build_heavy(game_state):
                        actions[unit_id] = factory.build_heavy()
            else:
                # find the water cost to grow lichen and only water the lichen if we have enough water
                water_cost = factory.water_cost(game_state)
                if factory.cargo.water > water_cost + 10:
                    actions[unit_id] = factory.water()
                
        for unit_id, unit in units.items():
            # (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_dir = np.random.randint(0, 5)
            move_cost = unit.move_cost(game_state, move_dir)
            # we attempt to move if move_cost is not None (valid movement) and we have enough power to submit a new action queue and move
            if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                # by default, any unit action has repeat=True and moves completed actions back to the end of the action queue
                # You can also queue up to env_cfg.UNIT_ACTION_QUEUE_SIZE *actions* for each unit.
                # You can submit action queues for every unit, but be wary that each *action queue* submission costs an additional
                # env_cfg.UNIT_ACTION_QUEUE_POWER_COST[unit.unit_type] power

                # here, we tell this unit to move in a direction once and stay still until controlled again
                actions[unit_id] = [unit.move(move_dir, repeat=False)]
        return actions

### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict() # store potentially multiple dictionaries as kaggle imports code directly
def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step
    
    
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = Agent(player)
        agent = agent_dict[player]
        # TODO verify this works with kaggle input and add logic to fix that
        agent.env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
    agent = agent_dict[player]
    new_game_state = process_obs(player, agent.game_state, step, json.loads(observation.obs))
    agent.game_state = new_game_state
    agent.step = step
    if step <= agent.game_state["board"]["factories_per_team"] + 1:
        actions = agent.early_setup(step, new_game_state, remainingOverageTime)
    else:
        real_env_steps = new_game_state["real_env_steps"]
        actions = agent.act(real_env_steps, new_game_state, remainingOverageTime)

    return process_action(actions)
