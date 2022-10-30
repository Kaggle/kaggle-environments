from typing import List
import numpy as np
from luxai2022.config import EnvConfig
from luxai2022.state.state import State
from luxai2022.unit import Unit
def generate_weather_schedule(rng: np.random.RandomState, cfg: EnvConfig):
    # randomly generate 3-5 events, each lasting 20 turns
    # no event can overlap another
    num_events = rng.randint(cfg.NUM_WEATHER_EVENTS_RANGE[0], cfg.NUM_WEATHER_EVENTS_RANGE[1] + 1)
    last_event_end_step = 0
    # TODO - make a smarter algorithm to speed up the generation here
    available_times = set(list(range(cfg.max_episode_length - 30)))
    schedule = np.zeros(cfg.max_episode_length, dtype=int)
    for i in range(num_events):
        weather_id = rng.randint(1, len(cfg.WEATHER_ID_TO_NAME))
        weather = cfg.WEATHER_ID_TO_NAME[weather_id]
        weather_cfg = cfg.WEATHER[weather]
        weather_time_range = weather_cfg["TIME_RANGE"]
        # use rejection sampling
        while True:
            start_time = rng.randint(0, cfg.max_episode_length - 20)
            duration = rng.randint(weather_time_range[0], weather_time_range[1] + 1)
            requested_times = set(list(range(start_time, duration)))
            if requested_times.issubset(available_times):
                available_times.difference_update(requested_times)
                schedule[start_time: start_time + duration] = weather_id
                break
    return schedule
    

def apply_weather(state: State, agents: List[str], current_weather):
    if current_weather == "MARS_QUAKE":
        for agent in agents:
            for unit in state.units[agent].values():
                unit: Unit
                old_rubble = state.board.rubble[unit.pos.y, unit.pos.x]  
                state.board.rubble[unit.pos.y, unit.pos.x] = min(state.env_cfg.MAX_RUBBLE, old_rubble + state.env_cfg.ROBOTS[unit.unit_type.name].RUBBLE_AFTER_DESTRUCTION)
        return dict(power_gain_factor=1, power_loss_factor=1)
    elif current_weather == "COLD_SNAP":
        return dict(power_gain_factor=1, power_loss_factor=state.env_cfg.WEATHER["COLD_SNAP"]["POWER_CONSUMPTION"])
    elif current_weather == "DUST_STORM":
        return dict(power_gain_factor=state.env_cfg.WEATHER["DUST_STORM"]["POWER_GAIN"], power_loss_factor=1)
    elif current_weather == "SOLAR_FLARE":
        return dict(power_gain_factor=state.env_cfg.WEATHER["SOLAR_FLARE"]["POWER_GAIN"], power_loss_factor=1)
    return dict(power_gain_factor=1, power_loss_factor=1)