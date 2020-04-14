import importlib
import json
from os import path
import numpy as np

def renderer(state, env):
  return "inline rendering not supported."

def run_right_agent(obs):
  # keep running right.
  return [5] * obs.controlled_players

def run_left_agent(obs):
  return [1] * obs.controlled_players

def do_nothing_agent(obs):
  return [0] * obs.controlled_players

agents = {"run_right": run_right_agent, "run_left": run_left_agent,
          "do_nothing": do_nothing_agent}


def update_observations_and_rewards(configuration, state, obs, rew=None):
  state[0].observation.controlled_players = configuration.team_1
  state[1].observation.controlled_players = configuration.team_2

  if not configuration.team_2:
    # Single agent setup.
    state[0].observation.minimap = obs.flatten().tolist()
    state[1].observation.minimap = ([0]*72*96*4)
    if rew is not None:
      state[0].reward = rew.item()
      state[1].reward = (-rew).item()
  else:
    # Two agent setup.
    for agent in range(2):
      state[agent].observation.minimap = obs[agent].flatten().tolist()
      if rew is not None:
        state[agent].reward = rew[agent].item()


def update_state_on_invalid_action(bad_agent, good_agent, message):
  bad_agent.status = "INVALID"
  bad_agent.reward = -100
  bad_agent.info.debug_info = message

  good_agent.status = "DONE"
  good_agent.reward = 100
  good_agent.info.debug_info = "Opponent made invalid move. You win."

def football_env():
  # Use lazy-import to avoid this heavy dependency unless it is really needed.
  return importlib.import_module("gfootball.env")

# Global dictionary with active environments.
m_envs = {}

def interpreter(state, env):
  global m_envs
  if (env.id not in m_envs) or env.done:
    print("Staring a new environment %s: with scenario: %s" % (env.id, env.configuration.scenario_name))
    m_envs[env.id] = football_env().create_environment(env_name=env.configuration.scenario_name,
                                            stacked=False,
                                            logdir='/tmp/football',
                                            write_goal_dumps=False,
                                            write_full_episode_dumps=False,
                                            render=False,
                                            number_of_left_players_agent_controls=env.configuration.team_1,
                                            number_of_right_players_agent_controls=env.configuration.team_2)
    obs = m_envs[env.id].reset()
    update_observations_and_rewards(configuration=env.configuration, state=state, obs=obs)

  if env.done:
    return state

  # Check if both players responded.
  for agent in range(2):
    # TODO: it seems that ACTIVE/INACTIVE/DONE are not present in 'status_codes.json'
    # not sure what are the correct values here.
    if (state[agent].status != "OK" and state[agent].status != "ACTIVE"):
      # something went wrong.
      print("AGENT %d returned invalid state: %s" % (agent, state[agent].status))
      return state

  # verify actions.
  controlled_players = env.configuration.team_1

  if len(state[0].action) != env.configuration.team_1:
    # Player 1 sent wrong data.
    update_state_on_invalid_action(state[0], state[1],
                                   "Too many actions passed: Expected %d, got %d." %
                                   (env.configuration.team_1, len(state[0].action)))
    return state
  actions_to_env = state[0].action

  if len(state[1].action) != env.configuration.team_2:
    # Player 2 sent wrong data.
    update_state_on_invalid_action(state[1], state[0],
                                   "Too many actions passed: Expected %d, got %d." %
                                   (env.configuration.team_2, len(state[1].action)))
    return state

  if env.configuration.team_2:
    actions_to_env = actions_to_env + state[1].action

  obs, rew, done, info = m_envs[env.id].step(actions_to_env)

  update_observations_and_rewards(configuration=env.configuration, state=state, obs=obs, rew=rew)

  ## TODO: pass other information from 'info' to the state/agent.
  if done:
    for agent in range(2):
      state[agent].status = "INACTIVE"

  return state


dirpath = path.dirname(__file__)
jsonpath = path.abspath(path.join(dirpath, "football.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    jspath = path.abspath(path.join(dirpath, "football.js"))
    with open(jspath) as f:
        return f.read()
