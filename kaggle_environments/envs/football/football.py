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

  ## TODO: this has to be re-done.
  # the shapes are:
  ## if you control a single player (configuration.team_1 + configuration.team_2 == 1): (72, 96, 4)
  ## else the shape is: (team_1+team2, 72, 96, 4)

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
      ## TODO: this is not correct
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

def cleanup(env):
  # TODO: find a way to expose it in the main Kaggle API.
  # Some things (like video rendering) happen only after the environment is marked as finished.
  global m_envs
  del m_envs[env.id]

def cleanup_all():
  global m_envs
  del m_envs


def get_video_path(env):
  ## HACK: we should expose this value inside the football env
  if not hasattr(env, "video_path"):
    suffix = '.webm' if env.configuration.running_in_notebook else '.avi'
    names = m_envs[env.id].unwrapped._env._trace._dump_config['episode_done']._dump_names
    if names:
      env.video_path = names[-1] + suffix
      ## HACK: video is only written if environment is closed.
      m_envs[env.id].unwrapped._env.close()
    
  return env.video_path
  

def interpreter(state, env):
  global m_envs
  if (env.id not in m_envs) or env.done:
    if env.id not in m_envs:
      print("Staring a new environment %s: with scenario: %s" % (env.id, env.configuration.scenario_name))

      other_config_options = {}

      if env.configuration.running_in_notebook:
        # Use webp to encode videos (so that you can see them in the browser).
        other_config_options["video_format"] = "webm"
        assert not env.configuration.render, "Render is not supported inside notebook environment."

      m_envs[env.id] = football_env().create_environment(env_name=env.configuration.scenario_name,
                                              stacked=False,
                                              logdir=path.join(env.configuration.logdir, env.id),
                                              write_goal_dumps=False,
                                              write_full_episode_dumps=env.configuration.save_video,
                                              write_video=env.configuration.save_video,
                                              render=env.configuration.render,
                                              number_of_left_players_agent_controls=env.configuration.team_1,
                                              number_of_right_players_agent_controls=env.configuration.team_2,
                                              other_config_options=other_config_options)
    else:
      print("Resetting environment %s: with scenario: %s" % (env.id, env.configuration.scenario_name))
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

def render_ipython(env):
	from IPython.display import display, HTML
	from base64 import b64encode

	video = open(get_video_path(env), 'rb').read()
	data_url = "data:video/webm;base64," + b64encode(video).decode()

	display(HTML("""
<video width=800 controls>
  <source src="%s" type="video/webm">
</video>
""" % data_url)) 
