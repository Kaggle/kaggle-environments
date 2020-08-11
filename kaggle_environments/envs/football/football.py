import importlib
import json
from os import path
import numpy as np
import uuid


def renderer(state, env):
    html_renderer(env)


def run_right_agent(obs):
    # keep running right.
    return [5] * obs.controlled_players


def run_left_agent(obs):
    # keep running left.
    return [1] * obs.controlled_players


def do_nothing_agent(obs):
    # do nothing.
    return [0] * obs.controlled_players


def builtin_ai_agent(obs):
    # execute builtin AI behavior.
    return [19] * obs.controlled_players


agents = {
    "run_right": run_right_agent,
    "run_left": run_left_agent,
    "do_nothing": do_nothing_agent,
    "builtin_ai": builtin_ai_agent,
}


def parse_single_player(obs_raw_entry):
    # Remove pixel information.
    if "frame" in obs_raw_entry:
        del obs_raw_entry["frame"]
    for k, v in obs_raw_entry.items():
        if type(v) == np.ndarray:
            obs_raw_entry[k] = v.tolist()
    return obs_raw_entry


def update_observations_and_rewards(configuration, state, obs, rew=None):
    """Updates agent-visible observations given 'raw' observations from environment.

    Observations in 'obs' are coming directly from the environment and are in 'raw' format.
    """
    state[0].observation.controlled_players = configuration.team_1
    state[1].observation.controlled_players = configuration.team_2

    assert len(obs) == configuration.team_1 + configuration.team_2
    if rew is not None:
        if type(rew) == np.float32:
            rew = [rew]
        assert len(rew) == configuration.team_1 + configuration.team_2
        if configuration.team_1 > 0:
            state[0].reward = float(rew[0])
        else:
            state[0].reward = float(-rew[0])
        state[1].reward = -state[1].reward

    state[0].observation.players_raw = [
        parse_single_player(obs[x]) for x in range(configuration.team_1)
    ]
    state[1].observation.players_raw = [
        parse_single_player(obs[x + configuration.team_1])
        for x in range(configuration.team_2)
    ]


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
    global m_envs
    del m_envs[env.configuration.id]


def cleanup_all():
    global m_envs
    del m_envs


def retrieve_video_link(dumps):
    for entry in dumps:
        if entry['name'] == 'episode_done':
            print("Received video link.")
            return entry['video']
    return None


def interpreter(state, env):
    global m_envs
    if "id" not in env.configuration or env.configuration.id is None:
        env.configuration.id = str(uuid.uuid4())

    if (env.configuration.id not in m_envs) or env.done:
        if env.configuration.id not in m_envs:
            print("Staring a new environment %s: with scenario: %s" %
                  (env.configuration.id, env.configuration.scenario_name))

            other_config_options = {}
            if env.configuration.running_in_notebook:
                # Use webp to encode videos (so that you can see them in the browser).
                other_config_options["video_format"] = "webm"
                assert not env.configuration.render, "Render is not supported inside notebook environment."

            env.football_video_path = None
            if 'agents' in env.info:
                names = env.info['agents']
                assert len(names) == 2
                other_config_options['custom_display_stats'] = [
                    'LEFT PLAYER: %s' % names[0]['teamName'],
                    'RIGHT PLAYER: %s' % names[1]['teamName']]
            m_envs[env.configuration.id] = football_env().create_environment(
                env_name=env.configuration.scenario_name,
                stacked=False,
                # We use 'raw' representation to transfer data between server and agents.
                representation='raw',
                logdir=path.join(env.configuration.logdir, env.configuration.id),
                write_goal_dumps=False,
                write_full_episode_dumps=env.configuration.save_video,
                write_video=env.configuration.save_video,
                render=env.configuration.render,
                number_of_left_players_agent_controls=env.configuration.team_1,
                number_of_right_players_agent_controls=env.configuration.team_2,
                other_config_options=other_config_options)
        else:
            print("Resetting environment %s: with scenario: %s" %
                  (env.configuration.id, env.configuration.scenario_name))
        obs = m_envs[env.configuration.id].reset()
        update_observations_and_rewards(configuration=env.configuration,
                                        state=state,
                                        obs=obs)

    if env.done:
        return state

    # Check if both players responded.
    for agent in range(2):
        # TODO: it seems that ACTIVE/INACTIVE/DONE are not present in 'status_codes.json'
        # not sure what are the correct values here.
        if (state[agent].status != "OK" and state[agent].status != "ACTIVE"):
            # something went wrong.
            print("AGENT %d returned invalid state: %s" %
                  (agent, state[agent].status))
            return state

    # verify actions.
    controlled_players = env.configuration.team_1

    if len(state[0].action) != env.configuration.team_1:
        # Player 1 sent wrong data.
        update_state_on_invalid_action(
            state[0], state[1], "Invalid number of actions provided: Expected %d, got %d." %
                                (env.configuration.team_1, len(state[0].action)))
        return state
    actions_to_env = state[0].action

    if len(state[1].action) != env.configuration.team_2:
        # Player 2 sent wrong data.
        update_state_on_invalid_action(
            state[1], state[0], "Invalid number of actions provided: Expected %d, got %d." %
                                (env.configuration.team_2, len(state[1].action)))
        return state

    if env.configuration.team_2:
        actions_to_env = actions_to_env + state[1].action

    obs, rew, done, info = m_envs[env.configuration.id].step(actions_to_env)

    if "dumps" in info:
        env.football_video_path = retrieve_video_link(info["dumps"])

    update_observations_and_rewards(configuration=env.configuration,
                                    state=state,
                                    obs=obs,
                                    rew=rew)

    ## TODO: pass other information from 'info' to the state/agent.
    if done:
        for agent in range(2):
            state[agent].status = "INACTIVE"

    return state


dirpath = path.dirname(__file__)
jsonpath = path.abspath(path.join(dirpath, "football.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer(env):
    if not env.football_video_path:
        trace = m_envs[env.configuration.id].env._env._trace
        trace._dump_config['episode_done']._min_frequency = 0
        dumps = trace.process_pending_dumps(True)
        env.football_video_path = retrieve_video_link(dumps)
        if not env.football_video_path:
            raise Exception(
                "No video found. Was environment created with save_video enabled?"
            )
        trace.write_dump('episode_done')

    from IPython.display import display, HTML
    from base64 import b64encode

    video = open(env.football_video_path, 'rb').read()
    env.football_video_path = None
    data_url = "data:video/webm;base64," + b64encode(video).decode()

    html = """
<video width=800 controls>
  <source src="%s" type="video/webm">
</video>
""" % data_url
    display(HTML(html))
    return ""
