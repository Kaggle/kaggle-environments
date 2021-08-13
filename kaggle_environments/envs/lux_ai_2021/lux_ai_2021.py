import json
import math
import random
import sys
from os import path
from .agents import agents as all_agents
from subprocess import Popen, PIPE
import atexit
from .test_agents.python.lux.game import Game
from threading import Thread
from queue import Queue, Empty

t = None
q = None
dimension_process = None
game_state = Game()
def cleanup_dimensions():
    global dimension_process
    if dimension_process is not None:
        dimension_process.kill()
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()
def interpreter(state, env):
    global dimension_process, game_state, t, q
    player1 = state[0]
    player2 = state[1]

    ### 1.1: Initialize dimensions in the background within the orchestrator if we haven't already ###
    if dimension_process is None:
        # dimension_process = Popen(["ts-node", "-P", path.abspath(path.join(dir_path, "dimensions/tsconfig.json")), path.abspath(path.join(dir_path, "dimensions/run.ts"))], stdin=PIPE, stdout=PIPE)
        try:
            dimension_process = Popen(["node", path.abspath(path.join(dir_path, "dimensions/main.js"))], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except FileNotFoundError:
            import warnings
            warnings.warn("Node not installed")
            return state

        # following 4 lines from https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
        q = Queue()
        t = Thread(target=enqueue_output, args=(dimension_process.stdout, q))
        t.daemon = True # thread dies with the program
        t.start()
        atexit.register(cleanup_dimensions)

    ### TODO: check if process is still running, handle failure cases here

    ### 1.2: Initialize a blank state game if new episode is starting ###
    if env.done:
        if "seed" in env.configuration:
            seed = env.configuration["seed"]
        else:
            seed = math.floor(random.random() * 1e9);
            env.configuration["seed"] = seed
        if "loglevel" in env.configuration:
            loglevel = env.configuration["loglevel"]
        else:
            loglevel = 2 # warnings, 1: errors, 0: none
            env.configuration["loglevel"] = loglevel
        initiate = {
            "type": "start",
            "agent_names": [], # unsure if this is provided?
            "config": env.configuration
        }
        dimension_process.stdin.write((json.dumps(initiate) + "\n").encode())
        dimension_process.stdin.flush()
        agent1res = json.loads(dimension_process.stderr.readline())
        agent2res = json.loads(dimension_process.stderr.readline())
        match_obs_meta = json.loads(dimension_process.stderr.readline())
        
        player1.observation.player = 0
        player2.observation.player = 1
        player1.observation.updates = agent1res
        # player2.observation.updates = agent2res # duplicated and not added
        player1.observation.globalCityIDCount = match_obs_meta["globalCityIDCount"]
        player1.observation.globalUnitIDCount = match_obs_meta["globalUnitIDCount"]
        player1.observation.width = match_obs_meta["width"]
        player1.observation.height = match_obs_meta["height"]

        game_state = Game()
        game_state._initialize(agent1res)

        return state
    
    ### 2. : Pass in actions (json representation along with id of who made that action), agent information (id, status) to dimensions via stdin
    dimension_process.stdin.write((json.dumps(state) + "\n").encode())
    dimension_process.stdin.flush()


    ### 3.1 : Receive and parse the observations returned by dimensions via stdout
    agent1res = json.loads(dimension_process.stderr.readline())
    agent2res = json.loads(dimension_process.stderr.readline())
    game_state._update(agent1res)

    # receive meta info such as global ID and map sizes for purposes of being able to start from specific state
    match_obs_meta = json.loads(dimension_process.stderr.readline())
    match_status = json.loads(dimension_process.stderr.readline())

    while True:
        try:  line = q.get_nowait()
        except Empty:
            # no standard error received, break
            break
        else:
            # standard error output received, print it out
            print(line.decode(), file=sys.stderr, end='')

    ### 3.2 : Send observations to each agent through here. Like dimensions, first observation can include initialization stuff, then we do the looping

    player1.observation.updates = agent1res

    player1.observation.globalCityIDCount = match_obs_meta["globalCityIDCount"]
    player1.observation.globalUnitIDCount = match_obs_meta["globalUnitIDCount"]
    player1.observation.width = match_obs_meta["width"]
    player1.observation.height = match_obs_meta["height"]
    # player2.observation.updates = agent2res # duplicated and not added

    player1.observation.player = 0
    player2.observation.player = 1

    ### 3.3 : handle rewards
    # reward here is defined as the sum of number of city tiles
    player1.reward = compute_reward(game_state.players[0])
    player2.reward = compute_reward(game_state.players[1])
    player1.observation.reward = int(player1.reward)
    player2.observation.reward = int(player2.reward)

    ### 3.4 Handle finished match status
    if match_status["status"] == "finished":
        player1.status = "DONE"
        player2.status = "DONE"
    return state

def compute_reward(player):
    ct_count = sum([len(v.citytiles) for k, v in player.cities.items()])
    unit_count = len(game_state.players[player.team].units)
    # max board size is 32 x 32 => 1024 max city tiles and units, so this should keep it strictly so we break by city tiles then unit count
    return ct_count * 10000 + unit_count

def renderer(state, env):
    raise NotImplementedError("To render the replay, please set the render mode to json or html")


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "lux_ai_2021.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    html_path = path.abspath(path.join(dir_path, "index.html"))
    return ("html_path", html_path)

agents = all_agents
