from subprocess import Popen, PIPE
from threading  import Thread
from queue import Queue, Empty

import atexit
import os
import sys
agent_processes = [None, None, None, None]
t = None
q = None
def cleanup_process():
    global agent_processes
    for proc in agent_processes:
        if proc is not None:
            proc.kill()
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

def agent(observation, configuration):
    global agent_processes, t, q

    agent_process = agent_processes[observation.player]
    ### Do not edit ###
    if agent_process is None:
        if "__raw_path__" in configuration:
            cwd = os.path.dirname(configuration["__raw_path__"])
        else:
            cwd = os.path.dirname(__file__)
        agent_process = Popen(["node", "dist/Bot.js"], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd)
        agent_processes[observation.player] = agent_process
        atexit.register(cleanup_process)

        # following 4 lines from https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
        q = Queue()
        t = Thread(target=enqueue_output, args=(agent_process.stderr, q))
        t.daemon = True # thread dies with the program
        t.start()
    
    # print observations to agent
    import json
    agent_process.stdin.write((json.dumps(observation) + "\n").encode())
    agent_process.stdin.write((json.dumps(configuration) + "\n").encode())
    agent_process.stdin.flush()

    # wait for data written to stdout
    agent1res = (agent_process.stdout.readline()).decode()

    while True:
        try:  line = q.get_nowait()
        except Empty:
            # no standard error received, break
            break
        else:
            # standard error output received, print it out
            print(line.decode(), file=sys.stderr, end='')

    agent1res = agent1res.strip()
    outputs = agent1res.split(",")
    actions = {}
    for cmd in outputs:
        if cmd != "":
            shipyard_id, action_str = cmd.split(":")
            actions[shipyard_id] = action_str
    return actions
