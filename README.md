# [<img src="https://kaggle.com/static/images/site-logo.png" height="50" style="margin-bottom:-15px" />](https://kaggle.com) Environments

```bash
pip install kaggle-environments
```

# TLDR;

```python
from kaggle_environments import make

# Setup a tictactoe environment.
env = make("tictactoe")

# Basic agent which marks the first available cell.
def my_agent(obs):
  return [c for c in range(len(obs.board)) if obs.board[c] == 0][0]

# Run the basic agent against a default agent which chooses a "random" move.
env.run([my_agent, "random"])

# Render an html ipython replay of the tictactoe game.
env.render(mode="ipython")
```

# Overview

Kaggle Environments was created to evaluate episodes. While other libraries have set interface precedents (such as Open.ai Gym), the emphasis of this library focuses on:

1. Episode evaluation (compared to training agents).
2. Configurable environment/agent lifecycles.
3. Simplified agent and environment creation.
4. Cross language compatible/transpilable syntax/interfaces.

## Help Documentation

```python
# Additional documentation (especially interfaces) can be found on all public functions:
from kaggle_environments import make
help(make)
env = make("tictactoe")
dir(env)
help(env.reset)
```

# Agents

> A function which given an observation generates an action.

## Writing

Agent functions can have observation and configuration parameters and must return a valid action. Details about the observation, configuration, and actions can seen by viewing the specification.

```python
from kaggle_simulations import make
env = make("connectx", {rows: 10, columns: 8, inarow: 5})

def agent(observation, configration):
  print(observation) # {board: [...], mark: 1}
  print(configuration) # {rows: 10, columns: 8, inarow: 5}
  return 3 # Action: always place a mark in the 3rd column.

# Run an episode using the agent above vs the default random agent.
env.run([agent, "random"])

# Print schemas from the specification.
print(env.specification.observation)
print(env.specification.configuration)
print(env.specification.action)
```

## Loading Agents

Agents are always functions, however there are some shorthand syntax options to make generating/using them easier.

```python
# Agent def accepting an observation and returning an action.
def agent1(obs):
  return [c for c in range(len(obs.board)) if obs.board[c] == 0][0]

# Load a default agent called "random".
agent2 = "random"

# Load an agent from source.
agent3 = """
def act(obs):
  return [c for c in range(len(obs.board)) if obs.board[c] == 0][0]
"""

# Load an agent from a file.
agent4 = "C:\path\file.py"

# Return a fixed action.
agent5 = 3

# Return an action from a url.
agent6 = "http://localhost:8000/run/agent"
```

## Default Agents

Most environments contain default agents to play against. To see the list of available agents for a specific environment run:

```python
from kaggle_simulations import make
env = make("tictactoe")

# The list of available default agents.
print(*env.agents)

# Run random agent vs reaction agent.
env.run(["random", "reaction"])
```

## Training

Open AI Gym interface is used to assist with training agents. The `None` keyword is used below to denote which agent to train (i.e. train as first or second player of connectx).

```python
from kaggle_environments import make

env = make("connectx", debug=True)

# Training agent in first position (player 1) against the default random agent.
trainer = env.train([None, "random"])

obs = trainer.reset()
for _ in range(100):
    env.render()
    action = 0 # Action for the agent being trained.
    obs, reward, done, info = trainer.step(action)
    if done:
        obs = trainer.reset()
```

## Debugging

There are 3 types of errors which can occur from agent execution:

1. **Timeout** - the agent runtime exceeded the allowed limit. There are 2 timeouts:
   1. `agentTimeout` - Used for initialization of an agent on first "act".
   2. `actTimeout` - Used for obtaining an action.
2. **Error** - the agent raised and error during execution.
3. **Invalid** - the agent action response didn't match the action specification or the environment deemed it invalid (i.e. playing twice in the same cell in tictactoe).

To help debug your agent and why it threw the errors above, add the `debug` flag when setting up the environment.

```python
from kaggle_simulations import make

def agent():
  return "Something Bad"

env = make("tictactoe", debug=True)

env.run([agent, "random"])
# Prints: "Invalid Action: Something Bad"
```

# Environments

> A function which given a state and agent actions generates a new state.

| Name      | Description                          | Make                      |
| --------- | ------------------------------------ | ------------------------- |
| connectx  | Connect 4 in a row but configurable. | `env = make("connectx")`  |
| tictactoe | Classic Tic Tac Toe                  | `env = make("tictactoe")` |
| identity  | For debugging, action is the reward. | `env = make("identity")`  |

## Making

An environment instance can be made from an existing specification (such as those listed above).

```python
from kaggle_environments import make

# Create an environment instance.
env = make(
  # Specification or name to registered specification.
  "connectx",

  # Override default and environment configuration.
  configuration={"rows": 9, "columns": 10},

  # Initialize the environment from a prior state (episode resume).
  steps=[],

  # Enable verbose logging.
  debug=True
)
```

## Configuration

There are two types of configuration: Defaults applying to every environment and those specific to the environment. The following is a list of the default configuration:

| Name         | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| episodeSteps | Maximum number of steps in the episode.                         |
| agentExec    | How the agent is executed alongside the envionment.             |
| agentTimeout | Maximum runtime (seconds) to initialize an agent.               |
| actTimeout   | Maximum runtime (seconds) to obtain an action from an agent.    |
| runTimeout   | Maximum runtime (seconds) of an episode (not necessarily DONE). |

```python
env = make("connectx", configuration={
  "columns": 19, # Specific to ConnectX.
  "actTimeout": 10,
  "agentExec": "LOCAL"
})
```

## Resetting

Environments are reset by default after "make" (unless starting steps are passed in) as well as when calling "run". Reset can be called at anytime to clear the environment.

```python
num_agents = 2
reset_state = env.reset(num_agents)
```

## Running

Execute an episode against the environment using the passed in agents until they are no longer running (i.e. status != ACTIVE).

```python
steps = env.run([agent1, agent2])
print(steps)
```

## Evaluating

Evaluation is used to run an episode (environment + agents) multiple times and just return the rewards.

```python
from kaggle_simulations import evaluate

# Same definitions as "make" above.
envrionment = "connectx"
configuration = {rows: 10, columns: 8, inarow: 5}
steps = []
debug = False

# Which agents to run repeatedly.  Same as env.run(agents)
agents = ["random", agent1]

# How many times to run them.
num_episodes = 10

rewards = evaluate(environment, agents, configuration, steps, num_episodes, debug)
```

## Stepping

Running above essentially just steps until no agent is still active. To execute a singular game loop, pass in actions directly for each agent. Note that this is normally used for training agents (most useful in a single agent setup such as using the gym interface).

```python
agent1_action = agent1(env.state[0].observation)
agent2_action = agent2(env.state[1].observation)
state = env.step(agent1_action, agent2_action)
```

## Playing

A few environments offer an interactive play against agents within jupyter notebooks. An example of this is using connectx:

```python
from kaggle_simulations import make

env = make("connectx")
# None indicates which agent will be manually played.
env.play([None, "random"])
```

## Rendering

The following rendering modes are supported:

- json - Same as doing a json dump of `env.toJSON()`
- ansi - Ascii character representation of the environment.
- human - ansi just printed to stdout
- html - HTML player representation of the environment.
- ipython - html just printed to the output of a ipython notebook.

```python
out = env.render(mode="ansi")
print(out)
```

# Command Line

```sh
> python main.py -h
```

## List Registered Environments

```sh
> python main.py list
```

## Evaluate Episode Rewards

```sh
python main.py evaluate --environment tictactoe --agents random random --episodes 10
```

## Run an Episode

```sh
> python main.py run --environment tictactoe --agents random /pathtomy/agent.py --debug True
```

## Load an Episode

This is useful when converting an episode json output into html.

```sh
python main.py load --environment tictactoe --steps [...] --render '{"mode": "html"}'
```

# HTTP Server

The HTTP server contains the same interface/actions as the CLI above merging both POST body and GET params.

## Setup

```bash
python main.py http-server --port=8012 --host=0.0.0.0
```

## Adding Middleware

```python
# middleware.py
import time

def request(req):
    time.sleep(30)
    req.agents = ["random", "random"]
    return req

def response(req, resp):
    time.sleep(10)
    return resp
```

```bash
python3 main.py http-server --middleware=/path/to/middleware.py
```

### Running Agents on Separate Servers

```python
# How to run agent on a separate server.
import requests
import json

path_to_agent1 = "/home/ajeffries/git/playground/agent1.py"
path_to_agent2 = "/home/ajeffries/git/playground/agent2.py"

agent1_url = f"http://localhost:5001?agents[]={path_to_agent1}"
agent2_url = f"http://localhost:5002?agents[]={path_to_agent2}"

body = {
    "action": "run",
    "environment": "tictactoe",
    "agents": [agent1_url, agent2_url]
}
resp = requests.post(url="http://localhost:5000", data=json.dumps(body)).json()

# Inflate the response replay to visualize.
from kaggle_environments import make
env = make("tictactoe", steps=resp["steps"], debug=True)
env.render(mode="ipython")
print(resp)
```
