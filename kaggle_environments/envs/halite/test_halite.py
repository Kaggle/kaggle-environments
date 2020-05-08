from kaggle_environments import make


EPISODE_STEPS = 10
STEPS = []


def step_appender_agent(obs, config):
    global STEPS
    STEPS.append(obs.step)
    return {}

env = make(
    "halite", configuration={"agentExec": "LOCAL", "episodeSteps": EPISODE_STEPS}, 
    debug=True
)

env.run({step_appender_agent})

print(f"Steps: {STEPS}")

assert STEPS == list(range(EPISODE_STEPS - 1))
