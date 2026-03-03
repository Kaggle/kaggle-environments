import json
import os
from kaggle_environments import make

env = make("codenames", debug=True)
env.run(["random", "random", "random", "random"])

replay_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer", "default", "replays")
os.makedirs(replay_dir, exist_ok=True)
replay_path = os.path.join(replay_dir, "test-replay.json")

with open(replay_path, "w") as f:
    json.dump(env.toJSON(), f)

print(f"Generated {replay_path}")
