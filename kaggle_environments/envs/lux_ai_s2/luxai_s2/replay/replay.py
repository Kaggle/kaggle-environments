import json
import os
import os.path as osp
from typing import List, Tuple

from luxai_s2.state import State


def decode_replay_file(replay_file) -> Tuple[State, List]:
    """
    Takes an input replay file of any kind from any source and extracts the full trajectory with observations
    or just initial state and actions if observations aren't there
    """
    ext = osp.splitext(replay_file)[-1]
    if ext == ".json":
        # probably kaggle replay
        with open(replay_file, "r") as f:
            replay = json.load(f)
        init_state = replay["init_state"]
        pass
    elif ext == ".h5":
        pass


def generate_replay(states: List[State]):
    """
    Generates a compressed replay.

    """
    return [s.get_compressed_obs() for s in states]
