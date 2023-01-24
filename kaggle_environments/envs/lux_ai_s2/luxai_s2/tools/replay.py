import os
import os.path as osp
from typing import List

from luxai_s2.replay import decode_replay_file
from luxai_s2.state import State


def replay_trajectory(replay_file: str):
    decoded = decode_replay_file(replay_file)
    if "observations" in decoded:
        print("replay file already contains observations, there is nothing to do")
        return replay_file
