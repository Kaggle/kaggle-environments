import os
from typing import List
from luxai2022.replay import decode_replay_file
from luxai2022.state import State
import os.path as osp
def replay_trajectory(replay_file: str):
    decoded = decode_replay_file(replay_file)
    if "observations" in decoded:
        print("replay file already contains observations, there is nothing to do")
        return replay_file
    

