# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module
from os import listdir
from .agent import Agent
from .api import get_episode_replay, list_episodes, list_episodes_for_team, list_episodes_for_submission
from .core import *
from .main import http_request
from . import errors

__version__ = "1.17.5"

__all__ = ["Agent", "environments", "errors", "evaluate", "http_request",
           "make", "register", "utils", "__version__",
           "get_episode_replay", "list_episodes", "list_episodes_for_team", "list_episodes_for_submission"]

# Register Environments.

for name in listdir(utils.envs_path):
    try:
        env = import_module(f".envs.{name}.{name}", __name__)
        if name == "open_spiel":
            for env_name, env_dict in env.ENV_REGISTRY.items():
                register(env_name, {
                    "agents": env_dict.get("agents"),
                    "html_renderer": env_dict.get("html_renderer"),
                    "interpreter": env_dict.get("interpreter"),
                    "renderer": env_dict.get("renderer"),
                    "specification": env_dict.get("specification"),
                })
        else:
          register(name, {
              "agents": getattr(env, "agents", []),
              "html_renderer": getattr(env, "html_renderer", None),
              "interpreter": getattr(env, "interpreter"),
              "renderer": getattr(env, "renderer"),
              "specification": getattr(env, "specification"),
          })
    except Exception as e:
        if "football" not in name:
            print("Loading environment %s failed: %s" % (name, e))
