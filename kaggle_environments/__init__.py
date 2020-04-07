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
from .core import *
from . import errors
from . import utils

version = "0.2.0"

__all__ = ["Agent", "environments", "errors", "evaluate",
           "make", "register", "utils", "version"]

# Register Environments.

for name in listdir(utils.envs_path):
    try:
        env = import_module(f".envs.{name}.{name}", __name__)
        register(name, {
            "agents": getattr(env, "agents", []),
            "html_renderer": getattr(env, "html_renderer", None),
            "interpreter": getattr(env, "interpreter"),
            "renderer": getattr(env, "renderer"),
            "specification": getattr(env, "specification"),
        })
    except:
        pass
