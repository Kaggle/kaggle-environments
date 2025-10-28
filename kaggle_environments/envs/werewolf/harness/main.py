"""Main file for the Game Arena submission."""

import os
import random
import sys

_AGENT_OBJECT = None
_SETUP_COMPLETE = False
_TELEMETRY = None


def agent(observation, configuration):
    """Kaggle agent for Game Arena."""
    global _AGENT_OBJECT, _SETUP_COMPLETE, _TELEMETRY

    if not _SETUP_COMPLETE:
        print("--- Performing one-time agent setup... ---")

        # 1. Add the vendored 'lib' directory to Python's search path.
        print("Updating system path with vendored libraries...")
        script_dir = os.path.dirname(configuration["__raw_path__"])
        lib_dir = os.path.join(script_dir, "lib")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        print(f"System path updated. First entry is now: {sys.path[0]}")

        # 2. Now that the path is set, we can import our libraries.
        # pylint: disable=g-import-not-at-top

        from kaggle_environments.envs.werewolf.werewolf import AgentFactoryWrapper, LLM_SYSTEM_PROMPT
        from kaggle_environments.envs.werewolf.harness.base import LLMWerewolfAgent

        from game_arena.harness import telemetry
        from game_arena.google import model_proxy_telemetry
        print("Successfully imported game_arena modules.")
        print("Setting up telemetry...")
        telemetry.set_exporter(model_proxy_telemetry.send)
        _TELEMETRY = telemetry.get_logger(__name__)

        if "MODEL_ENUM" not in globals():
            raise ValueError("MODEL_ENUM was not injected. Agent cannot run.")
        model_enum = globals()["MODEL_ENUM"]  # pylint: disable=invalid-name
        print(f"MODEL_ENUM is {model_enum}")

        chosen_key = None
        if "MODEL_PROXY_KEY" in os.environ:
            chosen_key = os.environ["MODEL_PROXY_KEY"]
            print("API key found in environment.")
        if chosen_key is None:
            if "MODEL_API_KEY" not in globals():
                raise ValueError("MODEL_API_KEY was not injected. Agent cannot run.")
            model_api_key = globals()["MODEL_API_KEY"]
            print("API Key found in globals.")
            # If comma separated, choose one at random.
            all_keys = model_api_key.split(",")
            all_keys = [key.strip() for key in all_keys]
            all_keys = [key for key in all_keys if key]
            chosen_key = random.choice(all_keys)
        if "MODEL_PROXY_URL" not in os.environ:
            raise ValueError("MODEL_PROXY_URL was not injected. Agent cannot run.")

        _AGENT_OBJECT = AgentFactoryWrapper(
            agent_class=LLMWerewolfAgent,
            model_name=model_enum,
            system_prompt=LLM_SYSTEM_PROMPT,
            litellm_model_proxy_kwargs={
                "api_base": os.environ["MODEL_PROXY_URL"],
                "api_key": chosen_key
            }
        )

        _SETUP_COMPLETE = True
        _TELEMETRY(setup_complete=True)
        print("--- Agent setup complete. ---")

    return _AGENT_OBJECT(observation, configuration)
