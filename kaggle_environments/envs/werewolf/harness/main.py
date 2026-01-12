"""Main file for the Game Arena submission."""

import os
import sys

_AGENT_OBJECT = None
_SETUP_COMPLETE = False


def agent(observation, configuration):
    """Kaggle agent for Game Arena."""
    global _AGENT_OBJECT, _SETUP_COMPLETE

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

        from kaggle_environments.envs.werewolf.harness.base import LLMWerewolfAgent
        from kaggle_environments.envs.werewolf.werewolf import LLM_SYSTEM_PROMPT, AgentFactoryWrapper

        if "MODEL_NAME" not in os.environ:
            raise ValueError("MODEL_NAME was not specified as an environment variable. Agent cannot be configured.")

        if "MODEL_PROXY_KEY" not in os.environ:
            raise ValueError(
                "MODEL_PROXY_KEY was not specified as an environment variable. Model proxy cannot function correctly."
            )

        if "MODEL_PROXY_URL" not in os.environ:
            raise ValueError("MODEL_PROXY_URL was not injected. Agent cannot run.")

        _AGENT_OBJECT = AgentFactoryWrapper(
            agent_class=LLMWerewolfAgent,
            model_name=f"openai/{os.environ['MODEL_NAME']}",
            system_prompt=LLM_SYSTEM_PROMPT,
            litellm_model_proxy_kwargs={
                "api_base": f"{os.environ['MODEL_PROXY_URL']}/openapi",
                "api_key": os.environ["MODEL_PROXY_KEY"],
            },
        )

        _SETUP_COMPLETE = True
        print("--- Agent setup complete. ---")

    return _AGENT_OBJECT(observation, configuration)
