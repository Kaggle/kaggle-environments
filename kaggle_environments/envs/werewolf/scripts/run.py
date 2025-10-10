import argparse
import logging
import os
import random

import yaml

from kaggle_environments.envs.werewolf.harness.base import LLMWerewolfAgent
from kaggle_environments.envs.werewolf.runner import (
    LogExecutionTime,
    append_timestamp_to_dir,
    log_git_hash,
    run_werewolf,
    setup_logger,
)
from kaggle_environments.envs.werewolf.werewolf import LLM_SYSTEM_PROMPT, AgentFactoryWrapper, register_agents

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a single Werewolf game.")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs/run/run_config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="werewolf_run", help="Output directory for the log and replay file."
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "-r", "--random_agents", action="store_true", help="Use random agents for all players for fast testing."
    )
    parser.add_argument(
        "-a", "--append_timestamp_to_dir", action="store_true", help="Append a timestamp to the output directory."
    )
    parser.add_argument(
        "-s", "--shuffle_roles", action="store_true", help="If provided, shuffle the roles provided in the config."
    )

    args = parser.parse_args()

    # Create a unique subdirectory for this run
    run_output_dir = append_timestamp_to_dir(args.output_dir, append=args.append_timestamp_to_dir)

    os.makedirs(run_output_dir, exist_ok=True)

    base_name = "werewolf_game"
    setup_logger(output_dir=run_output_dir, base_name=base_name)

    log_git_hash()

    # Load game configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        game_config = config.get("game_config", {})

    # shuffle roles
    if args.shuffle_roles:
        role_and_params = [(agent["role"], agent.get("role_params", {})) for agent in game_config["agents"]]
        random.shuffle(role_and_params)
        for agent, (new_role, new_role_params) in zip(game_config["agents"], role_and_params):
            agent["role"] = new_role
            agent["role_params"] = new_role_params

    # Extract agent harnesses from the config and register the agents
    agents_ = [agent.get("agent_id", "random") for agent in game_config.get("agents", [])]
    agent_dict = {}
    for agent_name in agents_:
        if agent_name.startswith("llm/"):
            model_name = agent_name.lstrip("llm/")
            agent_dict[agent_name] = AgentFactoryWrapper(
                LLMWerewolfAgent, model_name=model_name, system_prompt=LLM_SYSTEM_PROMPT
            )
    register_agents(agent_dict)

    if args.random_agents:
        logger.info("Using random agents for all players.")
        agents_ = ["random"] * len(agents_)

    logger.info(f"Starting Werewolf game run. Output will be saved to: {run_output_dir}")
    with LogExecutionTime(logger_obj=logger, task_str="single game"):
        run_werewolf(
            output_dir=run_output_dir, base_name=base_name, config=game_config, agents=agents_, debug=args.debug
        )
    logger.info(f"Game finished. Replay and log saved in: {run_output_dir}")


if __name__ == "__main__":
    main()
