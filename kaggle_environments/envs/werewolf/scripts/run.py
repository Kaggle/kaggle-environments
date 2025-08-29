import argparse
import os
import sys
import yaml
import logging
import random

# Add the project root to the Python path to allow importing from kaggle_environments
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kaggle_environments.envs.werewolf.runner import run_werewolf, setup_logger, append_timestamp_to_dir, LogExecutionTime
from kaggle_environments.envs.werewolf.werewolf import AgentFactoryWrapper, agents, LLM_SYSTEM_PROMPT, register_agents
from kaggle_environments.envs.werewolf.harness.base import LLMWerewolfAgent

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a single Werewolf game.")
    parser.add_argument(
        "-c", "--config_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "configs/run/run_config.yaml"),
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str,
        default="werewolf_run",
        help="Output directory for the log and replay file."
    )
    parser.add_argument("-d", "--debug", action="store_true", help='Enable debug mode.')
    parser.add_argument("-r", "--random_agents", action="store_true",
                        help='Use random agents for all players for fast testing.')
    parser.add_argument("-a", "--append_timestamp_to_dir", action="store_true",
                        help="Append a timestamp to the output directory.")
    parser.add_argument("-s", "--shuffle_roles", action="store_true",
                        help="If provided, shuffle the roles provided in the config.")

    args = parser.parse_args()

    # Create a unique subdirectory for this run
    run_output_dir = append_timestamp_to_dir(args.output_dir, append=args.append_timestamp_to_dir)

    os.makedirs(run_output_dir, exist_ok=True)

    base_name = "werewolf_game"
    setup_logger(output_dir=run_output_dir, base_name=base_name)

    # Load game configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
        game_config = config.get('game_config', {})

    # shuffle roles
    if args.shuffle_roles:
        roles = [agent['role'] for agent in game_config['agents']]
        random.shuffle(roles)
        for agent, new_role in zip(game_config['agents'], roles):
            agent['role'] = new_role

    # Extract agent harnesses from the config and register the agents
    agents_ = [agent.get('agent_id', 'random') for agent in game_config.get('agents', [])]
    agent_dict = {}
    for agent_name in agents_:
        if agent_name.startswith('llm/'):
            model_name = agent_name.lstrip('llm/')
            agent_dict[agent_name] = AgentFactoryWrapper(
                LLMWerewolfAgent,
                model_name=model_name,
                system_prompt=LLM_SYSTEM_PROMPT
            )
    register_agents(agent_dict)

    if args.random_agents:
        logger.info("Using random agents for all players.")
        agents_ = ['random'] * len(agents_)

    logger.info(f"Starting Werewolf game run. Output will be saved to: {run_output_dir}")
    with LogExecutionTime(logger_obj=logger, task_str="single game"):
        run_werewolf(
            output_dir=run_output_dir,
            base_name=base_name,
            config=game_config,
            agents=agents_,
            debug=args.debug
        )
    logger.info(f"Game finished. Replay and log saved in: {run_output_dir}")


if __name__ == "__main__":
    main()
