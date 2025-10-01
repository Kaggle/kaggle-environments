"""Run pairwise zero-sum setting where one player play the entire team of Werewolf and another player play
the team of Villager. Given a config, we play all possible pairwise combinations N times.
"""

import argparse
import logging
import math
import multiprocessing
import os
import random
from copy import deepcopy
from typing import List

import tenacity
import yaml
from tqdm import tqdm

from kaggle_environments.envs.werewolf.game.consts import RoleConst
from kaggle_environments.envs.werewolf.runner import LogExecutionTime, append_timestamp_to_dir, setup_logger
from kaggle_environments.envs.werewolf.scripts.utils import run_single_game_cli

# Initialize a placeholder logger
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Loads the configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_team_roles(base_roles: List[str]) -> (List[str], List[str]):
    """Partitions roles into villager and werewolf teams."""
    villager_roles = []
    werewolf_roles = []
    for role_name in base_roles:
        role = RoleConst(role_name)
        if role == RoleConst.WEREWOLF:
            werewolf_roles.append(role_name)
        else:
            villager_roles.append(role_name)
    return villager_roles, werewolf_roles


run_single_game_with_retry = tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
)(run_single_game_cli)


def game_runner_wrapper(args):
    """Wrapper to unpack arguments for the multiprocessing pool."""
    game_dir, game_config, use_random_agents, debug, _, _ = args
    run_single_game_with_retry(game_dir, game_config, use_random_agents, debug)


def assign_roles_dup_agents(roles, agent_config, player_ids):
    agents = [deepcopy(agent_config) for _ in range(len(roles))]
    for role, agent, player_id in zip(roles, agents, player_ids):
        agent["role"] = role
        agent["id"] = player_id
    return agents


def prepare_pairwise_agents(villager_roles, werewolf_roles, player_a_config, player_b_config, player_ids):
    pid_v, pid_w = player_ids[: len(villager_roles)], player_ids[len(villager_roles) :]
    agents_v = assign_roles_dup_agents(villager_roles, player_a_config, pid_v)
    agents_w = assign_roles_dup_agents(werewolf_roles, player_b_config, pid_w)
    agents = agents_v + agents_w
    return agents


def generate_game_tasks(output_dir, num_tournaments, config, use_random_agents, debug):
    """
    Generates game configurations for a pairwise matrix tournament.
    """
    base_game_config = config["game_config"]
    all_players = base_game_config["agents"]
    num_players = len(all_players)
    base_roles = [agent["role"] for agent in all_players]
    player_ids = [agent["id"] for agent in all_players]

    villager_roles, werewolf_roles = get_team_roles(base_roles)

    if not werewolf_roles:
        raise ValueError("Configuration must include at least one werewolf role.")
    if not villager_roles:
        raise ValueError("Configuration must include at least one villager role.")

    for tourney_idx in range(num_tournaments):
        for i in range(num_players):
            for j in range(num_players):
                game_dir = os.path.join(output_dir, f"tourney_{tourney_idx}", f"game_{i}_vs_{j}")
                os.makedirs(game_dir, exist_ok=True)

                player_a_config = all_players[i]
                player_b_config = all_players[j]

                game_agents_config = prepare_pairwise_agents(
                    villager_roles, werewolf_roles, player_a_config, player_b_config, player_ids
                )

                # since name has to be unique and all names come from config, we by default shuffle all names
                # since name might change
                random.shuffle(player_ids)
                for agent_ind, agent in enumerate(game_agents_config):
                    agent["id"] = player_ids[agent_ind]

                random.shuffle(game_agents_config)

                game_config = {**base_game_config, "agents": game_agents_config}
                yield game_dir, game_config, use_random_agents, debug, tourney_idx, f"{i}_vs_{j}"


def run_tournament(output_dir, num_tournaments, config, use_random_agents, debug, parallel, num_processes):
    """
    Runs a tournament by generating all game tasks and processing them,
    potentially in parallel.
    """
    total_games = num_tournaments * len(config["game_config"]["agents"]) ** 2

    if parallel:
        logger.info(f"Running games in parallel with up to {num_processes} processes.")

    game_tasks = generate_game_tasks(output_dir, num_tournaments, config, use_random_agents, debug)

    # the following shuffle is to reduce the load of a particular LLM api
    game_tasks = [*game_tasks]
    random.shuffle(game_tasks)

    with tqdm(total=total_games, desc="Processing Games") as pbar:
        if parallel:
            with multiprocessing.Pool(processes=num_processes) as pool:
                for _ in pool.imap_unordered(game_runner_wrapper, game_tasks):
                    pbar.update(1)
        else:
            for task_args in game_tasks:
                game_runner_wrapper(task_args)
                pbar.update(1)

    logger.info("All game tasks have been processed.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "configs", "run", "run_config.yaml")

    parser = argparse.ArgumentParser(description="Run a pairwise matrix tournament for the Werewolf game.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for game replays and logs.",
        default="werewolf_pairwise_matrix",
    )
    parser.add_argument(
        "-c", "--config", type=str, default=default_config_path, help="Path to the base configuration YAML file."
    )
    parser.add_argument(
        "-t",
        "--num_tournaments",
        type=int,
        default=1,
        help="Number of tournaments to run. Each tournament is a full N*N matrix of games.",
    )
    parser.add_argument(
        "-r", "--use_random_agents", action="store_true", help="Use random agents for all players for fast testing."
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode for the game environment. Forces sequential execution.",
    )
    parser.add_argument("-p", "--parallel", action="store_true", help="Run games in parallel using multiple processes.")
    parser.add_argument(
        "-n", "--num_processes", type=int, default=None, help="Number of processes for parallel execution."
    )
    parser.add_argument(
        "-a", "--append_timestamp_to_dir", action="store_true", help="Append a timestamp to the output directory."
    )

    args = parser.parse_args()

    output_dir = append_timestamp_to_dir(args.output_dir, append=args.append_timestamp_to_dir)

    os.makedirs(output_dir, exist_ok=True)

    setup_logger(output_dir, "run_pairwise_matrix")

    config = load_config(args.config)

    if args.num_processes is None:
        num_processes = max(1, math.floor(multiprocessing.cpu_count() * 0.8))
    else:
        num_processes = args.num_processes

    logger.info("Starting tournament with the following settings:")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Number of Tournaments: {args.num_tournaments}")
    logger.info(f"Parallel Execution: {args.parallel}")
    if args.parallel:
        logger.info(f"Number of Processes: {num_processes}")
    logger.info(f"Debug Mode: {args.debug}")
    logger.info(f"Use Random Agents: {args.use_random_agents}")

    with LogExecutionTime(logger_obj=logger, task_str="pairwise matrix tournament"):
        run_tournament(
            output_dir=output_dir,
            num_tournaments=args.num_tournaments,
            config=config,
            use_random_agents=args.use_random_agents,
            debug=args.debug,
            parallel=args.parallel,
            num_processes=num_processes,
        )
    logger.info("Tournament finished successfully.")


if __name__ == "__main__":
    main()
