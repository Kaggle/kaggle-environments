import argparse
import collections
import copy
import os
import random
from itertools import permutations
from typing import List

import yaml
from tqdm import tqdm

from kaggle_environments.envs.werewolf.runner import run_werewolf, setup_logger


def load_config(config_path):
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_rotationally_unique_configs(roles: List[str]) -> List[List[str]]:
    """
    Generates all unique permutations of roles, filtering out any that are
    rotational duplicates. This creates a set of 'necklaces'.
    """
    # Get all unique linear permutations first.
    all_perms = list(set(permutations(roles)))

    seen_configs = set()
    unique_necklaces = []

    for perm in all_perms:
        if perm not in seen_configs:
            # This is a new, unique arrangement we haven't seen.
            unique_necklaces.append(list(perm))

            # Now, add all of its rotations to the 'seen' set so we can
            # ignore them if we encounter them later.
            temp_deque = collections.deque(perm)
            for _ in range(len(perm)):
                temp_deque.rotate(1)
                seen_configs.add(tuple(temp_deque))

    return unique_necklaces


def run_single_game(game_dir, base_name, game_config, agents_for_run, debug):
    """Sets up and runs a single game instance."""
    setup_logger(output_dir=game_dir, base_name=base_name)

    out_config = {"game_config": game_config}
    config_path = os.path.join(game_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(out_config, f, default_flow_style=False)

    run_werewolf(
        output_dir=game_dir,
        base_name=base_name,
        config=game_config,
        agents=agents_for_run,
        debug=debug
    )


def run_experiment(output_dir, num_blocks, config, use_random_agents, debug):
    """
    Runs a tournament using a stratified block design, sampling from
    rotationally unique role configurations. Retries individual games on failure.
    """
    base_game_config = config['game_config']
    players_data = base_game_config['agents']
    base_roles = [agent['role'] for agent in players_data]

    print("Generating unique role configurations (filtering rotations)...")
    all_role_configs = get_rotationally_unique_configs(base_roles)
    print(f"Found {len(all_role_configs)} rotationally unique arrangements.")

    available_role_configs = []

    total_games = num_blocks * len(players_data)

    with tqdm(total=total_games, desc="Processing Games") as pbar:
        for block_index in range(num_blocks):
            block_dir = os.path.join(output_dir, f"block_{block_index}")
            os.makedirs(block_dir, exist_ok=True)
            print(f"\n--- Starting block {block_index + 1}/{num_blocks} (Output: {block_dir}) ---")

            if not available_role_configs:
                if num_blocks > len(all_role_configs):
                    print("Warning: Sampling with replacement as num_blocks > unique configurations.")
                available_role_configs = list(all_role_configs)
                random.shuffle(available_role_configs)

            block_role_config = available_role_configs.pop()

            random.shuffle(players_data)
            current_players_deque = collections.deque(players_data)

            for game_in_block in range(len(players_data)):
                game_dir = os.path.join(block_dir, f"game_{game_in_block}")
                os.makedirs(game_dir, exist_ok=True)
                base_name = "replay"

                # Prepare game-specific configurations
                current_players = list(current_players_deque)
                game_agents_config = []
                for i, player_config in enumerate(current_players):
                    new_config = copy.deepcopy(player_config)
                    new_config['role'] = block_role_config[i]
                    game_agents_config.append(new_config)

                game_config = copy.deepcopy(base_game_config)
                game_config['agents'] = game_agents_config

                agents_for_run = [agent['agent_id'] for agent in game_agents_config]
                if use_random_agents:
                    agents_for_run = ['random'] * len(agents_for_run)

                if debug:
                    # In debug mode, run without try/except to see the full traceback
                    run_single_game(game_dir, base_name, game_config, agents_for_run, debug)
                else:
                    game_successful = False
                    while not game_successful:
                        try:
                            run_single_game(game_dir, base_name, game_config, agents_for_run, debug)
                            game_successful = True
                        except Exception as e:
                            print(f"\n--- ERROR in block {block_index + 1}, game {game_in_block + 1} ---")
                            print(f"Error: {e}")
                            print(f"Logs and config for the failed game are in: {game_dir}")
                            print("Retrying game...")

                pbar.update(1)
                current_players_deque.rotate(1)

            print(f"--- Block {block_index + 1} completed successfully. ---")


def main():
    parser = argparse.ArgumentParser(description="Run a block-design experiment for the Werewolf game.")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory for game replays and logs.",
                        default="werewolf_block_experiment")
    parser.add_argument("-c", '--config', type=str,
                        default='kaggle_environments/envs/werewolf/scripts/configs/block_basic.yaml',
                        help="Path to the base configuration YAML file.")
    parser.add_argument("-b", "--num_blocks", type=int, default=10,
                        help="Number of blocks to run in the experiment.")
    parser.add_argument("-r", "--use_random_agents", action="store_true",
                        help='Use random agents for all players for fast testing.')
    parser.add_argument("-d", "--debug", action="store_true",
                        help='Enable debug mode for the game environment.')

    args = parser.parse_args()

    # General setup, logger will be configured per-game
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    run_experiment(
        output_dir=args.output_dir,
        num_blocks=args.num_blocks,
        config=config,
        use_random_agents=args.use_random_agents,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
