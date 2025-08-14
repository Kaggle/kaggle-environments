import random
import json
import os
from typing import List
import collections
from itertools import permutations

from tqdm import tqdm

from kaggle_environments import make

# official list of 8 models

"""
grok-4, o3, gemini-2.5 pro, o4-mini, qwen3, deepseek, claude opus, kimi-k2
"""



# --- Constants and Helper Functions ---
URLS = {
    "gemini": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png",
    "deepseek": "https://images.seeklogo.com/logo-png/61/1/deepseek-ai-icon-logo-png_seeklogo-611473.png",
    "kimi": "https://images.seeklogo.com/logo-png/61/1/kimi-logo-png_seeklogo-611650.png",
    "qwen": "https://images.seeklogo.com/logo-png/61/1/qwen-icon-logo-png_seeklogo-611724.png"
}


def write_json(content, output_path):
    with open(output_path, "w") as f:
        f.write(json.dumps(content, indent=4))


def save_html(env, output_path):
    html_content = env.render(mode="html")
    with open(output_path, "w") as f:
        f.write(html_content)


def save_json(env, output_path):
    json_content = env.render(mode="json")
    with open(output_path, "w") as f:
        f.write(json_content)


def run(config, agents: List[str], html_path, json_path):
    env = make('werewolf', debug=True, configuration=config)
    env.run(agents)
    save_html(env, html_path)
    save_json(env, json_path)
    print(f"HTML saved to: {html_path}")


# --- Core Logic for Generating Unique Configurations ---

def get_rotationally_unique_configs(roles: List[str]) -> List[List[str]]:
    """
    Generates all unique permutations of roles, filtering out any that are
    rotational duplicates. This creates a set of 'necklaces'.
    """
    # 1. Get all 840 unique linear permutations first.
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


# --- Main Experiment Function ---

def experiment(num_blocks: int):
    """
    Runs a tournament using a stratified block design, sampling from
    rotationally unique role configurations.
    """
    base_roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager", "Villager"]
    models = ["grok-4", "gpt-4.1", "gemini-2.5-pro", "o4-mini", "qwen3", "deepseek-r1", "claude-4-opus", "kimi-k2"]
    models_links = [
        "xai/grok-4-0709", "gpt-4.1", "gemini/gemini-2.5-pro", "o4-mini", 
        "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "together_ai/deepseek-ai/DeepSeek-R1", 
        "claude-4-opus-20250514", "together_ai/moonshotai/Kimi-K2-Instruct"
    ]
    brands = ["grok", "openai", "gemini", "openai", "qwen", "deepseek", "claude", "kimi"]


    players_data = list(zip(models, models_links, brands))

    parameter_dict = {
        "together_ai/deepseek-ai/DeepSeek-R1": {"max_tokens": 163839},
        "together_ai/moonshotai/Kimi-K2-Instruct": {"max_tokens": 128000},
        "claude-4-sonnet-20250514": {"max_tokens": 64000},
        "claude-4-opus-20250514": {"max_tokens": 32000},
        "gpt-4.1": {"max_tokens": 30000}
    }

    # 1. Generate the rotationally unique role configurations.
    print("Generating unique role configurations (filtering rotations)...")
    all_role_configs = get_rotationally_unique_configs(base_roles)
    print(f"Found {len(all_role_configs)} rotationally unique arrangements (down from 840).")

    # 2. Sample from these unique configurations for each block.
    num_configs_to_sample = min(num_blocks, len(all_role_configs))
    if num_blocks > len(all_role_configs):
        print(
            f"Warning: Requested {num_blocks} blocks, but only {len(all_role_configs)} unique arrangements exist. Sampling with replacement.")
        sampled_role_configs = random.choices(all_role_configs, k=num_blocks)
    else:
        sampled_role_configs = random.sample(all_role_configs, num_configs_to_sample)

    output_dir = "/home/hannw/repos/kaggle-environments/experiments/repeated_8player_blocked_unique_20250801"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_games = len(sampled_role_configs) * len(players_data)
    game_counter = 0

    with tqdm(total=total_games, desc="Processing Games") as pbar:
        for block_index, block_role_config in enumerate(sampled_role_configs):

            # --- Stratification: Shuffle players once per block ---
            random.shuffle(players_data)

            # Use a deque for efficient rotation. Players rotate through the fixed roles.
            current_players_deque = collections.deque(players_data)

            for game_in_block in range(len(players_data)):
                # The role configuration is fixed for this entire block.
                roles = block_role_config

                # Players are rotated for each game.
                current_players = list(current_players_deque)
                names, models, brands = zip(*current_players)

                agents_config = [
                    {"role": role, "id": name, "agent_id": f"llm_harness/{model}",
                     "thumbnail": URLS[brand], "display_name": model,
                     "agent_harness_name": "llm_harness",
                     "llms": [{"model_name": model, "parameters": parameter_dict.get(model, {})}]}
                    for role, name, brand, model in zip(roles, names, brands, models)
                ]

                config = {
                    "actTimeout": 300, "runTimeout": 3600, "agents": agents_config,
                    "discussion_protocol": {"name": "RoundRobinDiscussion", "params": {"max_rounds": 2}}
                }
                # agents = [f"llm/{model}" for model in models]
                agents = ["random"] * len(models)

                # Save config and run the game
                config_path = os.path.join(output_dir, f"block{block_index}/config{game_in_block}.json")
                html_path = os.path.join(output_dir, f"block{block_index}/game{game_in_block}.html")
                json_path = os.path.join(output_dir, f"block{block_index}/game{game_in_block}.json")
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
                write_json(config, config_path)

                run(config, agents, html_path, json_path)

                pbar.update(1)
                game_counter += 1

                # Rotate players for the next game in this block
                current_players_deque.rotate(1)


if __name__ == "__main__":
    # Run a tournament of 10 blocks.
    # This will sample 10 of the 105 unique role arrangements.
    # It will run 10 * 8 = 80 games in total.
    experiment(num_blocks=10)