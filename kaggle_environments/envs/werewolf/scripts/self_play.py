import argparse
import json
import logging
import os

from kaggle_environments.envs.werewolf.runner import run_werewolf, setup_logger
from kaggle_environments.envs.werewolf.werewolf import LLM_MODEL_NAMES


logger = logging.getLogger(__name__)

URLS = {
    "gemini": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png"
}


def self_play_config(litellm_model_path, brand):
    logger.info(f"Starting game with model '{litellm_model_path}' and brand '{brand}'")
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager", "Villager"]
    n_player = len(roles)
    model_name = os.path.basename(litellm_model_path)
    names = [f"{model_name}_{i}" for i in range(n_player)]
    thumbnails = [URLS[brand]] * len(names)

    agents = [f'llm/{litellm_model_path}'] * n_player
    agents_config = [{"role": role, "id": name, "agent_id": agent, "thumbnail": url} for role, name, agent, url in
                     zip(roles, names, agents, thumbnails)]
    logger.info(f"Agent configs: {json.dumps(agents_config, indent=4)}")
    config = {
        "actTimeout": 30,
        "agents": agents_config
    }
    return config, agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run self play using the same llm players.")
    parser.add_argument("-r", "--use_random_agent", action="store_true", help='Use random agent for fast testing.')
    parser.add_argument("-o", "--output_dir", type=str, help="output directory", default="werewolf_replay")
    parser.add_argument("-n", "--base_name", type=str, help="the base file name of .html, .json and .log",
                        default="out")
    parser.add_argument("-m", "--litellm_model_path", type=str, help="path to litellm model, check litellm documentation for available models.",
                        default="gemini/gemini-2.5-flash", choices=LLM_MODEL_NAMES)
    parser.add_argument("-b", "--brand", type=str, help="brand of the model", choices=list(URLS.keys()),
                        default="gemini")
    parser.add_argument("-d", "--disable_debug_mode", action="store_true", help='Disable debug mode. By default we enable debug.')
    args = parser.parse_args()

    setup_logger(output_dir=args.output_dir, base_name=args.base_name)
    logger.info("Starting script execution.")

    config, agents = self_play_config(args.litellm_model_path, args.brand)
    if args.use_random_agent:
        agents = ['random'] * len(agents)
    run_werewolf(args.output_dir, args.base_name, config, agents, debug=not args.disable_debug_mode)
