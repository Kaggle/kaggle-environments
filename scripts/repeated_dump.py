
import random
import json
import os
from typing import List

from tqdm import tqdm

from kaggle_environments import make


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


def shuffled(items):
    shuffled_items = list(items).copy()
    random.shuffle(shuffled_items)
    return shuffled_items


def save_html(env, output_path):
    html_content = env.render(mode="html")
    with open(output_path, "w") as f:
        f.write(html_content)


def save_json(env, output_path):
    json_content = env.render(mode="json")
    with open(output_path, "w") as f:
        f.write(json_content)


def run(config, agents: List[str], html_path, json_path):
    # 1. Create the Werewolf environment
    env = make(
        'werewolf',
        debug=True,
        configuration=config
    )

    # 2. Define the players for the game.
    num_players = len(agents)

    # 3. Run a full game episode.
    # The 'run' method takes a list of agents.
    env.run(agents)
    # for i, state in enumerate(env.steps):
    #     env.render_step_ind = i
    #     out = env.renderer(state, env)
    #     print(out)

    save_html(env, html_path)
    save_json(env, json_path)

    print(f"HTML saved to: {html_path}")
    print(f"JSON saved to: {json_path}")
    print("Open this file in your web browser to view the game.")


def experiment(num_reps):
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager", "Villager"]
    names = ["gemini-2.5-flash", "deepseek-r1", "kimi-k2", "qwen3", "gpt-4.1", "o4-mini", "claude-4-sonnet", "grok-4"]
    models = [
        "gemini/gemini-2.5-flash",
        "together_ai/deepseek-ai/DeepSeek-R1",
        "together_ai/moonshotai/Kimi-K2-Instruct",
        "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "gpt-4.1",
        "o4-mini",
        "claude-4-sonnet-20250514",
        "xai/grok-4-latest",
    ]
    brands = ['gemini', 'deepseek', 'kimi', 'qwen', 'openai', 'openai', 'claude', 'grok']


    parameter_dict = {
        "together_ai/deepseek-ai/DeepSeek-R1": {"max_tokens": 163839},
        "together_ai/moonshotai/Kimi-K2-Instruct": {"max_tokens": 128000},
        "claude-4-sonnet-20250514": {"max_tokens": 64000},
        "gpt-4.1": {"max_tokens": 30000},
    }

    # shuffle the roles and the models
    roles = shuffled(roles)
    names, models, brands = zip(*shuffled(zip(names, models, brands)))

    # import pdb; pdb.set_trace()

    agents_config = [
        {"role": role, "id": name, "agent_id": f"llm_harness/{model}",
         "thumbnail": URLS[brand], "display_name": model,
         "agent_harness_name": "llm_harness",
         "llms": [{"model_name": model, "parameters": parameter_dict.get(model, {})}]}
        for role, name, brand, model in zip(roles, names, brands, models)
    ]

    config = {
        "actTimeout": 120,
        "runTimeout": 1800,
        "agents": agents_config,
        "discussion_protocol": {
            # "name": "ParallelDiscussion",
            # "params": {
            #     "ticks": 1
            # }
            "name": "RoundRobinDiscussion",
            "params": {
                "max_rounds": 2
            }
        }
    }

    agents = [f"llm/{model}" for model in models]


    output_dir = "/home/hannw/repos/kaggle-environments/experiments/repeated_8player"
    progress_path = os.path.join(output_dir, 'progress.jsonl')
    write_json(config, f"{output_dir}/config.json")

    with tqdm(total=num_reps, desc="Processing reps") as pbar:
        for i in tqdm(range(num_reps)):
            html_path = os.path.join(output_dir, f"game_replay_{i}.html")
            json_path = os.path.join(output_dir, f"game_replay_{i}.json")
            run(config, agents, html_path, json_path)
            pbar.update(1)
            progress_data = {'current_item': pbar.n, 'total_items': pbar.total}
            with open(progress_path, 'a') as handle:
                handle.write(json.dumps(progress_data) + '\n')


if __name__ == "__main__":
    experiment(8)
