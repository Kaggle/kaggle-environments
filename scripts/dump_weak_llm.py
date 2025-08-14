from kaggle_environments import make

# 1. Create the Werewolf environment

URLS = {
    "gemini": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png",
    "deepseek": "https://images.seeklogo.com/logo-png/61/1/deepseek-ai-icon-logo-png_seeklogo-611473.png",
    "kimi": "https://images.seeklogo.com/logo-png/61/1/kimi-logo-png_seeklogo-611650.png",
    "qwen": "https://images.seeklogo.com/logo-png/61/1/qwen-icon-logo-png_seeklogo-611724.png"
}

# TODO: vertex AI model still has issues

roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager", "Villager"]
names = ["gemini-2.5-flash", "deepseek-r1", "kimi-k2", "qwen3", "gpt-4.1", "o4-mini", "claude-4-sonnet", "grok-4"]

parameter_dict = {
    "together_ai/deepseek-ai/DeepSeek-R1": {"max_tokens": 163839},
    "claude-4-sonnet-20250514": {"max_tokens": 64000}
}

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

agents_config = [
    {"role": role, "id": name, "agent_id": f"llm_harness/{model}",
     "thumbnail": URLS[brand], "display_name": model,
     "agent_harness_name": "llm_harness", "llms": [{"model_name": model, "parameters": parameter_dict.get(model, {})}]}
    for role, name, brand, model in zip(roles, names, brands, models)
]


env = make(
    'werewolf',
    debug=False,
    configuration={
        "actTimeout": 60,
        "agents": agents_config,
        "discussion_protocol": {
            "name": "ParallelDiscussion",
            "params": {
                "ticks": 2
            }
        }
    }
)

# 2. Define the players for the game.
num_players = len(roles)
agents = [f"llm/{model}" for model in models]

# 3. Run a full game episode.
# The 'run' method takes a list of agents.
env.run(agents)
for i, state in enumerate(env.steps):
    env.render_step_ind = i
    out = env.renderer(state, env)
    print(out)

# 4. Render the game to an HTML string.
# This is where the magic happens! The environment takes the werewolf.html template
# and fills it with the game data and rendering logic.
html_content = env.render(mode="html")

# 5. Save the generated HTML content to a new file.
# This is the file you will open in your browser.
output_filename = "game_replay.html"
with open(output_filename, "w") as f:
    f.write(html_content)

print(f"Werewolf game replay saved to: {output_filename}")
print("Open this file in your web browser to view the game.")

