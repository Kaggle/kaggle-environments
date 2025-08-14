from kaggle_environments import make

# 1. Create the Werewolf environment

URLS = {
    "gemini": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png",
    "deepseek": "https://images.seeklogo.com/logo-png/61/1/deepseek-ai-icon-logo-png_seeklogo-611473.png",
    "kimi": "https://images.seeklogo.com/logo-png/61/1/kimi-logo-png_seeklogo-611650.png"
}

# TODO: vertex AI model still has issues

roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager", "Villager"]
# names = ["Robert", "John", "Mary", "Alex", "Elizabeth", "Patrick", "Michael", "Jennifer"]
names = ["gemini-2.5-pro", "gemini-2.5-flash", "gpt-4.1", "o4-mini", "claude-4-sonnet", "claude-4-opus", "grok-4", "deepseek-r1"]
# names = ["gemini-2.5-pro", "gemini-2.5-flash", "gpt-4.1", "o4-mini", "claude-4-sonnet", "claude-4-opus", "grok-4", "kimi-k2"]


parameter_dict = {"deepseek-r1": {"max_tokens": 163839}}

models = [
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
    "gpt-4.1",
    "o4-mini",
    "claude-4-sonnet-20250514",
    "claude-4-opus-20250514",
    "xai/grok-4-latest",
    "together_ai/deepseek-ai/DeepSeek-R1",
    # "together_ai/moonshotai/Kimi-K2-Instruct"
]

brands = ['gemini', 'gemini', 'openai', 'openai', 'claude', 'claude', 'grok', 'deepseek']
# brands = ['gemini', 'gemini', 'openai', 'openai', 'claude', 'claude', 'grok', 'kimi']

agents_config = [
    {"role": role, "id": name, "agent_id": f"llm_harness/{model}",
     "thumbnail": URLS[brand], "display_name": model,
     "agent_harness_name": "llm_harness", "llms": [{"model_name": model}]}
    for role, name, brand, model in zip(roles, names, brands, models)
]


env = make(
    'werewolf',
    debug=True,
    configuration={
        "actTimeout": 300,
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

