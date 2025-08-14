from kaggle_environments import make

# 1. Create the Werewolf environment

URLS = {
    "gemini": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png"
}

roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager", "Villager"]
names = ["Peter", "Tom", "Harry", "Kevin", "Marry", "Sophia", "Alex", "Amy"]
model_names = ["gemini-2.5-pro", "gemini-2.5-flash", "gpt-4.1", "o3", "o4-mini", "claude-4-sonnet", "claude-4-opus", "grok-4"]
thumbnails = [URLS['gemini'], URLS['gemini'], URLS['openai'], URLS['openai'], URLS['openai'], URLS['claude'], URLS['claude'], URLS['grok']]
agent_harness_names = ['basic'] * len(roles)
agents_config = [
    {"role": role, "id": name, "agent_id": f"{harness_name}-{model_name}", "display_name": name,
     "agent_harness_name": harness_name, "thumbnail": url, "llms": [{"model_name": model_name}]}
    for role, name, model_name, harness_name, url in zip(roles, names, model_names, agent_harness_names, thumbnails)
]

env = make(
    'werewolf',
    debug=True,
    configuration={
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
agent_paths = ['random'] * len(roles)

# 3. Run a full game episode.
# The 'run' method takes a list of agents.
env.run(agent_paths)
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
output_filename = "experiments/repeated_8player/game_replay.html"
with open(output_filename, "w") as f:
    f.write(html_content)

print(f"Werewolf game replay saved to: {output_filename}")
print("Open this file in your web browser to view the game.")

