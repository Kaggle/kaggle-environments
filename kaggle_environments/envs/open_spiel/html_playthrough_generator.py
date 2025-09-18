import os

import pyspiel

from kaggle_environments import make

open_spiel_game_name = "connect_four"
game = pyspiel.load_game(open_spiel_game_name)
game_type = game.get_type()
environment_name = f"open_spiel_{game_type.short_name}"
agents_to_run = ["random"] * game.num_players()
replay_width = 500
replay_height = 450
debug_mode = True
env = make(environment_name, debug=debug_mode)

print(f"Running game with agents: {agents_to_run}...")
env.run(agents_to_run)
print("Game finished.")

print("Generating HTML replay...")
html_replay = env.render(mode="html", width=replay_width, height=replay_height)

output_html_file = f"kaggle_environments/envs/open_spiel/{environment_name}_game_replay.html"
print(f"Saving replay to: '{output_html_file}'")
with open(output_html_file, "w", encoding="utf-8") as f:
    f.write(html_replay)

print("-" * 20)
print(f"Successfully generated replay: {os.path.abspath(output_html_file)}")
print("-" * 20)
