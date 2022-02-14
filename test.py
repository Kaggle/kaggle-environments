
from kaggle_environments import make
env = make("kore_fleets", debug=True)
game = env.run(["simp", "simp"])

"""
for i in range(len(game)):
    print("turn",i)
    print("actions")
    print(game[i][0]['action'])
    print(game[i][1]['action'])
    print("players")
    for player in game[i][0]['observation']['players']:
        print(player)
    print("")
"""

#env.render(mode="ansi")
