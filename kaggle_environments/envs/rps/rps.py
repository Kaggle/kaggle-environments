import json
from os import path
from .agents import agents as all_agents


def interpreter(state, env):
    print(state)
    for agent_state in state:
        observation = agent_state.observation


    player1 = state[0]
    player2 = state[1]

    p1_moves = player1.observation.p1_moves
    player2.observation.p1_moves = p1_moves

    p2_moves = player1.observation.p2_moves
    player2.observation.p2_moves = p2_moves

    results = player1.observation.results
    player2.observation.results = results

    # Specification can fully handle the reset.
    if env.done:
        return state

    # Check for validity of actions

    if player1.action is None:
        player1.status = f"No move returned"
        player2.status = "DONE"
        return state

    if player1.action > env.configuration.weapons:
        player1.status = f"Invalid move: {player1.action}"
        player2.status = "DONE"
        return state

    if player2.action is None:
        player2.status = f"No move returned"
        player1.status = "DONE"
        return state

    if player2.action > env.configuration.weapons:
        player2.status = f"Invalid move: {player2.action}"
        player1.status = "DONE"
        return state

    p1_moves.append(player1.action)
    p2_moves.append(player2.action)
    rounds_played = len(p1_moves)

    if (player1.action + player2.action) % 2 > 0:
        if (player1.action > player2.action):
            results.append(1)
        elif (player1.action < player2.action):
            results.append(2)
        else:
            results.append(0)

    else:
        if (player1.action < player2.action):
            results.append(1)
        elif (player1.action > player2.action):
            results.append(2)
        else:
            results.append(0)

    p1_score = results.count(1)
    p2_score = results.count(2)

    if p1_score > (env.configuration.steps - rounds_played) + p2_score:
        player1.reward = 1
        player1.status = "DONE"
        player2.reward = 0
        player2.status = "DONE"
        return state

    if p2_score > (env.configuration.steps - rounds_played) + p1_score:
        player1.reward = 0
        player1.status = "DONE"
        player2.reward = 1
        player2.status = "DONE"
        return state

    if (env.configuration.steps - rounds_played) == 1:
        player1.reward = 0.5
        player1.status = "DONE"
        player2.reward = 0.5
        player2.status = "DONE"
        return state

    return state


def renderer(state, env):
    weapon_name = ["Rock", "Paper", "Scissors", "Spock", "Lizard", "Airplane", "Sun", "Moon", "Camera", "Grass", "Fire", "Film", "Spanner", "Toilet", "School", "Air", "Death", "Planet", "Curse", "Guitar", "Lock", "Bowl", "Pickaxe", "Cup", "Peace", "Beer", "Computer", "Rain", "Castle", "Water", "Snake", "TV", "Blood", "Rainbow", "Porcupine", "UFO", "Eagle", "Alien", "Monkey", "Prayer", "King", "Mountain", "Queen", "Satan", "Wizard", "Dragon", "Mermaid", "Diamond", "Police", "Trophy", "Woman", "Money", "Baby", "Devil", "Man", "Link", "Home", "Video Game", "Train", "Math", "Car", "Robot", "Noise", "Heart", "Bicycle", "Electricity", "Tree", "Lightning", "Potato", "Ghost", "Duck", "Power", "Wolf", "Microscope", "Cat", "Nuke", "Chicken", "Cloud", "Fish", "Truck", "Spider", "Helicopter", "Bee", "Bomb", "Brain", "Tornado", "Community", "Sand", "Zombie", "Pit", "Bank", "Chain", "Vampire", "Gun", "Bath", "Law", "Monument", "Baloon", "Pancake", "Sword", "Book"]

    rounds_played = len(state[0].observation.results)
    p1_score = state[0].observation.results.count(1)
    p2_score = state[0].observation.results.count(2)
    board = ""

    # This line prints results each round, good for debugging
    for i in range(rounds_played):
        board = board + f"Round {i + 1}: {weapon_name[state[0].observation.p1_moves[i]]} vs {weapon_name[state[0].observation.p2_moves[i]]}, Score: {state[0].observation.results[0:(i + 1)].count(1)} to {state[0].observation.results[0:(i + 1)].count(2)}\n"

    board = board + f"Game ended on round {rounds_played}, final score: {p1_score} to {p2_score}\n"

    return board


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "rps.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    js_path = path.abspath(path.join(dir_path, "rps.js"))
    with open(js_path, encoding="utf8") as js_file:
        return js_file.read()


agents = all_agents


weapons = {
    "Air": "💨",
    "Airplane": "✈️",
    "Alien": "👽",
    "Baby": "👶🏽",
    "Baloon": "🎈",
    "Bank": "🏦",
    "Bath": "🛁",
    "Bee": "🐝",
    "Beer": "🍺",
    "Bicycle": "🚲",
    "Blood": "💉",
    "Bomb": "💣",
    "Book": "📖",
    "Bowl": "🥣",
    "Brain": "🧠",
    "Camera": "📷",
    "Car": "🚗",
    "Castle": "🏰",
    "Cat": "🐈",
    "Chain": "⛓️",
    "Chicken": "🐓",
    "Cloud": "☁️",
    "Community": "👥",
    "Computer": "💻",
    "Cup": "☕",
    "Curse": "🥀",
    "Death": "☠",
    "Devil": "👹",
    "Diamond": "💎",
    "Dragon": "🐉",
    "Duck": "🦆",
    "Eagle": "🦅",
    "Electricity": "💡",
    "Film": "🎥",
    "Fire": "🔥",
    "Fish": "🐟",
    "Ghost": "👻",
    "Grass": "🌱",
    "Guitar": "🎸",
    "Gun": "🔫",
    "Heart": "❤️",
    "Helicopter": "🚁",
    "Home": "🏠",
    "King": "🤴🏼",
    "Law": "⚖️",
    "Lightning": "⚡",
    "Link": "🔗",
    "Lizard": "🦎",
    "Lock": "🔒",
    "Man": "👨🏾",
    "Math": "🔢",
    "Mermaid": "🧜🏽‍♀️",
    "Microscope": "🔬",
    "Money": "💰",
    "Monkey": "🐒",
    "Monument": "🏛️",
    "Moon": "🌙",
    "Mountain": "🏔️",
    "Noise": "🔔",
    "Nuke": "☢️",
    "Pancake": "🥞",
    "Paper": "📄",
    "Peace": "🕊️",
    "Pickaxe": "⛏️",
    "Pit": "🕳️",
    "Planet": "🌎",
    "Police": "👮🏽‍♀️",
    "Porcupine": "🦔",
    "Potato": "🥔",
    "Power": "🔋",
    "Prayer": "🙏🏽",
    "Queen": "👸🏽",
    "Rain": "🌧️",
    "Rainbow": "🌈",
    "Robot": "🤖",
    "Rock": "👊",
    "Sand": "🏖️",
    "Satan": "😈",
    "School": "🏫",
    "Scissors": "✂️",
    "Snake": "🐍",
    "Spanner": "🔧",
    "Spider": "🕷️",
    "Spock": "🖖",
    "Sun": "☀️",
    "Sword": "🗡️",
    "TV": "📺",
    "Toilet": "🚽",
    "Tornado": "🌪️",
    "Train": "🚂",
    "Tree": "🌲",
    "Trophy": "🏆",
    "Truck": "🚚",
    "UFO": "🛸",
    "Vampire": "🧛🏽‍♂️",
    "Video Game": "🎮",
    "Water": "💧",
    "Wizard": "🧙🏼‍♂️",
    "Wolf": "🐺",
    "Woman": "👩🏻",
    "Zombie": "🧟‍♂️"
}