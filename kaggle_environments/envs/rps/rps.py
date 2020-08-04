import json
from os import path
import random


def interpreter(state, env):
    player1 = state[0]
    player2 = state[1]

    # Keep the game in sync between both agents.

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

    # Update the game history

    p1_moves.append(player1.action)
    p2_moves.append(player2.action)
    rounds_played = len(p1_moves)

    # Calculate result of the round

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

    # Update the scores for the round
    p1_score = results.count(1)
    p2_score = results.count(2)

    # Check for a player 1 win

    if p1_score > (env.configuration.steps - rounds_played) + p2_score:
        player1.reward = 1
        player1.status = "DONE"
        player2.reward = 0
        player2.status = "DONE"
        return state

    # Check for a player 2 win

    if p2_score > (env.configuration.steps - rounds_played) + p1_score:
        player1.reward = 0
        player1.status = "DONE"
        player2.reward = 1
        player2.status = "DONE"
        return state

    # Check for ending on rounds

    if (env.configuration.steps - rounds_played) == 1:
        player1.reward = 0.5
        player1.status = "DONE"
        player2.reward = 0.5
        player2.status = "DONE"
        return state

    # return state if game is ongoing

    return state


def renderer(state, env):
    # Hard Code Weapon Names and Symbols for Rendering

    weapon_name = ["Rock", "Paper", "Scissors", "Spock", "Lizard", "Airplane", "Sun", "Moon", "Camera", "Grass", "Fire", "Film", "Spanner", "Toilet", "School", "Air", "Death", "Planet", "Curse", "Guitar", "Lock", "Bowl", "Pickaxe", "Cup", "Peace", "Beer", "Computer", "Rain", "Castle", "Water", "Snake", "TV", "Blood", "Rainbow", "Porcupine", "UFO", "Eagle", "Alien", "Monkey", "Prayer", "King", "Mountain", "Queen", "Satan", "Wizard", "Dragon", "Mermaid", "Diamond", "Police", "Trophy", "Woman", "Money", "Baby", "Devil", "Man", "Link", "Home", "Video Game", "Train", "Math", "Car", "Robot", "Noise", "Heart", "Bicycle", "Electricity", "Tree", "Lightning", "Potato", "Ghost", "Duck", "Power", "Wolf", "Microscope", "Cat", "Nuke", "Chicken", "Cloud", "Fish", "Truck", "Spider", "Helicopter", "Bee", "Bomb", "Brain", "Tornado", "Community", "Sand", "Zombie", "Pit", "Bank", "Chain", "Vampire", "Gun", "Bath", "Law", "Monument", "Baloon", "Pancake", "Sword", "Book"]
    weapon_emoji = ["👊🏽", "📄", "✂️", "🖖", "🦎", "✈️", "☀️", "🌙", "📷", "🌱", "🔥", "🎥", "🔧", "🚽", "🏫", "💨", "☠", "🌎", "🥀", "🎸", "🔒", "🥣", "⛏️", "☕", "🕊️", "🍺", "💻", "🌧️", "🏰", "💧", "🐍", "📺", "💉", "🌈", "🦔", "🛸", "🦅", "👽", "🐒", "🙏🏽", "🤴🏼", "🏔️", "👸🏽", "😈", "🧙🏼‍♂️", "🐉", "🧜🏽‍♀️", "💎", "👮🏽‍♀️", "🏆", "👩🏻", "💰", "👶🏽", "👹", "👨🏾", "🔗", "🏠", "🎮", "🚂", "🔢", "🚗", "🤖", "🔔", "❤️", "🚲", "💡", "🌲", "⚡", "🥔", "👻", "🦆", "🔋", "🐺", "🔬", "🐈", "☢️", "🐓", "☁️", "🐟", "🚚", "🕷️", "🚁", "🐝", "💣", "🧠", "🌪️", "👥", "🏖️", "🧟‍♂️", "🕳️", "🏦", "⛓️", "🧛🏽‍♂️", "🔫", "🛁", "⚖️", "🏛️", "🎈", "🥞", "🗡️", "📖"]

    def get_weapon_name(number):
        if number < 0:
            return "error"
        elif number > len(weapon_emoji):
            return str(number)
        else:
            return weapon_emoji[number - 1]

    rounds_played = len(state[0].observation.results)
    p1_score = state[0].observation.results.count(1)
    p2_score = state[0].observation.results.count(2)
    board = ""

    # This line prints results each round, good for debugging
    for i in range(rounds_played):
        board = board + f"Round {i + 1}: {get_weapon_name(state[0].observation.p1_moves[i])} vs {get_weapon_name(state[0].observation.p2_moves[i])}, Score: {state[0].observation.results[0:(i + 1)].count(1)} to {state[0].observation.results[0:(i + 1)].count(2)}\n"

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


def random_agent(observation, configuration):
    return random.choice(range(1, (configuration.weapons + 1)))


def rock(observation, configuration):
    return 1


def paper(observation, configuration):
    return 2


def scissors(observation, configuration):
    return 3


def rockish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.choice(range(1, (configuration.weapons + 1)))
    else:
        return 1


def paperish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.choice(range(1, (configuration.weapons + 1)))
    else:
        return 2


def scissorsish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.choice(range(1, (configuration.weapons + 1)))
    else:
        return 3


def copy_opponent(observation, configuration):
    me = observation.mark
    opponent = 2 if me == 1 else 1
    rounds_played = len(observation.results)

    if rounds_played > 0:
        if me == 1:
            weapon = observation.p2_moves[-1]
        else:
            weapon = observation.p1_moves[-1]
    else:
        weapon = random.choice(range(1, (configuration.weapons + 1)))

    return weapon


def reactionary(observation, configuration):
    me = observation.mark
    opponent = 2 if me == 1 else 1
    rounds_played = len(observation.results)

    if rounds_played > 0:
        if me == 1:
            if observation.results[-1] == 1:
                weapon = observation.p1_moves[-1]
            else:
                if observation.p2_moves[-1] < configuration.weapons:
                    weapon = observation.p2_moves[-1] + 1
                else:
                    weapon = 1
        else:
            if observation.results[-1] == 2:
                weapon = observation.p2_moves[-1]
            else:
                if observation.p1_moves[-1] < configuration.weapons:
                    weapon = observation.p1_moves[-1] + 1
                else:
                    weapon = 1
    else:
        return random.choice(range(1, (configuration.weapons + 1)))

    return weapon


def counter_reactionary(observation, configuration):
    me = observation.mark
    opponent = 2 if me == 1 else 1
    rounds_played = len(observation.results)

    if rounds_played > 0:
        if me == 1:
            if observation.results[-1] < 2:
                if (observation.p1_moves[-1] + 2) <= configuration.weapons:
                    weapon = observation.p1_moves[-1] + 2
                else:
                    weapon = observation.p1_moves[-1] + 2 - configuration.weapons
            else:
                if observation.p2_moves[-1] < configuration.weapons:
                    weapon = observation.p2_moves[-1] + 1
                else:
                    weapon = 1
        else:
            if observation.results[-1] > 1:
                if (observation.p2_moves[-1] + 2) <= configuration.weapons:
                    weapon = observation.p2_moves[-1] + 2
                else:
                    weapon = observation.p2_moves[-1] + 2 - configuration.weapons
            else:
                if observation.p1_moves[-1] < configuration.weapons:
                    weapon = observation.p1_moves[-1] + 1
                else:
                    weapon = 1
    else:
        return random.choice(range(1, (configuration.weapons + 1)))

    return weapon


agents = {"random": random_agent, "rock": rock, "paper": paper, "scissors": scissors, "rockish": rockish, "paperish": paperish, "scissorsish": scissorsish, "copy_opponent": copy_opponent, "reactionary": reactionary, "counter_reactionary": counter_reactionary}