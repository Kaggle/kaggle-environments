import json
from os import path
from .agents import agents as all_agents
from .utils import get_score


def interpreter(state, env):
    player1 = state[0]
    player2 = state[1]

    # Specification can fully handle the reset.
    if env.done:
        return state

    def is_valid_action(player, sign_count):
        return (
            player.action is not None and
            isinstance(player.action, int) and
            0 <= player.action < sign_count
        )

    # Check for validity of actions
    is_player1_valid = is_valid_action(player1, env.configuration.signs)
    is_player2_valid = is_valid_action(player2, env.configuration.signs)
    if not is_player2_valid:
        player2.status = "INVALID"
        player2.reward = 0

        if is_player1_valid:
            player1.status = "DONE"
            player1.reward = 1
            return state

    if not is_player1_valid:
        player1.status = "INVALID"
        player1.reward = 0

        if is_player2_valid:
            player2.status = "DONE"
            player2.reward = 1
            return state
        else:
            return state

    score = get_score(player1.action, player2.action)
    player1.observation.lastOpponentAction = player2.action
    player1.reward += score
    player2.observation.lastOpponentAction = player1.action
    player2.reward -= score
    player1.observation.reward = int(player1.reward)
    player2.observation.reward = int(player2.reward)
    remaining_steps = env.configuration.episodeSteps - player1.observation.step - 1

    # This is the last step
    if remaining_steps <= 1:
        player1.status = "DONE"
        player2.status = "DONE"
        # Player performance too similar, consider the match a tie.
        if abs(player1.reward) < env.configuration.tieRewardThreshold:
            player1.reward = 0
            player2.reward = 0
    return state


def renderer(state, env):
    sign_names = ["Rock", "Paper", "Scissors", "Spock", "Lizard"]
    rounds_played = len(env.steps)
    board = ""

    # This line prints results each round, good for debugging
    for i in range(1, rounds_played):
        step = env.steps[i]
        right_move = step[0].observation.lastOpponentAction
        left_move = step[1].observation.lastOpponentAction
        board += f"Round {i}: {sign_names[left_move]} vs {sign_names[right_move]}, Score: {step[0].reward} to {step[1].reward}\n"

    board += f"Game ended on round {rounds_played - 1}, final score: {state[0].reward} to {state[0].reward}\n"
    return board


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "rps.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    js_path = path.abspath(path.join(dir_path, "rps.js"))
    with open(js_path, encoding="utf-8") as js_file:
        return js_file.read()


agents = all_agents
