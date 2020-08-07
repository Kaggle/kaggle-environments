import random


def random_agent(observation, configuration):
    return random.randint(0, configuration.weapons - 1)


def rock(observation, configuration):
    return 0


def paper(observation, configuration):
    return 1


def scissors(observation, configuration):
    return 2


def rockish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.randint(0, configuration.weapons - 1)
    else:
        return 0


def paperish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.randint(0, configuration.weapons - 1)
    else:
        return 1


def scissorsish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.randint(0, configuration.weapons - 1)
    else:
        return 2


def copy_opponent(observation, configuration):
    me = observation.mark
    opponent = 2 if me == 1 else 1
    rounds_played = len(observation.results)

    if rounds_played > 0:
        if me == 1:
            return observation.p2_moves[-1]
        else:
            return observation.p1_moves[-1]
    else:
        return random.randint(0, configuration.weapons - 1)


def reactionary(observation, configuration):
    me = observation.mark
    if me == 1:
        my_moves = observation.p1_moves
        opponent_moves = observation.p2_moves
    else:
        my_moves = observation.p2_moves
        opponent_moves = observation.p1_moves

    if len(observation.results) == 0:
        return random.randint(0, configuration.weapons - 1)
    
    if observation.results[-1] == me:
        return my_moves[-1]
    else:
        return (opponent_moves[-1] + 1) % configuration.weapons


def counter_reactionary(observation, configuration):
    me = observation.mark
    if me == 1:
        my_moves = observation.p1_moves
        opponent_moves = observation.p2_moves
    else:
        my_moves = observation.p2_moves
        opponent_moves = observation.p1_moves

    if len(observation.results) == 0:
        return random.randint(0, configuration.weapons - 1)

    if observation.results[-1] < 2:
        return (my_moves[-1] + 2) % configuration.weapons
    else:
        return (opponent_moves[-1] + 1) % configuration.weapons


agents = {
    "random": random_agent,
    "rock": rock,
    "paper": paper,
    "scissors": scissors,
    "rockish": rockish,
    "paperish": paperish,
    "scissorsish": scissorsish,
    "copy_opponent": copy_opponent,
    "reactionary": reactionary,
    "counter_reactionary": counter_reactionary
}