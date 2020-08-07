import random


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