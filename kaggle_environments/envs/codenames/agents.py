import random

def random_agent(obs, config):
    if obs.current_turn in [0, 2]:
        return {"clue": "random", "number": 1}
    else:
        valid_guesses = [i for i in range(25) if not obs.revealed[i]]
        if valid_guesses:
            return random.choice(valid_guesses)
        return -1
