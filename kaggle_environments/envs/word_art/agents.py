import random


def random_agent(observation, configuration):
    """A trivial baseline: artists emit a short placeholder; guessers pick a random letter."""
    if observation.role == "artist":
        return "* * *\n * *\n* * *"
    # Guesser
    letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return letter


def cheating_agent(observation, configuration):
    """A perfect agent: artists transmit the word verbatim via the art channel,
    guessers parse it back. Useful for tests since the art channel is free-form text.
    """
    if observation.role == "artist":
        return observation.target_word
    return observation.teammate_art


def silent_agent(observation, configuration):
    """Always submits an empty string."""
    return ""


agents = {
    "random": random_agent,
    "cheating": cheating_agent,
    "silent": silent_agent,
}
