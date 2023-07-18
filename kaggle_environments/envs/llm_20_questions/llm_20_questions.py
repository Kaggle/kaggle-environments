import json
import string
from os import path
from random import choice


def guesser_agent(obs):
    return "Is it a person?"
    

def answerer_agent(obs):
    return "yes"


agents = {"random": guesser_agent, "reaction": answerer_agent}


def interpreter(state, env):
    if env.done:
        return state

    # Isolate the active and inactive agents.
    active = state[0] if state[0].status == "ACTIVE" else state[1]
    inactive = state[0] if state[0].status == "INACTIVE" else state[1]
    if active.status != "ACTIVE" or inactive.status != "INACTIVE":
        active.status = "DONE" if active.status == "ACTIVE" else active.status
        inactive.status = "DONE" if inactive.status == "INACTIVE" else inactive.status
        return state

    if active.observation.role == "guesser":
        active.observation.questions.append(active.action)
        inactive.observation.questions.append(active.action)
        if keyword_guessed(active.action):
            active.reward, inactive.reward = (40 - active.observation.step) / 2
            active.status, inactive.status = "DONE"
            active.observation.keyword = keyword
            return state
    else:
        active.observation.keyword = keyword
        response = active.action
        if response.lower().__contains__("yes"):
            response = "yes"
        elif response.lower().__contains__("no"):
            response = "no"
        else:
            response = "maybe"
        active.observation.answers.append(response)
        inactive.observation.answers.append(response)

    if active.observation.step == 39:
        active.reward, inactive.reward = -1

    # Swap active and inactive agents to switch turns.
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state


def renderer(state, env):
    transcript = ""

    for s in state:
        if s.observation.role == "answerer":
            for i in range(0, len(s.observation.answers)):
                transcript = "{}Q: {} A: {}\n".format(transcript, s.observation.questions[i], s.observation.answers[i])

    return transcript


jsonpath = path.abspath(path.join(path.dirname(__file__), "llm_20_questions.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    return


def keyword_guessed( guess: str) -> bool:
    keyword = "paris"
    alts = []
    def normalize(s: str) -> str:
      t = str.maketrans("", "", string.punctuation)
      return s.lower().replace("the", "").replace(" ", "").translate(t)

    if normalize(guess) == normalize(keyword):
      return True
    for s in alts:
      if normalize(s) == normalize(guess):
        return True

    return False
