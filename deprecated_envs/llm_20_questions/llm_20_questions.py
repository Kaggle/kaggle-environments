import json
import os
import random
import string
from os import path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .keywords import KEYWORDS_JSON

llm_parent_dir = "/kaggle/input/flan-t5/pytorch/large"

device = None
model = None
tokenizer = None
model_initialized = False

ERROR = "ERROR"
DONE = "DONE"
INACTIVE = "INACTIVE"
ACTIVE = "ACTIVE"
TIMEOUT = "TIMEOUT"

GUESS = "guess"
ASK = "ask"
GUESSER = "guesser"
ANSWERER = "answerer"

RANDOM_GUESSER = "random_guesser"
RANDOM_ANSWERER = "random_answerer"


def weighted_random_category(keywords_list):
    try:
        return random.choices(population=keywords_list, weights=[len(entry["words"]) for entry in keywords_list], k=1)[
            0
        ]
    except:
        pass
    return random.choice(keywords_list)


keywords_list = json.loads(KEYWORDS_JSON)
keyword_cat = weighted_random_category(keywords_list)
category = keyword_cat["category"]
keyword_obj = random.choice(keyword_cat["words"])
keyword = keyword_obj["keyword"]
alts = keyword_obj["alts"]

try:
    with open("/kaggle_simulations/data.json") as f:
        json_content = f.read()
        d_keywords_list = json.loads(json_content)
        d_keyword_cat = weighted_random_category(d_keywords_list)
        d_category = d_keyword_cat["category"]
        d_keyword_obj = random.choice(d_keyword_cat["words"])
        d_keyword = d_keyword_obj["keyword"]
        d_alts = d_keyword_obj["alts"]
        # re-assign
        category = d_category
        keyword = d_keyword
        alts = d_alts
except:
    pass


def random_guesser(obs):
    if obs.turnType == GUESS:
        return "banana"
    if random.random() < 0.5:
        return "Is is a person?"
    if random.random() < 0.5:
        return "Is it a place?"
    return "Is it a thing?"


def random_answerer():
    if random.random() > 0.5:
        return "yes"
    return "no"


def guesser_agent(obs):
    info_prompt = """You are playing a game of 20 questions where you ask the questions and try to figure out the keyword, which will be a real or fictional person, place, or thing. \nHere is what you know so far:\n{q_a_thread}"""
    questions_prompt = """Ask one yes or no question."""
    guess_prompt = """Guess the keyword. Only respond with the exact word/phrase. For example, if you think the keyword is [paris], don't respond with [I think the keyword is paris] or [Is the keyword Paris?]. Respond only with the word [paris]."""

    q_a_thread = ""
    for i in range(0, len(obs.answers)):
        q_a_thread = "{}Q: {} A: {}\n".format(q_a_thread, obs.questions[i], obs.answers[i])

    prompt = ""
    if obs.turnType == ASK:
        prompt = "{}{}".format(info_prompt.format(q_a_thread=q_a_thread), questions_prompt)
    elif obs.turnType == GUESS:
        prompt = "{}{}".format(info_prompt.format(q_a_thread=q_a_thread), guess_prompt)
    else:
        return ""

    return call_llm(prompt)


def answerer_agent(obs):
    info_prompt = """You are a very precise answerer in a game of 20 questions. The keyword that the questioner is trying to guess is [the {category} {keyword}]. """
    answer_question_prompt = """Answer the following question with only yes, no, or if unsure maybe: {question}"""

    if obs.turnType == "answer":
        prompt = "{}{}".format(
            info_prompt.format(category=category, keyword=keyword),
            answer_question_prompt.format(question=obs.questions[-1]),
        )
        return call_llm(prompt)
    else:
        return ""


agents = {
    GUESSER: guesser_agent,
    ANSWERER: answerer_agent,
    RANDOM_ANSWERER: random_answerer,
    RANDOM_GUESSER: random_guesser,
}


def guesser_action(active, inactive, step):
    inactive.observation.keyword = keyword
    inactive.observation.category = category
    guessed = False
    bad_guess = False
    if not active.action:
        active.status = ERROR
        bad_guess = True
    elif active.observation.turnType == ASK:
        question = active.action[:750]
        active.observation.questions.append(question)
        inactive.observation.questions.append(question)
    elif active.observation.turnType == GUESS:
        guess = active.action[:100]
        active.observation.guesses.append(guess)
        inactive.observation.guesses.append(guess)
    if active.action and keyword_guessed(active.action):
        guessed = True
        score = 20 - int(step / 3)
        end_game(active, score, DONE)
        end_game(inactive, score, DONE)
    return [guessed, bad_guess]


def end_game(agent, reward, status):
    agent.observation.keyword = keyword
    agent.observation.category = category
    agent.reward = reward
    agent.status = status


def answerer_action(active, inactive):
    active.observation.keyword = keyword
    active.observation.category = category
    response = active.action
    bad_response = False
    if not response:
        response = "none"
        end_game(active, -1, ERROR)
        end_game(inactive, 1, DONE)
        bad_response = True
    elif "yes" in response.lower():
        response = "yes"
    elif "no" in response.lower():
        response = "no"
    else:
        response = "maybe"
        end_game(active, -1, ERROR)
        end_game(inactive, 1, DONE)
        bad_response = True
    active.observation.answers.append(response)
    inactive.observation.answers.append(response)
    return bad_response


def increment_turn(active, inactive, step, guessed):
    if step == 59 and not guessed:
        end_game(active, -1, DONE)
        end_game(inactive, -1, DONE)
    elif active.observation.turnType == "guess":
        active.observation.turnType = "ask"
    elif active.observation.turnType == "ask":
        active.observation.turnType = "guess"
        active.status = INACTIVE
        inactive.status = ACTIVE
    else:
        active.status = INACTIVE
        inactive.status = ACTIVE


def interpreter(state, env):
    if env.done:
        return state

    # Isolate the active and inactive agents.
    active1 = state[0] if state[0].status == ACTIVE else state[1]
    inactive1 = state[0] if state[0].status == INACTIVE else state[1]
    active2 = state[2] if state[2].status == ACTIVE else state[3]
    inactive2 = state[2] if state[2].status == INACTIVE else state[3]
    if active1.status == DONE and inactive1.status == DONE:
        active1 = None
        inactive1 = None
    if active2.status == DONE or inactive2.status == DONE:
        active2 = None
        inactive2 = None
    if active1 is None and inactive1 is None and active2 is None and inactive2 is None:
        return state

    step = state[0].observation.step

    end_early = (active1 and active1.status) in (TIMEOUT, ERROR) or (active2 and active2.status in (TIMEOUT, ERROR))
    one_guessed = False
    two_guessed = False
    one_bad_guess = False
    two_bad_guess = False
    one_bad_response = False
    two_bad_response = False

    if active1 is None or active2 is None:
        raise ValueError

    if active1.observation.role == GUESSER:
        [one_guessed, one_bad_guess] = guesser_action(active1, inactive1, step)
    else:
        one_bad_response = answerer_action(active1, inactive1)

    if active2.observation.role == GUESSER:
        [two_guessed, two_bad_guess] = guesser_action(active2, inactive2, step)
    else:
        two_bad_response = answerer_action(active2, inactive2)

    if active1.status in (TIMEOUT, ERROR) or one_bad_response or one_bad_guess:
        end_game(active1, -1, active1.status)
        end_game(inactive1, 1, DONE)
    elif end_early or two_bad_response or two_bad_guess:
        end_game(active1, 1, DONE)
        end_game(inactive1, 1, DONE)
    else:
        increment_turn(active1, inactive1, step, one_guessed)

    if active2.status in (TIMEOUT, ERROR) or two_bad_response or two_bad_guess:
        end_game(active2, -1, active2.status)
        end_game(inactive2, 1, DONE)
    elif end_early or one_bad_response or one_bad_guess:
        end_game(active2, 1, DONE)
        end_game(inactive2, 1, DONE)
    else:
        increment_turn(active2, inactive2, step, two_guessed)

    # make sure to end the game if only one team guessed correctly this round
    if one_guessed and not two_guessed:
        end_game(active2, 0, DONE)
        end_game(inactive2, 0, DONE)
    elif two_guessed and not one_guessed:
        end_game(active1, 0, DONE)
        end_game(inactive1, 0, DONE)

    return state


def renderer(state, env):
    for s in state:
        print("role: ", s.observation.role)
        if s.observation.role == GUESSER:
            transcript = ""
            for i in range(0, len(s.observation.guesses)):
                transcript = "{}Q: {} A: {}\nG: {}\n".format(
                    transcript, s.observation.questions[i], s.observation.answers[i], s.observation.guesses[i]
                )
            print(transcript)

        print("keyword: ", s.observation.keyword)
        print("score: ", s.reward)
        print("")
        print("")
        print("")

    return ""


jsonpath = path.abspath(path.join(path.dirname(__file__), "llm_20_questions.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    jspath = path.abspath(path.join(path.dirname(__file__), "llm_20_questions.js"))
    with open(jspath, encoding="utf-8") as f:
        return f.read()


def normalize(s: str) -> str:
    t = str.maketrans("", "", string.punctuation)
    return s.lower().replace("the", "").replace(" ", "").translate(t)


def compare_words(a, b) -> bool:
    a = normalize(a)
    b = normalize(b)
    if a == b:
        return True
    # don't check for plurals if string is too short
    if len(a) < 3 or len(b) < 3:
        return False
    # accept common plurals
    if a[-1] == "s" and a[:-1] == b:
        return True
    if b[-1] == "s" and a == b[:-1]:
        return True
    if a[-2:] == "es" and a[:-2] == b:
        return True
    if b[-2:] == "es" and a == b[:-2]:
        return True
    return False


def keyword_guessed(guess: str) -> bool:
    if compare_words(guess, keyword):
        return True
    for s in alts:
        if compare_words(s, guess):
            return True
    return False


def call_llm(prompt: str) -> str:
    global model_initialized
    global device
    global model
    global tokenizer

    if not model_initialized:
        if os.path.exists(llm_parent_dir) and len(os.listdir(llm_parent_dir)) > 0:
            dirs = os.listdir(llm_parent_dir)
            llm_dir = "{}/{}".format(llm_parent_dir, dirs[0])
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = T5ForConditionalGeneration.from_pretrained(llm_dir).to(device)
            tokenizer = T5Tokenizer.from_pretrained(llm_dir)
            model_initialized = True
        else:
            print("t5-flan model required to use default agents. Add any version of the large model.")
            print("https://www.kaggle.com/models/google/flan-t5/frameworks/pyTorch/variations/large.")
            raise Exception("t5-flan model required to use default agents.")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer[0]
