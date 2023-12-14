import json
import os
import pandas as pd
import random
import string
import torch

from .keywords import KEYWORDS_JSON
from os import path
from pathlib import Path
from random import choice
from string import Template
from transformers import T5Tokenizer, T5ForConditionalGeneration


llm_parent_dir = "/kaggle/input/flan-t5/pytorch/large"

device = None
model = None
tokenizer = None
model_initialized = False

keywords_list = json.loads(KEYWORDS_JSON)
keyword_cat = random.choice(keywords_list)
category = keyword_cat["category"]
keyword_obj = random.choice(keyword_cat["words"])
keyword = keyword_obj["keyword"]
alts = keyword_obj["alts"]


def guesser_agent(obs):
    info_prompt = """You are playing a game of 20 questions where you ask the questions and try to figure out the keyword, which will be a real or fictional person, place, or thing. \nHere is what you know so far:\n{q_a_thread}"""
    questions_prompt = """Ask one yes or no question."""
    guess_prompt = """Guess the keyword. Only respond with the exact word/phrase. For example, if you think the keyword is [paris], don't respond with [I think the keyword is paris] or [Is the kewyord Paris?]. Respond only with the word [paris]."""

    q_a_thread = ""
    for i in range(0, len(obs.answers)):
        q_a_thread = "{}Q: {} A: {}\n".format(
            q_a_thread,
            obs.questions[i],
            obs.answers[i]
        )

    prompt = ""
    if obs.turnType == "ask":
        prompt = "{}{}".format(
            info_prompt.format(q_a_thread=q_a_thread),
            questions_prompt
        )
    elif obs.turnType == "guess":
        prompt = "{}{}".format(
            info_prompt.format(q_a_thread=q_a_thread),
            guess_prompt
        )
    else:
        return ""
    
    return call_llm(prompt)

    

def answerer_agent(obs):
    info_prompt = """You are a very precise answerer in a game of 20 questions. The keyword that the questioner is trying to guess is [the {category} {keyword}]. """
    answer_question_prompt = """Answer the following question with only yes, no, or if unsure maybe: {question}"""

    if obs.turnType == "answer":
        prompt = "{}{}".format(
            info_prompt.format(category=category,keyword=keyword),
            answer_question_prompt.format(question=obs.questions[-1])
        )
        return call_llm(prompt)
    else: 
        return ""


agents = {"guesser": guesser_agent, "answerer": answerer_agent}


def interpreter(state, env):
    if env.done:
        return state

    # Isolate the active and inactive agents.
    active1 = state[0] if state[0].status == "ACTIVE" else state[1]
    inactive1 = state[0] if state[0].status == "INACTIVE" else state[1]
    active2 = state[2] if state[2].status == "ACTIVE" else state[3]
    inactive2 = state[2] if state[2].status == "INACTIVE" else state[3]
    if active1.status == "DONE" and inactive1.status == "DONE":
        active1 = None
        inactive1 = None
    if active2.status == "DONE" or inactive2.status == "DONE":
        active2 = None
        inactive2 = None
    if active1 is None and inactive1 is None and active2 is None and inactive2 is None:
        return state

    step = state[0].observation.step

    if active1 is not None:
        guessed = False
        if active1.observation.role == "guesser":
            if not active1.action:
                active1.status = "ERROR"
            elif active1.observation.turnType == "ask":
                active1.observation.questions.append(active1.action)
                inactive1.observation.questions.append(active1.action)
            elif active1.observation.turnType == "guess":
                active1.observation.guesses.append(active1.action)
                inactive1.observation.guesses.append(active1.action)
            if active1.action and keyword_guessed(active1.action):
                guessed = True
                score = 20 - int(step / 3)
                active1.reward = score
                inactive1.reward = score
                active1.status = "DONE"
                inactive1.status = "DONE"
                active1.observation.keyword = keyword
                active1.observation.category = category
            inactive1.observation.keyword = keyword
            inactive1.observation.category = category
        else:
            active1.observation.keyword = keyword
            active1.observation.category = category
            response = active1.action
            if not response:
                active1.status = "ERROR"
            elif response and response.lower().__contains__("yes"):
                response = "yes"
            elif response and response.lower().__contains__("no"):
                response = "no"
            else:
                active1.status = "ERROR"
            active1.observation.answers.append(response)
            inactive1.observation.answers.append(response)

        if step == 59 and not guessed:
            active1.observation.keyword = keyword
            active1.observation.category = category
            inactive1.observation.keyword = keyword
            inactive1.observation.category = category
            active1.reward = -1
            inactive1.reward = -1
            active1.status = "DONE"
            inactive1.status = "DONE"
        elif active1.observation.turnType == "guess":
            active1.observation.turnType = "ask"
        elif active1.observation.turnType == "ask":
            active1.observation.turnType = "guess"
            active1.status = "INACTIVE"
            inactive1.status = "ACTIVE"
        else:
            active1.status = "INACTIVE"
            inactive1.status = "ACTIVE"
    
    if active2 is not None:
        guessed = False
        if active2.observation.role == "guesser":
            if not active2.action:
                active2.status = "ERROR"
            elif active2.observation.turnType == "ask":
                active2.observation.questions.append(active2.action)
                inactive2.observation.questions.append(active2.action)
            elif active2.observation.turnType == "guess":
                active2.observation.guesses.append(active2.action)
                inactive2.observation.guesses.append(active2.action)
            if active2.action and keyword_guessed(active2.action):
                guessed = True
                score = 20 - int(step / 3)
                active2.reward = score
                inactive2.reward = score
                active2.status = "DONE"
                inactive2.status = "DONE"
                active2.observation.keyword = keyword
                active2.observation.category = category
            inactive2.observation.keyword = keyword
            inactive2.observation.category = category
        else:
            active2.observation.keyword = keyword
            active2.observation.category = category
            response = active2.action
            if not response:
                active2.status = "ERROR"
            elif response.lower().__contains__("yes"):
                response = "yes"
            elif response.lower().__contains__("no"):
                response = "no"
            else:
                active2.status = "ERROR"
            active2.observation.answers.append(response)
            inactive2.observation.answers.append(response)

        if step == 59 and not guessed:
            active2.observation.keyword = keyword
            active2.observation.category = category
            inactive2.observation.keyword = keyword
            inactive2.observation.category = category
            active2.reward = -1
            inactive2.reward = -1
            active2.status = "DONE"
            inactive2.status = "DONE"
        elif active2.observation.turnType == "guess":
            active2.observation.turnType = "ask"
        elif active2.observation.turnType == "ask":
            active2.observation.turnType = "guess"
            active2.status = "INACTIVE"
            inactive2.status = "ACTIVE"
        else:
            active2.status = "INACTIVE"
            inactive2.status = "ACTIVE"

    return state


def renderer(state, env):

    for s in state:
        print("role: ", s.observation.role)
        if s.observation.role == "guesser":
            transcript = ""
            for i in range(0, len(s.observation.guesses)):
                transcript = "{}Q: {} A: {}\nG: {}\n".format(
                    transcript, s.observation.questions[i],
                    s.observation.answers[i],
                    s.observation.guesses[i]
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
    with open(jspath) as f:
        return f.read()


def keyword_guessed(guess: str) -> bool:
    def normalize(s: str) -> str:
      t = str.maketrans("", "", string.punctuation)
      return s.lower().replace("the", "").replace(" ", "").translate(t)

    if normalize(guess) == normalize(keyword):
      return True
    for s in alts:
      if normalize(s) == normalize(guess):
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
