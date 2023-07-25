import json
import pandas as pd
import string
import torch
import warnings

from os import path
from pathlib import Path
from random import choice
from string import Template
from transformers import T5Tokenizer, T5ForConditionalGeneration


warnings.simplefilter("ignore")

# change to download path
# possibly upgrade to bigger model if performance is bad
llm = "/kaggle/input/flan-t5/pytorch/base/2"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained(llm).to(device)
tokenizer = T5Tokenizer.from_pretrained(llm)

keyword = "paris"
category = "city"
alts = []


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
        if active.observation.turnType == "ask":
            active.observation.questions.append(active.action)
            inactive.observation.questions.append(active.action)
        elif active.observation.turnType == "guess":
            active.observation.guesses.append(active.action)
            inactive.observation.guesses.append(active.action)
        if keyword_guessed(active.action):
            active.reward = (61 - active.observation.step) / 2
            inactive.reward = (61 - active.observation.step) / 2
            active.status = "DONE"
            inactive.status = "DONE"
            active.observation.keyword = keyword
            active.observation.category = category
            return state
    else:
        active.observation.keyword = keyword
        active.observation.category = category
        response = active.action
        if response.lower().__contains__("yes"):
            response = "yes"
        elif response.lower().__contains__("no"):
            response = "no"
        else:
            response = "maybe"
        active.observation.answers.append(response)
        inactive.observation.answers.append(response)

    if active.observation.role == "guesser":
        if active.observation.step == 59:
            active.observation.keyword = keyword
            active.observation.category = category
            active.reward = -1
            inactive.reward = -1
    else:
        if inactive.observation.step == 59:
            inactive.observation.keyword = keyword
            inactive.observation.category = category
            active.reward = -1
            inactive.reward = -1

    # Swap active and inactive agents to switch turns if guesser has gotten to both ask a question and guess the answer or the answerer has answered.
    if active.observation.turnType == "guess":
        active.observation.turnType = "ask"
        return state
    
    if active.observation.turnType == "ask":
        active.observation.turnType = "guess"
    
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state


def renderer(state, env):
    transcript = ""

    for s in state:
        if s.observation.role == "guesser":
            for i in range(0, len(s.observation.guesses)):
                transcript = "{}Q: {} A: {}\nG: {}\n".format(
                    transcript, s.observation.questions[i],
                    s.observation.answers[i],
                    s.observation.guesses[i]
                )

    print(transcript)
    return transcript


jsonpath = path.abspath(path.join(path.dirname(__file__), "llm_20_questions.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    return


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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer[0]
