
from kaggle_environments import make
from llm_20_questions import random_guesser, random_answerer

def test_lux_completes():
    env = make("llm_20_questions", debug=True)
    env.run([random_guesser, random_answerer, random_guesser, random_answerer])
    json = env.toJSON()
    assert json["name"] == "llm_20_questions"
    assert json["statuses"] == ["DONE", "DONE", "DONE", "DONE"]