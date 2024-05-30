
from kaggle_environments import make

def custom_questioner(obs):
    if obs.turnType == "guess":
        return "banana"
    return "Is it a banana?"

def custom_answerer():
    return "no"

def bad_answerer():
    return "maybe?"

def error_agent():
    raise ValueError

def test_llm_20_q_completes():
    env = make("llm_20_questions", debug=True)
    env.run([custom_questioner, custom_answerer, custom_questioner, custom_answerer])
    json = env.toJSON()
    assert json["name"] == "llm_20_questions"
    assert json["statuses"] == ["DONE", "DONE", "DONE", "DONE"]

def test_llm_20_q_errors_on_bad_answer():
    env = make("llm_20_questions", debug=True)
    env.run([custom_questioner, custom_answerer, custom_questioner, bad_answerer])
    json = env.toJSON()
    assert json["name"] == "llm_20_questions"
    assert json["rewards"] == [1, 1, 1, None]
    assert json["statuses"] == ["DONE", "DONE", "DONE", "ERROR"]
    print(len(json["steps"]))
    assert len(json["steps"]) == 3

def test_llm_20_q_errors_on_error_answer():
    env = make("llm_20_questions", debug=True)
    env.run([custom_questioner, custom_answerer, custom_questioner, error_agent])
    json = env.toJSON()
    assert json["name"] == "llm_20_questions"
    assert json["rewards"] == [1, 1, 1, None]
    assert json["statuses"] == ["DONE", "DONE", "DONE", "ERROR"]
    print(len(json["steps"]))
    assert len(json["steps"]) == 3

def test_llm_20_q_errors_on_error_question():
    env = make("llm_20_questions", debug=True)
    env.run([custom_questioner, custom_answerer, error_agent, custom_answerer])
    json = env.toJSON()
    assert json["name"] == "llm_20_questions"
    assert json["rewards"] == [1, 1, None, 1]
    assert json["statuses"] == ["DONE", "DONE", "ERROR", "DONE"]
    print(len(json["steps"]))
    assert len(json["steps"]) == 2