
from kaggle_environments import make

def custom_questioner(obs):
    if obs.turnType == "guess":
        return "banana"
    return "Is it a banana?"

def custom_answerer():
    return "no"

def test_lux_completes():
    env = make("llm_20_questions", debug=True)
    env.run([custom_questioner, custom_answerer, custom_questioner, custom_answerer])
    json = env.toJSON()
    assert json["name"] == "llm_20_questions"
    assert json["statuses"] == ["DONE", "DONE", "DONE", "DONE"]