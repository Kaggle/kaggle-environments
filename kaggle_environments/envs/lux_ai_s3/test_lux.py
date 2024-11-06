import sys
from kaggle_environments import make

def xtest_lux_completes():
    env = make("lux_ai_s3", debug=True)
    env.run(["random_agent", "random_agent"])
    json = env.toJSON()
    assert json["name"] == "lux_ai_s3"
    assert json["statuses"] == ["DONE", "DONE"]
