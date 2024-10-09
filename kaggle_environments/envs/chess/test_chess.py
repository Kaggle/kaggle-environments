from kaggle_environments import make

def test_chess_inits():
    env = make("chess", debug=True)
    env.run(["random", "random"])
    json = env.toJSON()
    assert json["name"] == "chess"
    assert json["statuses"] == ["ERROR", "DONE"]