from kaggle_environments import make

def test_chess_inits():
    env = make("chess", debug=True)
    env.run(["random", "random"])
    json = env.toJSON()
    assert json["name"] == "chess"
    assert json["statuses"] == ["DONE", "DONE"]

def test_chess_three_fold():
    env = make("chess", debug=True)
    env.run(["king_shuffle", "king_shuffle"])
    json = env.toJSON()
    assert json["name"] == "chess"
    assert json["statuses"] == ["DONE", "DONE"]
    assert json["rewards"] == [0, 0]

def test_chess_100_move_rule():
    env = make("chess", debug=True)
    env.run(["board_shuffle", "board_shuffle"])
    json = env.toJSON()
    assert json["name"] == "chess"
    assert json["statuses"] == ["DONE", "DONE"]
    assert json["rewards"] == [0, 0]
