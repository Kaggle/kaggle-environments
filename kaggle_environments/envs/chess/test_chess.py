from kaggle_environments import make
from Chessnut import Game
from chess import is_insufficient_material

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
    assert json["rewards"] == [1.0, 1.0]

def test_chess_100_move_rule():
    env = make("chess", debug=True)
    env.run(["board_shuffle", "board_shuffle"])
    json = env.toJSON()
    assert json["name"] == "chess"
    assert json["statuses"] == ["DONE", "DONE"]
    assert json["rewards"] == [1.0, 1.0]

def test_sufficient_material():
    game = Game()
    assert not is_insufficient_material(game.board)

def test_insufficient_material_with_two_kings():
    game = Game('8/8/K7/8/8/3k4/8/8 w - - 58 282')
    assert is_insufficient_material(game.board)

def test_insufficient_material_with_two_kings_and_bishop():
    game = Game('6k1/8/7B/8/8/8/8/2K5 b - - 90 250')
    assert is_insufficient_material(game.board)

def test_insufficient_material_with_two_kings_and_two_knights():
    game = Game('6k1/8/6NN/8/8/8/8/2K5 b - - 90 250')
    assert is_insufficient_material(game.board)

def test_sufficient_material_with_king_knight_and_bishop():
    game = Game('6k1/8/6NB/8/8/8/8/2K5 b - - 90 250')
    assert not is_insufficient_material(game.board)

def test_sufficient_material_with_king_bishop_and_bishop():
    game = Game('6k1/8/6BB/8/8/8/8/2K5 b - - 90 250')
    assert not is_insufficient_material(game.board)