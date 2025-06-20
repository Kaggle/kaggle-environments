#!/usr/bin/env python
"""
dump_chess_positions.py
-----------------------

Generate a corpus of *partial* Chess positions using Stockfish.

Each output line is a JSON object with FEN string and PGN of the game.

Usage examples
--------------
# 1 000 positions to default path with Stockfish in $PATH
python dump_chess_positions.py --n 1000

# 20 000 positions, 50ms per move, custom Stockfish binary, write to .jsonl
python dump_chess_positions.py -n 20000 --time 0.05 \
    --stockfish ~/engines/stockfish15/stockfish \
    --out data/chess_positions.jsonl
"""
import argparse, random, sys, json
from pathlib import Path

import chess, chess.engine, chess.pgn   # python-chess

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def stockfish_selfplay(board: chess.Board, engine_path: str,
                       time_limit: float, max_moves: int) -> list[chess.Move]:
    """Play a full game (until mate/stalemate or max_moves)."""
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    moves: list[chess.Move] = []
    try:
        while not board.is_game_over() and len(moves) < max_moves:
            # Analyze to get multiple moves, then pick one
            info = eng.analyse(board, chess.engine.Limit(time=time_limit), multipv=8)
            if not info:  # Should not happen if game is not over
                break
            # Get the move from each PV line and pick one randomly
            top_moves = [item["pv"][0] for item in info]
            chosen_move = random.choice(top_moves)
            
            board.push(chosen_move)
            moves.append(chosen_move)
    finally:
        eng.quit()
    return moves

def sample_partial_position(engine_path: str, time_limit: float,
                           max_moves: int,
                           frac_range=(0.05, 0.95)) -> dict:
    """Return a random cut-off of a Stockfish self-play game as FEN and PGN."""
    board = chess.Board()
    full = stockfish_selfplay(board, engine_path, time_limit, max_moves)
    cut = int(len(full) * random.uniform(*frac_range))
    partial_moves = full[:max(1, cut)]            # at least one move

    # Replay moves to get the position
    board_copy = chess.Board()
    for m in partial_moves:
        board_copy.push(m)
    
    # Generate move sequence string
    game = chess.pgn.Game()
    node = game
    for move in partial_moves:
        node = node.add_variation(move)
    
    # Get just the moves without headers
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    moves_only = game.accept(exporter).strip()
    
    return {
        "fen": board_copy.fen(),
        "pgn": moves_only
    }

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Dump random Chess positions")
    p.add_argument("-n", "--num", type=int, required=True,
                   help="number of positions to generate")
    p.add_argument("-o", "--out", default=str(Path(__file__).parent / "chess_positions.jsonl"),
                   help="output file (one line per position in JSONL format)")
    p.add_argument("--stockfish", default="stockfish",
                   help="path to Stockfish binary (must speak UCI)")
    p.add_argument("--time", type=float, default=0.02,
                   help="time limit per move in seconds (Stockfish)")
    p.add_argument("--max-moves", type=int, default=120,
                   help="stop a self-play game after this many plies")
    p.add_argument("--min-frac", type=float, default=0.05,
                   help="earliest cut-off fraction of game (0-1)")
    p.add_argument("--max-frac", type=float, default=0.95,
                   help="latest cut-off fraction of game (0-1)")
    args = p.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = (args.min_frac, args.max_frac)
    if not 0 < rng[0] < rng[1] <= 1:
        p.error("--min-frac and --max-frac must satisfy 0 < min < max ≤ 1")

    with out_path.open("w") as f:
        for i in range(args.num):
            position_data = sample_partial_position(args.stockfish, args.time,
                                                   args.max_moves, rng)
            f.write(json.dumps(position_data) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{args.num}...", file=sys.stderr)

    print(f"Wrote {args.num} positions ➜ {out_path}")

if __name__ == "__main__":
    main()
