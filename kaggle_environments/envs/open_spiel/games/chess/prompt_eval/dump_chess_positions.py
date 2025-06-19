#!/usr/bin/env python
"""
dump_chess_positions.py
-----------------------

Generate a corpus of *partial* Chess positions using Stockfish.

Each output line is a FEN string representing a chess position.

Usage examples
--------------
# 1 000 positions to default path with Stockfish in $PATH
python dump_chess_positions.py --n 1000

# 20 000 positions, depth-2 search, custom Stockfish binary, write to .txt
python dump_chess_positions.py -n 20000 --depth 2 \
    --stockfish ~/engines/stockfish15/stockfish \
    --out data/chess_positions.txt
"""
import argparse, random, sys
from pathlib import Path

import chess, chess.engine              # python-chess

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def stockfish_selfplay(board: chess.Board, engine_path: str,
                       depth: int, max_moves: int) -> list[chess.Move]:
    """Play a full game (until mate/stalemate or max_moves)."""
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    moves: list[chess.Move] = []
    try:
        while not board.is_game_over() and len(moves) < max_moves:
            res = eng.play(board, chess.engine.Limit(depth=depth))
            board.push(res.move)
            moves.append(res.move)
    finally:
        eng.quit()
    return moves

def sample_partial_position(engine_path: str, depth: int,
                           max_moves: int,
                           frac_range=(0.05, 0.95)) -> str:
    """Return a random cut-off of a Stockfish self-play game as FEN."""
    board = chess.Board()
    full = stockfish_selfplay(board, engine_path, depth, max_moves)
    cut = int(len(full) * random.uniform(*frac_range))
    partial_moves = full[:max(1, cut)]            # at least one move

    # Replay moves to get the position
    board_copy = chess.Board()
    for m in partial_moves:
        board_copy.push(m)
    
    return board_copy.fen()

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Dump random Chess positions")
    p.add_argument("-n", "--num", type=int, required=True,
                   help="number of positions to generate")
    p.add_argument("-o", "--out", default=str(Path(__file__).parent / "chess_positions.txt"),
                   help="output file (one line per position)")
    p.add_argument("--stockfish", default="stockfish",
                   help="path to Stockfish binary (must speak UCI)")
    p.add_argument("--depth", type=int, default=1,
                   help="search depth per move (Stockfish)")
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
            fen = sample_partial_position(args.stockfish, args.depth,
                                         args.max_moves, rng)
            f.write(fen + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{args.num}...", file=sys.stderr)

    print(f"Wrote {args.num} positions ➜ {out_path}")

if __name__ == "__main__":
    main()
