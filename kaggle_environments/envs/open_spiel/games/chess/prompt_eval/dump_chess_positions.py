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
            # Determine how many top moves to consider based on move number
            move_num = len(moves) + 1
            if move_num <= 2:  # First 2 moves (1 per side)
                num_candidates = 4
            elif move_num <= 4:  # Next 2 moves (moves 3-4)
                num_candidates = 3
            elif move_num <= 6:  # Next 2 moves (moves 5-6)
                num_candidates = 2
            else:  # All remaining moves
                num_candidates = 1
            
            # Analyze to get multiple moves, then pick one
            info = eng.analyse(board, chess.engine.Limit(time=time_limit), multipv=num_candidates)
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

def filter_balanced_positions(input_file: str, output_file: str, 
                            engine_path: str, max_centipawn_diff: int = 300) -> None:
    """Filter positions from input_file and save balanced ones to output_file."""
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            processed = 0
            saved = 0
            
            for line in infile:
                position_data = json.loads(line.strip())
                fen = position_data["fen"]
                
                # Set up board from FEN
                board = chess.Board(fen)
                
                # Analyze position with 0.5s time limit
                try:
                    info = eng.analyse(board, chess.engine.Limit(time=0.5))
                    score = info.get("score")
                    
                    if score and score.relative:
                        # Convert to centipawns
                        if score.relative.is_mate():
                            # Skip mate positions as they're too decisive
                            pass
                        else:
                            cp_score = abs(score.relative.score())
                            if cp_score <= max_centipawn_diff:
                                outfile.write(line)
                                outfile.flush()  # Ensure continuous writing
                                saved += 1
                                print(f"Saved position {saved} (cp: {cp_score})", file=sys.stderr)
                    
                    processed += 1
                    if processed % 50 == 0:
                        print(f"Processed {processed} positions, saved {saved}", file=sys.stderr)
                        
                except Exception as e:
                    print(f"Error analyzing position {processed}: {e}", file=sys.stderr)
                    
    finally:
        eng.quit()
    
    print(f"Filtering complete: {saved}/{processed} positions saved to {output_file}")

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Dump random Chess positions")
    p.add_argument("-n", "--num", type=int,
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
    p.add_argument("--filter", action="store_true",
                   help="filter existing positions for balance instead of generating new ones")
    p.add_argument("--input", default=str(Path(__file__).parent / "chess_positions.jsonl"),
                   help="input file for filtering (JSONL format)")
    p.add_argument("--filtered-out", default=str(Path(__file__).parent / "filtered_chess_positions.jsonl"),
                   help="output file for filtered positions")
    p.add_argument("--max-cp-diff", type=int, default=300,
                   help="maximum centipawn difference for balanced positions")
    args = p.parse_args(argv)

    if args.filter:
        # Filter existing positions
        if not Path(args.input).exists():
            p.error(f"Input file {args.input} does not exist")
        
        filter_balanced_positions(args.input, args.filtered_out, 
                                args.stockfish, args.max_cp_diff)
    else:
        # Generate new positions
        if not args.num:
            p.error("--num is required when not filtering")
            
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
