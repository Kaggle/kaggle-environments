# Chess Board JSON Format for LLM Evaluation

## Core Format Structure

```json
{
  "board": [
    [{"a8": "black_rook"}, {"b8": "black_knight"}, {"c8": "black_bishop"}, {"d8": "black_queen"}, {"e8": "black_king"}, {"f8": "black_bishop"}, {"g8": "black_knight"}, {"h8": "black_rook"}],
    [{"a7": "black_pawn"}, {"b7": "black_pawn"}, {"c7": "black_pawn"}, {"d7": "black_pawn"}, {"e7": "black_pawn"}, {"f7": "black_pawn"}, {"g7": "black_pawn"}, {"h7": "black_pawn"}],
    [{"a6": "empty"}, {"b6": "empty"}, {"c6": "empty"}, {"d6": "empty"}, {"e6": "empty"}, {"f6": "empty"}, {"g6": "empty"}, {"h6": "empty"}],
    [{"a5": "empty"}, {"b5": "empty"}, {"c5": "empty"}, {"d5": "empty"}, {"e5": "empty"}, {"f5": "empty"}, {"g5": "empty"}, {"h5": "empty"}],
    [{"a4": "empty"}, {"b4": "empty"}, {"c4": "empty"}, {"d4": "empty"}, {"e4": "white_pawn"}, {"f4": "empty"}, {"g4": "empty"}, {"h4": "empty"}],
    [{"a3": "empty"}, {"b3": "empty"}, {"c3": "empty"}, {"d3": "empty"}, {"e3": "empty"}, {"f3": "empty"}, {"g3": "empty"}, {"h3": "empty"}],
    [{"a2": "white_pawn"}, {"b2": "white_pawn"}, {"c2": "white_pawn"}, {"d2": "white_pawn"}, {"e2": "empty"}, {"f2": "white_pawn"}, {"g2": "white_pawn"}, {"h2": "white_pawn"}],
    [{"a1": "white_rook"}, {"b1": "white_knight"}, {"c1": "white_bishop"}, {"d1": "white_queen"}, {"e1": "white_king"}, {"f1": "white_bishop"}, {"g1": "white_knight"}, {"h1": "white_rook"}]
  ],
  "game_state": {
    "active_player": "black",
    "move_number": 1,
    "castling_rights": {
      "white_kingside": true,
      "white_queenside": true,
      "black_kingside": true,
      "black_queenside": true
    },
    "en_passant_target": "e3"
  },
  "position_analysis": {
    "material_balance": {
      "white": 39,
      "black": 39
    },
    "king_safety": {
      "white_king_position": "e1",
      "black_king_position": "e8",
      "castled": {
        "white": false,
        "black": false
      }
    }
  },
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
}
```

## Board Representation Details

### Board Array Structure
- **8x8 array**: `board[rank][file]` where rank 0 = 8th rank, rank 7 = 1st rank
- **Square dictionaries**: Each position is `{"square": "piece"}` or `{"square": "empty"}`
- **File order**: a-h (left to right in each rank array)
- **Rank order**: 8-1 (top to bottom in main array)
- **Empty squares**: `{"square": "empty"}` for explicit square identification

### Piece Naming Convention
- Format: `{color}_{piece_type}`
- Colors: `white`, `black`
- Piece types: `king`, `queen`, `rook`, `bishop`, `knight`, `pawn`

### Square Identification
- Every square explicitly labeled (e.g., `{"e4": "white_pawn"}`)
- Eliminates translation errors between array indices and chess notation
- LLM can immediately see both piece and location

## Game State Information

### Player Turn
- `active_player`: `"white"` or `"black"`
- Clear indication of whose move it is

### Move Counting
- `move_number`: Full move number (increments after Black's move)

### Castling Rights
- Separate boolean flags for each castling possibility
- More explicit than FEN's KQkq notation

### En Passant
- Target square in algebraic notation (e.g., "e3")
- `null` if no en passant possible

### FEN Reference
- Top-level `fen` field provides standard notation
- Useful for compatibility and verification

## Position Analysis

### Material Count
- Numerical values using standard piece values
- Quick material balance assessment

### King Safety
- King positions and castling status
- Basic safety indicators