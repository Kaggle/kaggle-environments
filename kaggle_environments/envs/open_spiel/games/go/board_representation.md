# Go Game State JSON Structure

The game state is serialized into a JSON object before being passed to the AI model. This JSON object provides all the necessary information for the AI to understand the current board configuration and whose turn it is.  Optimized to be clear to the AI for prompting, rather than for minimal / cleanest structure.

## Example JSON Structure

Below is an example of the JSON structure:

    {
      "board_size": 9,
      "komi": 7.5,
      "current_player_to_move": "B",
      "move_number": 1,
      "previous_move_a1": null,
      "board_grid": [
        [ {"A9": "."}, {"B9": "."}, /* ..., */ {"J9": "."} ],
        [ {"A8": "."}, {"B8": "."}, /* ..., */ {"J8": "."} ],
        /* ... more rows ... */
        [ {"A1": "."}, {"B1": "."}, /* ..., */ {"J1": "."} ]
      ]
    }

## Top-Level Fields

* **board_size** (*Integer*)
    * Description: The dimension of the square Go board. For example, a 9x9 board will have **board_size** set to **9**.
    * Example: **9**

* **komi** (*Float*)
    * Description: The compensation points given to White for playing second.
    * Example: **7.5**

* **current_player_to_move** (*String*)
    * Description: Indicates whose turn it is to play.
    * Values:
        * **"B"**: Black to move.
        * **"W"**: White to move.
    * Example: **"B"**

* **move_number** (*Integer*)
    * Description: The current move number in the game. The first move is **1**.
    * Example: **1**

* **previous_move_a1** (*String | Null*)
    * Description: The A1 notation of the last move made. It is **null** if no moves have been made yet (e.g., at the start of the game).
    * Example (after Black plays C3): **"C3"**
    * Example (start of game): **null**

* **board_grid** (*List of Lists of Dictionaries*)
    * Description: Represents the Go board itself. It is a list of rows, where each row is a list of intersection states.
    * Structure:
        * The outer list contains a number of elements equal to **board_size**, each representing a row on the board.
        * Rows are ordered from top to bottom in A1 notation (e.g., for a 9x9 board, the first element of **board_grid** is row 9, the second is row 8, ..., and the last element is row 1).
        * Each inner list (representing a row) contains a number of dictionaries equal to **board_size**.
        * Each dictionary represents a single intersection on the board.
        * Each dictionary has exactly one key-value pair:
            * **Key** (*String*): The A1 coordinate of the intersection (e.g., **"A9"**, **"D4"**, **"J1"**). The column letters are A, B, C, D, E, F, G, H, J (omitting I). Rows are numbered from 1 (bottom) to **board_size** (top).
            * **Value** (*String*): The state of the intersection.
                * **"."**: Empty intersection.
                * **"B"**: Black stone.
                * **"W"**: White stone.
    * Example (for a 9x9 board, the first element of the first row, i.e., top-left corner A9, is empty):
        The **board_grid** would start like this:

            [ // board_grid
              [ // Row 9 (first element of board_grid)
                {"A9": "."}, {"B9": "."}, /* ... */, {"J9": "."}
              ],
              // ... other rows ...
            ]
    * Example (a single intersection dictionary for D4 with a Black stone): **{"D4": "B"}**