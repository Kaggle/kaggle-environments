def initialize_memory(observation, board_size):
    """Initializes memory fields if not present."""
    if "history" not in observation:
        observation.history = []
        observation.current_game = 0
        observation.blue_wins = 0
        observation.yellow_wins = 0
        observation.current_game_turns = []


def _unwrap_submission(action):
    """core_harness wraps the real action inside {"submission": ...}. Peel it."""
    if isinstance(action, dict) and "submission" in action:
        return action["submission"]
    return action


def track_turn(observation, state, acting_turn, action, pre_revealed):
    """Record what just happened as a structured entry in current_game_turns.

    Called AFTER process_action + update_visibility. Each recorded entry
    describes what the acting agent did (clue attempt, guess, or pass) —
    not merely which cells got revealed as a side effect. This distinction
    matters for the invalid-clue penalty path: the penalty reveal is not a
    guess and must not be attributed to the previous team's clue.
    """
    obs = observation
    team = "blue" if acting_turn in (0, 1) else "yellow"

    action = _unwrap_submission(action)

    newly_revealed = [
        i for i in range(len(obs.revealed))
        if obs.revealed[i] and not pre_revealed[i]
    ]
    words = obs.words
    # state[0] is the Blue Cluemaster; their observation carries full,
    # unmasked roles, which is what we want to record here (any newly
    # revealed cell's role is public info the model can already see on
    # the board).
    full_roles = state[0].observation.roles

    # --- CLUEMASTER ACTION ------------------------------------------------
    if acting_turn in (0, 2):
        # Malformed cluemaster actions (None, non-dict, missing keys) end
        # the game via INVALID; nothing meaningful to record for memory.
        if not isinstance(action, dict) or "clue" not in action:
            return

        # Valid clue: obs.clue is the accepted, non-empty clue string.
        if obs.clue != "":
            obs.current_game_turns.append({
                "team": team,
                "clue": obs.clue,
                "number": obs.clue_number,
                "guesses": [],
                "results": [],
            })
            return

        # Invalid clue: the interpreter reset obs.clue to "" and (usually)
        # revealed one opponent word as a penalty.
        entry = {
            "team": team,
            "clue": str(action.get("clue")),
            "number": action.get("number"),
            "invalid_clue": True,
        }
        if newly_revealed:
            i = newly_revealed[0]
            entry["penalty_revealed_word"] = words[i]
            entry["penalty_revealed_role"] = full_roles[i]
        obs.current_game_turns.append(entry)
        return

    # --- GUESSER ACTION ---------------------------------------------------
    guess_val = action.get("guess") if isinstance(action, dict) else action

    # A pass with prior correct guesses is legal and ends the turn; a pass
    # before any guess for the current clue is INVALID and ends the game.
    # Distinguish the two so the memory doesn't teach models that "empty
    # pass" is a fine option.
    if guess_val == -1:
        if obs.current_game_turns and "guesses" in obs.current_game_turns[-1]:
            entry = obs.current_game_turns[-1]
            if entry["guesses"]:
                entry["passed"] = True
            else:
                entry["invalid_pass"] = True
        return

    # Otherwise it should be an int guess. Only append when a cell was
    # actually revealed by this action (skips the INVALID cases: bad type,
    # out-of-range, or already-revealed index).
    if not isinstance(guess_val, int) or not newly_revealed:
        return

    if obs.current_game_turns and "guesses" in obs.current_game_turns[-1]:
        last_turn = obs.current_game_turns[-1]
        for i in newly_revealed:
            last_turn["guesses"].append(words[i])
            last_turn["results"].append(full_roles[i])


def save_game_to_history(observation, winner, window_size):
    """
    Summarizes and categorizes the game, then appends to history.

    Example of a stored game in history:
    {
      "game": 0,
      "winner": "yellow",
      "yellow_team_moves": [
        {"clue": "FRUIT", "number": 2, "guesses": ["APPLE", "BANANA"], "results": ["yellow", "yellow"]}
      ],
      "blue_team_moves": [
        {"clue": "OCEAN", "number": 1, "guesses": ["SHIP"], "results": ["neutral"]},
        {"clue": "hot-dog", "number": 2, "invalid_clue": true,
         "penalty_revealed_word": "APPLE", "penalty_revealed_role": "yellow"}
      ]
    }
    """
    obs = observation

    # Separate turns by team
    yellow_moves = [t for t in obs.current_game_turns if t["team"] == "yellow"]
    blue_moves = [t for t in obs.current_game_turns if t["team"] == "blue"]

    # Remove the "team" key from the inner dictionaries to save space
    for t in yellow_moves: del t["team"]
    for t in blue_moves: del t["team"]

    # Append the categorized game log
    obs.history.append({
        "game": obs.current_game,
        "winner": winner,
        "yellow_team_moves": yellow_moves,
        "blue_team_moves": blue_moves
    })

    # Enforce sliding window if configured
    if window_size > 0:
        obs.history = obs.history[-window_size:]
