def initialize_memory(observation, board_size):
    """Initializes memory fields if not present."""
    if "history" not in observation:
        observation.history = []
        observation.current_game = 0
        observation.current_game_turns = []
        observation._last_clue = ""
        observation._last_revealed = [False] * board_size

def track_turn(observation, state):
    """Tracks clues and guesses during the game."""
    obs = observation
    
    # Detect new clue
    if obs.clue != obs._last_clue and obs.clue != "":
        # current_turn is updated by prod_interpreter to the NEXT player (guesser)
        team = "red" if obs.current_turn == 1 else "blue"
        obs.current_game_turns.append({
            "team": team,
            "clue": obs.clue,
            "num": obs.clue_number,
            "guesses": [],
            "results": []
        })
        obs._last_clue = obs.clue
        
    # Detect new guesses
    revealed = obs.revealed
    words = obs.words
    
    for i in range(len(revealed)):
        if revealed[i] and not obs._last_revealed[i]:
            if obs.current_game_turns:
                last_turn = obs.current_game_turns[-1]
                last_turn["guesses"].append(words[i])
                # Read full roles from agent 0 (Spymaster)
                full_roles = state[0].observation.roles
                last_turn["results"].append(full_roles[i])
            obs._last_revealed[i] = True

def save_game_to_history(observation, winner, window_size):
    """
    Summarizes and categorizes the game, then appends to history.
    
    Example of a stored game in history:
    {
      "game": 0,
      "winner": "red",
      "red_team_moves": [
        {"clue": "FRUIT", "num": 2, "guesses": ["APPLE", "BANANA"], "results": ["red", "red"]}
      ],
      "blue_team_moves": [
        {"clue": "OCEAN", "num": 1, "guesses": ["SHIP"], "results": ["neutral"]}
      ]
    }
    """
    obs = observation
    
    # Separate turns by team
    red_moves = [t for t in obs.current_game_turns if t["team"] == "red"]
    blue_moves = [t for t in obs.current_game_turns if t["team"] == "blue"]
    
    # Remove the "team" key from the inner dictionaries to save space
    for t in red_moves: del t["team"]
    for t in blue_moves: del t["team"]
    
    # Append the categorized game log
    obs.history.append({
        "game": obs.current_game,
        "winner": winner,
        "red_team_moves": red_moves,
        "blue_team_moves": blue_moves
    })
    
    # Enforce sliding window if configured
    if window_size > 0:
        obs.history = obs.history[-window_size:]
