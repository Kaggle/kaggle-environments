import json
import random
from os import path
from .memory import initialize_memory, track_turn, save_game_to_history

BOARD_SIZE = 25

def initialize_game(state, config):
    board_size = config.board_size
    starting_team_words = config.starting_team_words
    second_team_words = config.second_team_words
    
    # Load words
    dir_path = path.dirname(__file__)
    words_path = path.abspath(path.join(dir_path, "words.txt"))
    with open(words_path, "r") as f:
        all_words = [line.strip().upper() for line in f.readlines() if line.strip()]
        
    # Setup deterministic random generator if seed is provided
    seed = config.get("seed")
    rng = random.Random(seed) if seed is not None else random
        
    sampled_words = rng.sample(all_words, board_size)
    
    # Determine playing order and word counts
    starting_team = rng.choice(["blue", "yellow"])
    if starting_team == "blue":
        blue_count = starting_team_words
        yellow_count = second_team_words
    else:
        blue_count = second_team_words
        yellow_count = starting_team_words
    
    # Assign roles
    roles = ["blue"] * blue_count + ["yellow"] * yellow_count + ["assassin"] * 1
    roles += ["neutral"] * (board_size - len(roles))
    rng.shuffle(roles)
    
    revealed = [False] * board_size
    
    for agent_state in state:
        agent_state.observation.words = sampled_words
        agent_state.observation.roles = roles[:]
        agent_state.observation.revealed = revealed[:]
        agent_state.observation.current_turn = 0 if starting_team == "blue" else 2
        agent_state.observation.clue = ""
        agent_state.observation.guesses_remaining = 0
        agent_state.observation.clue_number = 0
        
        initialize_memory(agent_state.observation, board_size)

def update_visibility(state):
    # Mask roles for guessers (agents 1 and 3)
    roles = state[0].observation.roles
    revealed = state[0].observation.revealed
    
    for i in range(4):
        if i in [1, 3]:  # Guessers
            # Guessers only see roles of revealed cards
            masked_roles = [roles[j] if revealed[j] else "Unknown" for j in range(BOARD_SIZE)]
            state[i].observation.roles = masked_roles
        else:
            state[i].observation.roles = roles[:]

def process_action(state, config):
    current_turn = state[0].observation.current_turn
    active_agent = state[current_turn]
    action = active_agent.action
    
    # helper to end game
    def end_game(winner=None):
        for i in range(4):
            if state[i].status != "INVALID":
                state[i].status = "DONE"
            if winner == "blue":
                state[i].reward = 1 if i in [0, 1] else -1
            elif winner == "yellow":
                state[i].reward = 1 if i in [2, 3] else -1
            else:
                state[i].reward = 0

    # Handle Agent Failure / Invalid Action
    if action is None:
        active_agent.status = "INVALID"
        end_game(winner="yellow" if current_turn in [0, 1] else "blue")
        return

    # SPYMASTER TURN
    if current_turn in [0, 2]:
        if not isinstance(action, dict) or "clue" not in action or "number" not in action:
            active_agent.status = "INVALID"
            end_game(winner="yellow" if current_turn == 0 else "blue")
            return
            
        # Clue validation
        normalized_clue = str(action["clue"]).strip().upper()
        words = state[0].observation.words
        revealed = state[0].observation.revealed
        roles = state[0].observation.roles
        opponent_team = "yellow" if current_turn == 0 else "blue"
        
        is_invalid_clue = False
        if " " in normalized_clue or "-" in normalized_clue:
            is_invalid_clue = True
            
        if not is_invalid_clue:
            for i in range(BOARD_SIZE):
                if not revealed[i]:
                    unrevealed_word = words[i].upper()
                    if unrevealed_word in normalized_clue or normalized_clue in unrevealed_word:
                        is_invalid_clue = True
                        break
                    
        if is_invalid_clue:
            # Penalty: Reveal a random opponent word and pass turn
            opponent_unrevealed = [i for i in range(BOARD_SIZE) if not revealed[i] and roles[i] == opponent_team]
            if opponent_unrevealed:
                to_reveal = random.choice(opponent_unrevealed)
                for s in state:
                    s.observation.revealed[to_reveal] = True
            
            for s in state:
                s.observation.clue = ""
                s.observation.guesses_remaining = 0
                s.observation.current_turn = 2 if current_turn == 0 else 0
                
            # Check if penalty won the game for opponent
            blue_left = sum(1 for i in range(BOARD_SIZE) if roles[i] == "blue" and not state[0].observation.revealed[i])
            yellow_left = sum(1 for i in range(BOARD_SIZE) if roles[i] == "yellow" and not state[0].observation.revealed[i])
            
            if blue_left == 0:
                end_game(winner="blue")
            elif yellow_left == 0:
                end_game(winner="yellow")
            else:
                for i in range(4):
                    state[i].status = "ACTIVE" if i == state[0].observation.current_turn else "INACTIVE"
            return
            
        # Update state normally
        for s in state:
            clue_num = int(action["number"])
            s.observation.clue = str(action["clue"])
            s.observation.clue_number = clue_num
            s.observation.guesses_remaining = BOARD_SIZE if clue_num <= 0 else clue_num + 1
            s.observation.current_turn = 1 if current_turn == 0 else 3
            
        # Set agent statuses
        for i in range(4):
            state[i].status = "ACTIVE" if i == state[0].observation.current_turn else "INACTIVE"
            
    # GUESSER TURN
    elif current_turn in [1, 3]:
        # action is an int (0-24) or -1 (pass) OR a dict with "guess": int
        guess_val = action.get("guess") if isinstance(action, dict) else action
        
        if not isinstance(guess_val, int) or guess_val < -1 or guess_val > BOARD_SIZE - 1:
            active_agent.status = "INVALID"
            end_game(winner="yellow" if current_turn == 1 else "blue")
            return
            
        # Pass
        if guess_val == -1:
            clue_num = state[0].observation.clue_number
            expected_remaining = BOARD_SIZE if clue_num <= 0 else clue_num + 1
            # 0 ("zero") and -1 ("infinity") clues both give unlimited guesses but STILL require at least 1 guess
            if state[0].observation.guesses_remaining == expected_remaining:
                active_agent.status = "INVALID"
                end_game(winner="yellow" if current_turn == 1 else "blue")
                return
                
            for s in state:
                s.observation.clue = ""
                s.observation.guesses_remaining = 0
                s.observation.current_turn = 2 if current_turn == 1 else 0
        else:
            # Check if already revealed
            if state[0].observation.revealed[guess_val]:
                active_agent.status = "INVALID"
                end_game(winner="yellow" if current_turn == 1 else "blue")
                return
                
            # Reveal
            for s in state:
                s.observation.revealed[guess_val] = True
            
            roles = state[0].observation.roles
            guessed_role = roles[guess_val]
            team_color = "blue" if current_turn == 1 else "yellow"
            
            # Assassin check
            if guessed_role == "assassin":
                end_game(winner="yellow" if team_color == "blue" else "blue")
                return
                
            # Neutral or Opponent word
            if guessed_role != team_color:
                for s in state:
                    s.observation.clue = ""
                    s.observation.guesses_remaining = 0
                    s.observation.current_turn = 2 if current_turn == 1 else 0
            else:
                # Correct guess
                for s in state:
                    s.observation.guesses_remaining -= 1
                    
                if state[0].observation.guesses_remaining <= 0:
                    for s in state:
                        s.observation.clue = ""
                        s.observation.guesses_remaining = 0
                        s.observation.current_turn = 2 if current_turn == 1 else 0

        # Win condition check
        revealed = state[0].observation.revealed
        roles = state[0].observation.roles
        blue_left = sum(1 for i in range(BOARD_SIZE) if roles[i] == "blue" and not revealed[i])
        yellow_left = sum(1 for i in range(BOARD_SIZE) if roles[i] == "yellow" and not revealed[i])
        
        if blue_left == 0:
            end_game(winner="blue")
            return
        elif yellow_left == 0:
            end_game(winner="yellow")
            return

        # Next turn setup if not done
        if state[0].status != "DONE":
            for i in range(4):
                state[i].status = "ACTIVE" if i == state[0].observation.current_turn else "INACTIVE"

def interpreter(state, env):
    # Initialization
    if len(state[0].observation.get("words", [])) == 0:
         initialize_game(state, env.configuration)
         active_player = state[0].observation.current_turn
         for i in range(4):
             state[i].status = "ACTIVE" if i == active_player else "INACTIVE"
         update_visibility(state)
         return state
             
    if env.done:
        return state

    process_action(state, env.configuration)
    update_visibility(state)
    
    # Custom Memory Logic
    obs = state[0].observation
    games_per_episode = env.configuration.get("games_per_episode", 1)
    
    # Always track turns within the current game for all agents
    for s in state:
        track_turn(s.observation, state)
    
    if games_per_episode > 1:
        is_done = all(s.status in ["DONE", "INVALID"] for s in state)
        if is_done:
            winner = None
            if state[0].reward == 1: winner = "blue"
            elif state[2].reward == 1: winner = "yellow"
            
            window_size = env.configuration.get("memory_window_size", 0)
            save_game_to_history(obs, winner, window_size)
            
            if obs.current_game + 1 < games_per_episode:
                # Continue to next game
                obs.current_game += 1
                obs.current_game_turns = []
                obs._last_clue = ""
                obs._last_revealed = [False] * len(obs.revealed)
                
                # Reset board (re-init)
                initialize_game(state, env.configuration)
                
                # Reset agent statuses based on new current_turn
                active_agent = state[0].observation.current_turn
                for i in range(4):
                    state[i].status = "ACTIVE" if i == state[0].observation.current_turn else "INACTIVE"
                    state[i].reward = 0
                    
    return state


def renderer(state, env):
    words = state[0].observation.words
    revealed = state[0].observation.revealed
    roles = state[0].observation.roles
    
    out = ""
    for r in range(5):
        row_str = ""
        for c in range(5):
            idx = r * 5 + c
            w = words[idx]
            if revealed[idx]:
                w = f"[{roles[idx].upper()[0]}] {w}"
            else:
                w = f"({roles[idx].upper()[0]}) {w}"
            row_str += f"{w:<15}"
        out += row_str + "\n"
    
    out += f"\nTurn: {state[0].observation.current_turn}\n"
    out += f"Clue: {state[0].observation.clue} ({state[0].observation.guesses_remaining} remaining)\n"
    return out


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "word_association.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    """Reads the built web visualizer output and serves it for rendering."""
    jspath = path.join(dir_path, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""

from .agents import random_agent
agents = {"random": random_agent}
