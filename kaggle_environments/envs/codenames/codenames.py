import json
import random
from os import path

def initialize_game(state, config):
    board_size = config.board_size
    starting_team_words = config.starting_team_words
    second_team_words = config.second_team_words
    
    # Load words
    dir_path = path.dirname(__file__)
    words_path = path.abspath(path.join(dir_path, "words.txt"))
    with open(words_path, "r") as f:
        all_words = [line.strip().upper() for line in f.readlines() if line.strip()]
        
    sampled_words = random.sample(all_words, board_size)
    
    # Determine playing order and word counts
    starting_team = random.choice(["red", "blue"])
    if starting_team == "red":
        red_count = starting_team_words
        blue_count = second_team_words
    else:
        red_count = second_team_words
        blue_count = starting_team_words
    
    # Assign roles
    roles = ["red"] * red_count + ["blue"] * blue_count + ["assassin"] * 1
    roles += ["neutral"] * (board_size - len(roles))
    random.shuffle(roles)
    
    revealed = [False] * board_size
    
    for agent_state in state:
        agent_state.observation.words = sampled_words
        agent_state.observation.roles = roles[:]
        agent_state.observation.revealed = revealed[:]
        agent_state.observation.current_turn = 0 if starting_team == "red" else 2
        agent_state.observation.clue = ""
        agent_state.observation.guesses_remaining = 0
        agent_state.observation.clue_number = 0

def update_visibility(state):
    # Mask roles for guessers (agents 1 and 3)
    roles = state[0].observation.roles
    revealed = state[0].observation.revealed
    
    for i in range(4):
        if i in [1, 3]:  # Guessers
            # Guessers only see roles of revealed cards
            masked_roles = [roles[j] if revealed[j] else "Unknown" for j in range(25)]
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
            if winner == "red":
                state[i].reward = 1 if i in [0, 1] else -1
            elif winner == "blue":
                state[i].reward = 1 if i in [2, 3] else -1
            else:
                state[i].reward = 0

    # Handle Agent Failure / Invalid Action
    if action is None:
        active_agent.status = "INVALID"
        end_game(winner="blue" if current_turn in [0, 1] else "red")
        return

    # SPYMASTER TURN
    if current_turn in [0, 2]:
        if not isinstance(action, dict) or "clue" not in action or "number" not in action:
            active_agent.status = "INVALID"
            end_game(winner="blue" if current_turn == 0 else "red")
            return
            
        # Clue validation
        normalized_clue = str(action["clue"]).strip().upper()
        words = state[0].observation.words
        revealed = state[0].observation.revealed
        roles = state[0].observation.roles
        opponent_team = "blue" if current_turn == 0 else "red"
        
        is_invalid_clue = False
        for i in range(25):
            if not revealed[i]:
                unrevealed_word = words[i].upper()
                if unrevealed_word in normalized_clue or normalized_clue in unrevealed_word:
                    is_invalid_clue = True
                    break
                    
        if is_invalid_clue:
            # Penalty: Reveal a random opponent word and pass turn
            opponent_unrevealed = [i for i in range(25) if not revealed[i] and roles[i] == opponent_team]
            if opponent_unrevealed:
                to_reveal = random.choice(opponent_unrevealed)
                for s in state:
                    s.observation.revealed[to_reveal] = True
            
            for s in state:
                s.observation.clue = ""
                s.observation.guesses_remaining = 0
                s.observation.current_turn = 2 if current_turn == 0 else 0
                
            # Check if penalty won the game for opponent
            red_left = sum(1 for i in range(25) if roles[i] == "red" and not state[0].observation.revealed[i])
            blue_left = sum(1 for i in range(25) if roles[i] == "blue" and not state[0].observation.revealed[i])
            
            if red_left == 0:
                end_game(winner="red")
            elif blue_left == 0:
                end_game(winner="blue")
            else:
                for i in range(4):
                    state[i].status = "ACTIVE" if i == state[0].observation.current_turn else "INACTIVE"
            return
            
        # Update state normally
        for s in state:
            clue_num = int(action["number"])
            s.observation.clue = str(action["clue"])
            s.observation.clue_number = clue_num
            s.observation.guesses_remaining = 25 if clue_num <= 0 else clue_num + 1
            s.observation.current_turn = 1 if current_turn == 0 else 3
            
        # Set agent statuses
        for i in range(4):
            state[i].status = "ACTIVE" if i == state[0].observation.current_turn else "INACTIVE"
            
    # GUESSER TURN
    elif current_turn in [1, 3]:
        # action is an int (0-24) or -1 (pass) OR a dict with "guess": int
        guess_val = action.get("guess") if isinstance(action, dict) else action
        
        if not isinstance(guess_val, int) or guess_val < -1 or guess_val > 24:
            active_agent.status = "INVALID"
            end_game(winner="blue" if current_turn == 1 else "red")
            return
            
        # Pass
        if guess_val == -1:
            clue_num = state[0].observation.clue_number
            expected_remaining = 25 if clue_num <= 0 else clue_num + 1
            # 0 ("zero") and -1 ("infinity") clues both give unlimited guesses but STILL require at least 1 guess
            if state[0].observation.guesses_remaining == expected_remaining:
                active_agent.status = "INVALID"
                end_game(winner="blue" if current_turn == 1 else "red")
                return
                
            for s in state:
                s.observation.clue = ""
                s.observation.guesses_remaining = 0
                s.observation.current_turn = 2 if current_turn == 1 else 0
        else:
            # Check if already revealed
            if state[0].observation.revealed[guess_val]:
                active_agent.status = "INVALID"
                end_game(winner="blue" if current_turn == 1 else "red")
                return
                
            # Reveal
            for s in state:
                s.observation.revealed[guess_val] = True
            
            roles = state[0].observation.roles
            guessed_role = roles[guess_val]
            team_color = "red" if current_turn == 1 else "blue"
            
            # Assassin check
            if guessed_role == "assassin":
                end_game(winner="blue" if team_color == "red" else "red")
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
        red_left = sum(1 for i in range(25) if roles[i] == "red" and not revealed[i])
        blue_left = sum(1 for i in range(25) if roles[i] == "blue" and not revealed[i])
        
        if red_left == 0:
            end_game(winner="red")
            return
        elif blue_left == 0:
            end_game(winner="blue")
            return

        # Next turn setup if not done
        if state[0].status != "DONE":
            for i in range(4):
                state[i].status = "ACTIVE" if i == state[0].observation.current_turn else "INACTIVE"

def interpreter(state, env):
    # Initialization
    if len(state[0].observation.get("words", [])) == 0:
         initialize_game(state, env.configuration)
         for i in range(4):
             state[i].status = "ACTIVE" if i == 0 else "INACTIVE"
         update_visibility(state)
         return state
             
    if env.done:
        return state

    process_action(state, env.configuration)
    update_visibility(state)
    
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
json_path = path.abspath(path.join(dir_path, "codenames.json"))
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
