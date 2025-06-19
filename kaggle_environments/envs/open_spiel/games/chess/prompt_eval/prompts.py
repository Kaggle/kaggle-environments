"""
Chess prompt generation for LLM evaluation.

Simple API for generating chess prompts with variable substitution.
"""


def generate_fen_prompt(fen_state: str, move_history: str, color: str) -> str:
    """
    Generate a chess prompt for the given position.
    
    Args:
        state_str: FEN string representing the current position
        move_history: String of moves played so far  
        color: Player color ("white" or "black")
    
    Returns:
        Formatted prompt string
    """
    return f"""Let's play chess. The current game state in FEN is:
{fen_state}
The moves played so far are:
{move_history}
You are playing as player {color}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in Algebraic Notation."""


