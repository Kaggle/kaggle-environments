"""Test agent for use with main.py http-server tests."""

import random


def agent(observation, configuration):
    """Simple test agent that picks a random legal move."""
    # Use legalActions if available (preferred - comes from action mask)
    legal_actions = observation.get("legalActions", [])
    if legal_actions:
        return random.choice(legal_actions)

    # Fallback: check board for empty cells
    board = observation.get("board", [])
    if board:
        empty_cells = [i for i, cell in enumerate(board) if cell == 0]
        if empty_cells:
            return random.choice(empty_cells)

    return 0
