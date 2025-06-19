"""
Chess move response parser for LLM evaluation.

Robust parser for extracting chess moves in Algebraic Notation from LLM responses.
Handles various response formats and provides clear success/failure indication.
"""

import re
from typing import Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class ParseResult:
    """Result of parsing an LLM response for a chess move."""
    success: bool
    move: Optional[str] = None
    error_message: Optional[str] = None


class ChessResponseParser:
    """Parser for extracting chess moves from LLM responses."""
    
    def __init__(self):
        # Common final answer patterns (case insensitive)
        # More permissive move pattern to catch various formats
        move_pattern = r'([a-h][1-8][a-h][1-8][qrbnQRBN]?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[Oo0]-[Oo0](?:-[Oo0])?)'
        
        self.final_answer_patterns = [
            rf'\*\*\s*final\s+answer\s*:\s*\*\*\s*\*?{move_pattern}\*?',  # Handle markdown around move
            rf'final\s+answer\s*:\s*\*?{move_pattern}\*?',
            rf'answer\s*:\s*\*?{move_pattern}\*?',
            rf'move\s*:\s*\*?{move_pattern}\*?',
        ]
        
        # General algebraic notation pattern (more permissive for standalone moves)
        self.algebraic_pattern = r'\b([a-h][1-8][a-h][1-8][qrbnQRBN]?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[Oo0]-[Oo0](?:-[Oo0])?)\b'
        
        # Compile patterns for efficiency
        self.compiled_final_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.final_answer_patterns]
        self.compiled_algebraic = re.compile(self.algebraic_pattern, re.IGNORECASE)
    
    def _is_valid_algebraic_notation(self, move: str) -> bool:
        """Check if a string looks like valid algebraic notation."""
        if not move:
            return False
            
        move = move.strip()
        
        # Castling (handle various formats)
        if move.lower() in ['o-o', 'o-o-o', '0-0', '0-0-0']:
            return True
        
        # Standard moves (piece + destination, captures, promotions, etc.)
        # This is a basic check - actual legality would need board validation
        basic_pattern = r'^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?$'
        uci_pattern = r'^[a-h][1-8][a-h][1-8][qrbnQRBN]?$'
        
        # Special check for invalid same-square UCI moves
        if re.match(uci_pattern, move, re.IGNORECASE):
            if len(move) >= 4 and move[:2] == move[2:4]:
                return False  # Same square move is invalid
        
        return bool(re.match(basic_pattern, move, re.IGNORECASE) or 
                   re.match(uci_pattern, move, re.IGNORECASE))
    
    def _extract_from_final_answer(self, response: str) -> Optional[str]:
        """Try to extract move from explicit final answer patterns."""
        for pattern in self.compiled_final_patterns:
            match = pattern.search(response)
            if match:
                move = match.group(1).strip()
                if self._is_valid_algebraic_notation(move):
                    return move
        return None
    
    def _extract_last_valid_move(self, response: str) -> Optional[str]:
        """Extract the last valid-looking move from the response."""
        matches = self.compiled_algebraic.findall(response)
        
        # Check matches from end to beginning
        for move in reversed(matches):
            move = move.strip()
            if self._is_valid_algebraic_notation(move):
                return move
        
        return None
    
    def parse(self, response: str) -> ParseResult:
        """
        Parse an LLM response to extract a chess move.
        
        Args:
            response: The raw LLM response text
            
        Returns:
            ParseResult with success status and extracted move or error message
        """
        if not response or not response.strip():
            return ParseResult(
                success=False,
                error_message="Empty response"
            )
        
        response = response.strip()
        
        # First, try to find explicit "final answer" patterns
        move = self._extract_from_final_answer(response)
        if move:
            return ParseResult(success=True, move=move)
        
        # If no explicit final answer, try to find the last valid move in the response
        move = self._extract_last_valid_move(response)
        if move:
            return ParseResult(success=True, move=move)
        
        # No valid move found
        return ParseResult(
            success=False,
            error_message="No valid chess move found in response"
        )


def parse_chess_response(response: str) -> ParseResult:
    """
    Convenience function to parse a chess response.
    
    Args:
        response: The raw LLM response text
        
    Returns:
        ParseResult with success status and extracted move or error message
    """
    parser = ChessResponseParser()
    return parser.parse(response)