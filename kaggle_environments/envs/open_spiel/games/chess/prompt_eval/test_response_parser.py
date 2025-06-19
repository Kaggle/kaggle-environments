"""
Tests for chess response parser.

Tests various LLM response formats and edge cases.
"""

from absl.testing import absltest
from response_parser import ChessResponseParser, parse_chess_response, ParseResult


class ChessResponseParserTest(absltest.TestCase):
    """Test cases for ChessResponseParser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = ChessResponseParser()
    
    def test_explicit_final_answer_formats(self):
        """Test various explicit final answer formats."""
        test_cases = [
            ("Final Answer: Nf3", "Nf3"),
            ("**Final Answer:** e4", "e4"),
            ("FINAL ANSWER: Qxd7+", "Qxd7+"),
            ("final answer: O-O", "O-O"),
            ("Answer: Bxc6", "Bxc6"),
            ("Move: Ra8#", "Ra8#"),
            ("**FINAL ANSWER:** O-O-O", "O-O-O"),
            ("Final Answer: e2e4", "e2e4"),  # UCI format
            ("answer: d7d8q", "d7d8q"),  # UCI promotion
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move, f"Expected {expected_move}, got {result.move}")
    
    def test_with_reasoning_text(self):
        """Test responses with lots of reasoning before the final answer."""
        response = """
        Looking at this position, I need to consider several factors:
        1. King safety is paramount
        2. Central control is important
        3. Piece development should be prioritized
        
        After analyzing the position thoroughly, I believe the best move is to develop the knight.
        The knight on g1 can go to f3, which attacks the center and prepares castling.
        
        Final Answer: Nf3
        """
        
        result = self.parser.parse(response)
        self.assertTrue(result.success)
        self.assertEqual(result.move, "Nf3")
    
    def test_markdown_formatting(self):
        """Test responses with markdown formatting."""
        test_cases = [
            ("I think **Final Answer: Qd5** is the best move here.", "Qd5"),
            ("After careful analysis:\n\n**Final Answer:** *Bb5+*", "Bb5+"),
            ("**FINAL ANSWER: O-O**", "O-O"),
            ("The answer is **e4**. Final Answer: e4", "e4"),
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move)
    
    def test_castling_variations(self):
        """Test different castling notations."""
        test_cases = [
            ("Final Answer: O-O", "O-O"),
            ("Final Answer: O-O-O", "O-O-O"),
            ("Final Answer: 0-0", "0-0"),
            ("Final Answer: 0-0-0", "0-0-0"),
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move)
    
    def test_fallback_to_last_move(self):
        """Test fallback when no explicit final answer is given."""
        response = """
        I'm analyzing this chess position. Let me consider a few options:
        - Nf3 develops the knight nicely
        - e4 controls the center 
        - d4 is also a good central move
        
        Actually, I think Nf3 is the strongest here.
        """
        
        result = self.parser.parse(response)
        self.assertTrue(result.success)
        self.assertEqual(result.move, "Nf3")
    
    def test_multiple_moves_takes_last(self):
        """Test that when multiple moves are mentioned, it takes the last valid one."""
        response = """
        I could play e4, but then Nf3 might be better.
        Actually, let me reconsider - Bb5 looks interesting.
        Final Answer: Qd4
        """
        
        result = self.parser.parse(response)
        self.assertTrue(result.success)
        self.assertEqual(result.move, "Qd4")
    
    def test_promotions(self):
        """Test pawn promotion notation."""
        test_cases = [
            ("Final Answer: e8=Q+", "e8=Q+"),
            ("Final Answer: a1=R", "a1=R"),
            ("Final Answer: h8=N#", "h8=N#"),
            ("Final Answer: d8=B", "d8=B"),
            ("Final Answer: e7e8q", "e7e8q"),  # UCI format
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move)
    
    def test_captures(self):
        """Test capture notation."""
        test_cases = [
            ("Final Answer: Nxd4", "Nxd4"),
            ("Final Answer: exd5", "exd5"),
            ("Final Answer: Qxf7+", "Qxf7+"),
            ("Final Answer: axb6", "axb6"),
            ("Final Answer: Rxe8#", "Rxe8#"),
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move)
    
    def test_check_and_checkmate(self):
        """Test check and checkmate notation."""
        test_cases = [
            ("Final Answer: Qh5+", "Qh5+"),
            ("Final Answer: Rd8#", "Rd8#"),
            ("Final Answer: Bb5+", "Bb5+"),
            ("Final Answer: Ra1#", "Ra1#"),
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move)
    
    def test_ambiguous_notation(self):
        """Test disambiguated moves."""
        test_cases = [
            ("Final Answer: Nbd7", "Nbd7"),
            ("Final Answer: R1a3", "R1a3"),
            ("Final Answer: Qd1d4", "Qd1d4"),
            ("Final Answer: N1f3", "N1f3"),
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move)
    
    def test_case_insensitive(self):
        """Test case insensitive parsing."""
        test_cases = [
            ("final answer: nf3", "nf3"),
            ("FINAL ANSWER: E4", "E4"),
            ("Final Answer: qXd7", "qXd7"),
            ("answer: o-o", "o-o"),
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move)
    
    def test_weird_formatting(self):
        """Test responses with weird formatting and extra tokens."""
        test_cases = [
            ("***Final Answer:*** Nf3 !!!", "Nf3"),
            ("🎯 Final Answer: e4 🎯", "e4"),
            (">>> Final Answer: Qd5 <<<", "Qd5"),
            ("Final Answer:    Bb5+   ", "Bb5+"),
            ("FINAL  ANSWER  :  O-O", "O-O"),
            ("Final\nAnswer:\nNf3", "Nf3"),
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move)
    
    def test_no_valid_move_found(self):
        """Test responses with no valid chess moves."""
        test_cases = [
            "I'm thinking about this position but can't decide.",
            "This is a complex position requiring deep analysis.",
            "Final Answer: I pass",
            "Final Answer: 42",
            "The best move is unclear to me.",
            "",
            "   ",
        ]
        
        for response in test_cases:
            result = self.parser.parse(response)
            self.assertFalse(result.success, f"Should have failed but succeeded for: {response}")
            self.assertIsNone(result.move)
            self.assertIsNotNone(result.error_message)
    
    def test_invalid_moves(self):
        """Test responses with invalid move notation."""
        test_cases = [
            "Final Answer: z9",  # Invalid square
            "Final Answer: Kx",  # Incomplete capture
            "Final Answer: e5e5",  # Same square UCI
            "Final Answer: i1",  # Invalid file
            "Final Answer: a0",  # Invalid rank
        ]
        
        for response in test_cases:
            result = self.parser.parse(response)
            self.assertFalse(result.success, f"Should have failed but succeeded for: {response}")
    
    def test_mixed_valid_invalid_takes_valid(self):
        """Test responses with mix of valid and invalid moves."""
        response = """
        I could try z9 but that's not a real square.
        Maybe e5e5 but that doesn't make sense.
        Actually, Final Answer: Nf3
        """
        
        result = self.parser.parse(response)
        self.assertTrue(result.success)
        self.assertEqual(result.move, "Nf3")
    
    def test_convenience_function(self):
        """Test the convenience function."""
        result = parse_chess_response("Final Answer: e4")
        self.assertTrue(result.success)
        self.assertEqual(result.move, "e4")
        self.assertIsInstance(result, ParseResult)
    
    def test_uci_format_moves(self):
        """Test UCI format moves."""
        test_cases = [
            ("Final Answer: e2e4", "e2e4"),
            ("Final Answer: g1f3", "g1f3"),
            ("Final Answer: a7a8q", "a7a8q"),  # Promotion
            ("Final Answer: e1g1", "e1g1"),   # Castling in UCI
        ]
        
        for response, expected_move in test_cases:
            result = self.parser.parse(response)
            self.assertTrue(result.success, f"Failed to parse: {response}")
            self.assertEqual(result.move, expected_move)
    
    def test_really_verbose_response(self):
        """Test parsing a very verbose LLM response."""
        response = """
        This is a fascinating chess position that requires careful analysis. Let me break down the key factors:
        
        **Position Assessment:**
        - Material is roughly equal
        - White has better piece coordination
        - Black's king looks slightly exposed
        
        **Candidate Moves:**
        1. Nf3 - develops the knight and controls central squares
        2. e4 - claims the center immediately
        3. d4 - alternative central advance
        4. Bc4 - develops the bishop toward the opponent's king
        
        **Deep Analysis:**
        After considering all these options, I believe the knight development is most principled here.
        The knight on f3 supports the center, prepares castling, and maintains flexibility.
        
        While e4 is tempting, it might be premature without proper piece support.
        The bishop move Bc4 is interesting but perhaps too aggressive early on.
        
        **Conclusion:**
        Based on classical opening principles and the specific characteristics of this position,
        I'm confident that the knight development is the strongest continuation.
        
        **Final Answer: Nf3**
        
        This move exemplifies sound opening play and gives White excellent winning chances.
        """
        
        result = self.parser.parse(response)
        self.assertTrue(result.success)
        self.assertEqual(result.move, "Nf3")


if __name__ == '__main__':
    absltest.main()