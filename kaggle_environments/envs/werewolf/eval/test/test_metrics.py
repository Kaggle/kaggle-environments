import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from kaggle_environments.envs.werewolf.eval.metrics import GameSetEvaluator, POLARIX_AVAILABLE

# Use an absolute path to the test data, making the test runnable from any directory
DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
SMOKE_TEST_DATA_DIR = str(DIR_PATH / "data" / "w_replace")


@pytest.mark.skip("Long running test.")
def test_evaluator_smoke_test():
    """
    Smoke test to ensure the GameSetEvaluator runs end-to-end without errors
    using the real test dataset.
    """
    # Initialize the evaluator with the absolute path to the test data
    evaluator = GameSetEvaluator(input_dir=SMOKE_TEST_DATA_DIR)

    # Run the full evaluation
    evaluator.evaluate(gte_samples=10)  # Use fewer samples for a faster test run

    # Capture the output of print_results to keep the test output clean
    with io.StringIO() as buf, redirect_stdout(buf):
        evaluator.print_results()
        output = buf.getvalue()

    # The main assertions are that the code runs without exceptions.
    # We can also do a basic check on the output.
    assert "Agent:" in output
    assert "Overall Win Rate:" in output
    if POLARIX_AVAILABLE:
        assert "Game Theoretic Evaluation (GTE):" in output
