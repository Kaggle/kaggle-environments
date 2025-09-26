import pytest

from kaggle_environments.envs.werewolf.game.actions import filter_language

test_data = [
    # Test 1: Basic lowercase substitution
    ("We must kill the monster.", "We must eliminate the monster."),
    # Test 2: Title case substitution
    ("Killing is wrong.", "Eliminating is wrong."),
    # Test 3: Uppercase substitution
    ("The town should not LYNCH anyone.", "The town should not EXILE anyone."),
    # Test 4: Word boundary check (should not affect "skill")
    ("His skill is unparalleled.", "His skill is unparalleled."),
    # Test 5: Mixed case and multiple substitutions
    (
        "The Mob will lynch the player they think will Kill them.",
        "The Mob will exile the player they think will Eliminate them.",
    ),
    # Test 6: Handling different word endings ('-ed', '-s')
    ("He killed the dragon, and she kills the goblin.", "He eliminated the dragon, and she eliminates the goblin."),
    # Test 7: No inappropriate words, should return original string
    ("This is a perfectly safe sentence.", "This is a perfectly safe sentence."),
    # Test 8: A more complex sentence with a third rule ('murder')
    (
        "The detective solved the Murder, preventing the killer from killing again.",
        "The detective solved the Remove, preventing the eliminator from eliminating again.",
    ),
    # Test 9: A tricky title case that isn't at the start of a sentence
    ("I think Killing is not the answer.", "I think Eliminating is not the answer."),
]


@pytest.mark.parametrize("input_text, expected_text", test_data)
def test_clean_script_scenarios(input_text, expected_text):
    """
    Tests the clean_script_preserve_case function with various scenarios.
    """
    assert filter_language(input_text) == expected_text


def test_empty_string():
    """
    Tests that an empty string input results in an empty string output.
    """
    assert filter_language("") == ""
