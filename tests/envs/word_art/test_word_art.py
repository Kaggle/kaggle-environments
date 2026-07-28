import pytest

from kaggle_environments import make
from kaggle_environments.envs.word_art.word_art import (
    _matches_target,
    _sample_words,
    _sanitize_art,
    check_art,
)
from kaggle_environments.errors import DeadlineExceeded


def _make(**config):
    return make("word_art", configuration=config)


# Tests that exercise scoring/role/history mechanics need a deterministic way
# for the artist to convey the target word to the guesser without tripping the
# no-word-in-art enforcement (which always applies). We encode each letter as
# a zero-padded 2-digit code (A=01, B=02, ..., Z=26): the encoded art contains
# only digits, so the word never appears as a substring forwards or reversed.
# Tests that DO want to test enforcement override this and send the raw word
# (see the "no-word-in-art enforcement" section).


def _encode_word(word):
    return "".join(f"{ord(c) - ord('A') + 1:02d}" for c in word.upper())


def _decode_art(art):
    return "".join(chr(int(art[i:i + 2]) + ord('A') - 1) for i in range(0, len(art), 2))


def silent(observation, configuration):
    return ""


def cheating(observation, configuration):
    """Artist encodes the word in digits; guesser decodes."""
    if observation.role == "artist":
        return _encode_word(observation.target_word)
    return _decode_art(observation.teammate_art)


def lazy_second_try(observation, configuration):
    """Guesses 'NOPE' on the first attempt, then the correct word on the second."""
    if observation.role == "artist":
        return _encode_word(observation.target_word)
    if not observation.previous_guesses:
        return "NOPE"
    return _decode_art(observation.teammate_art)


def random_letter(observation, configuration):
    if observation.role == "artist":
        return "A"
    return "Z"


def test_game_completes_default():
    env = _make(num_rounds=3, seed=42)
    env.run([silent, silent, silent, silent])
    j = env.toJSON()
    assert j["statuses"] == ["DONE", "DONE", "DONE", "DONE"]
    # All guesses wrong, every team scores 0
    assert j["rewards"] == [0, 0, 0, 0]


def test_guess_points_first_attempt_scoring():
    """Cheating both teams with an explicit [2, 1, 1] guess_points table:
    first-attempt correct = 2 pts per round.
    """
    env = _make(num_rounds=4, seed=1, guess_points=[2, 1, 1])
    env.run([cheating, cheating, cheating, cheating])
    j = env.toJSON()
    assert j["rewards"] == [8, 8, 8, 8]  # 2 pts * 4 rounds


def test_second_try_scores_one():
    """A team that always lands on attempt 2 scores guess_points[1] per round.
    Under the default guess_points ([1, 1, 1]) that's 1 pt per round.
    """
    env = _make(num_rounds=3, seed=9)
    env.run([lazy_second_try, lazy_second_try, lazy_second_try, lazy_second_try])
    j = env.toJSON()
    assert j["rewards"] == [3, 3, 3, 3]


def test_blue_first_yellow_misses():
    """Blue cheats (2 pts/round under [2,1,1]); yellow always wrong (0 pts/round)."""
    env = _make(num_rounds=4, seed=1, guess_points=[2, 1, 1])
    env.run([cheating, cheating, silent, silent])
    j = env.toJSON()
    assert j["rewards"][0] == 8
    assert j["rewards"][1] == 8
    assert j["rewards"][2] == 0
    assert j["rewards"][3] == 0


def test_history_shape_and_attempts():
    """History entries record per-team guess lists and points; the team that
    succeeds first has fewer entries than the team that exhausts all attempts."""
    env = _make(num_rounds=2, seed=1, guess_points=[2, 1, 1])
    env.run([cheating, cheating, silent, silent])
    j = env.toJSON()
    final_history = j["steps"][-1][0]["observation"]["history"]
    assert len(final_history) == 2
    for entry in final_history:
        # Blue cheats successfully on attempt 1: exactly one guess, 2 points.
        assert len(entry["blue_guesses"]) == 1
        assert entry["blue_points"] == 2
        # Yellow silently fails 3 times: three empty guesses, 0 points.
        assert len(entry["yellow_guesses"]) == 3
        assert entry["yellow_points"] == 0


def test_asymmetric_attempts_finish():
    """When blue scores on attempt 1 and yellow needs all 3, the game still
    progresses cleanly through every round. Episode terminates with both DONE."""
    env = _make(num_rounds=3, seed=4, max_attempts=3)
    env.run([cheating, cheating, silent, silent])
    j = env.toJSON()
    assert j["statuses"] == ["DONE", "DONE", "DONE", "DONE"]
    # Step count: 1 init + per round (1 art step + up to max_attempts guess steps).
    # With max_attempts=3 and yellow always failing, each round has 1 + 3 = 4 steps.
    # Total: 1 + 3*4 = 13.
    assert len(j["steps"]) == 1 + 3 * (1 + 3)


def test_guesser_sees_previous_guesses():
    """Guesser receives previous_guesses on the 2nd/3rd attempt so they don't repeat."""

    seen: dict[str, list[list[str]]] = {"blue": []}

    def recorder(observation, configuration):
        if observation.role == "artist":
            return _encode_word(observation.target_word)
        if observation.team == "blue":
            seen["blue"].append(list(observation.previous_guesses))
        # Always wrong → forces all 3 attempts
        return f"WRONG{len(observation.previous_guesses)}"

    env = _make(num_rounds=1, seed=2)
    env.run([recorder, recorder, recorder, recorder])
    # Three attempts for blue, with previous_guesses growing each time:
    #   attempt 1: []          → guess "WRONG0"
    #   attempt 2: ["WRONG0"]  → guess "WRONG1"
    #   attempt 3: ["WRONG0", "WRONG1"] → guess "WRONG2"
    assert seen["blue"] == [[], ["WRONG0"], ["WRONG0", "WRONG1"]]


def test_attempts_used_visible_to_both_teams():
    """Public counters blue_attempts_used / yellow_attempts_used update each
    sub-step and are visible to all four agents. We check the sub-step after
    yellow's second wrong guess (the last guess step before the round closes
    and counters reset)."""
    env = _make(num_rounds=1, seed=8)
    env.run([cheating, cheating, silent, silent])
    j = env.toJSON()
    # Find a guess sub-step where yellow has 2 wrong attempts on record and
    # blue has already scored on attempt 1.
    found = False
    for step in j["steps"]:
        if not step:
            continue
        obs = step[0]["observation"]
        if obs.get("phase") == "guess" and obs.get("yellow_attempts_used") == 2:
            for i in range(4):
                assert step[i]["observation"]["blue_attempts_used"] == 1
                assert step[i]["observation"]["yellow_attempts_used"] == 2
            found = True
            break
    assert found, "expected a guess sub-step with blue=1, yellow=2 attempts"


def test_role_rotation():
    """After round 0 completes, artist/guesser roles swap within each team."""
    env = _make(num_rounds=2, seed=3)
    statuses_initial = [s["status"] for s in env.state]
    assert statuses_initial[0] == "ACTIVE"  # blue artist round 0
    assert statuses_initial[1] == "INACTIVE"  # blue guesser round 0
    assert statuses_initial[2] == "ACTIVE"
    assert statuses_initial[3] == "INACTIVE"

    env.run([cheating, cheating, cheating, cheating])
    j = env.toJSON()
    assert j["statuses"] == ["DONE"] * 4

    # Find the first step of round 1 (phase=art, current_round=1).
    round1_art_step = next(
        s
        for s in j["steps"]
        if s and s[0]["observation"].get("current_round") == 1 and s[0]["observation"].get("phase") == "art"
    )
    assert round1_art_step[0]["observation"]["role"] == "guesser"
    assert round1_art_step[1]["observation"]["role"] == "artist"
    assert round1_art_step[2]["observation"]["role"] == "guesser"
    assert round1_art_step[3]["observation"]["role"] == "artist"


def test_seed_reproducibility():
    env1 = _make(num_rounds=3, seed=99)
    env1.run([cheating, cheating, cheating, cheating])
    env2 = _make(num_rounds=3, seed=99)
    env2.run([cheating, cheating, cheating, cheating])
    j1 = env1.toJSON()
    j2 = env2.toJSON()
    words1 = [h["word"] for h in j1["steps"][-1][0]["observation"]["history"]]
    words2 = [h["word"] for h in j2["steps"][-1][0]["observation"]["history"]]
    assert words1 == words2
    assert len(words1) == 3


def test_word_hidden_from_guesser():
    env = _make(num_rounds=1, seed=11)
    env.run([cheating, cheating, cheating, cheating])
    j = env.toJSON()
    for step in j["steps"]:
        for i, s in enumerate(step):
            obs = s["observation"]
            if obs.get("role") == "guesser":
                assert obs.get("target_word", "") == "", f"agent {i} (guesser) leaked target_word"


def test_art_hidden_from_opponent():
    """During guess sub-steps, only the team's own guesser sees teammate_art."""
    env = _make(num_rounds=2, seed=5)
    env.run([cheating, cheating, silent, silent])
    j = env.toJSON()
    for step in j["steps"]:
        if not step:
            continue
        phase = step[0]["observation"].get("phase")
        if phase != "guess":
            continue
        rnd = step[0]["observation"]["current_round"]
        blue_guesser = 1 - (rnd % 2)
        yellow_guesser = 2 + (1 - (rnd % 2))
        for i, s in enumerate(step):
            art = s["observation"].get("teammate_art", "")
            if i not in (blue_guesser, yellow_guesser):
                assert art == "", f"agent {i} leaked teammate_art in guess phase"


def test_case_insensitive_guess():
    def lowercase_guess(observation, configuration):
        if observation.role == "artist":
            return _encode_word(observation.target_word)
        # Decode → uppercase word, then lowercase before submitting. The env
        # should still accept it (matching is case-insensitive).
        return _decode_art(observation.teammate_art).lower()

    env = _make(num_rounds=2, seed=17, guess_points=[2, 1, 1])
    env.run([lowercase_guess] * 4)
    j = env.toJSON()
    # First-try correct → 2 per round * 2 rounds = 4
    assert j["rewards"] == [4, 4, 4, 4]


def test_empty_guess_is_wrong():
    """Yellow submits empty 3 times — 0 points; blue scores normally."""
    env = _make(num_rounds=2, seed=13)
    env.run([cheating, cheating, silent, silent])
    j = env.toJSON()
    assert j["rewards"][2] == 0
    assert j["rewards"][3] == 0
    assert j["rewards"][0] > 0


def test_configurable_guess_points():
    """Override max_attempts=2 and guess_points=[5, 1]. lazy_second_try wins
    on attempt 2 → 1 pt per round.
    """
    env = _make(num_rounds=2, seed=6, max_attempts=2, guess_points=[5, 1])
    env.run([lazy_second_try] * 4)
    j = env.toJSON()
    assert j["rewards"] == [2, 2, 2, 2]


def test_renderer():
    env = _make(num_rounds=2, seed=4)
    env.run([cheating, cheating, silent, silent])
    out = env.render(mode="ansi")
    assert isinstance(out, str)
    assert "Score" in out


def test_max_art_chars_truncation():
    def long_art_cheater(observation, configuration):
        if observation.role == "artist":
            return "X" * 1000 + observation.target_word
        return observation.teammate_art

    env = _make(num_rounds=2, seed=21, max_art_chars=500)
    env.run([long_art_cheater] * 4)
    j = env.toJSON()
    assert j["rewards"] == [0, 0, 0, 0]


# --- No-word-in-art enforcement --------------------------------------------
#
# These tests exercise the (always-on) enforcement that disqualifies art
# containing the target word. They use _make() directly — there's no flag
# to flip; enforcement is unconditional.


def _word_smuggler(transform):
    """Build an agent whose artist applies `transform(word)` and whose guesser
    parrots whatever they see back as the answer."""

    def agent(observation, configuration):
        if observation.role == "artist":
            return transform(observation.target_word)
        # Guesser: strip the placeholder, fall back to teammate_art
        art = observation.teammate_art or ""
        return art

    return agent


def _run_smuggle(transform, *, seed=1):
    env = make("word_art", configuration={"num_rounds": 1, "seed": seed})
    env.run([_word_smuggler(transform)] * 4)
    return env.toJSON()


def test_verbatim_word_is_disqualified():
    j = _run_smuggle(lambda w: w)
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is True
    assert entry["yellow_art_disqualified"] is True
    # Original art preserved in history for replay transparency.
    assert entry["word"] in entry["blue_art"]
    # Both teams fail to guess (placeholder doesn't contain the word).
    assert entry["blue_points"] == 0
    assert entry["yellow_points"] == 0


def test_lowercase_verbatim_word_is_disqualified():
    j = _run_smuggle(lambda w: w.lower())
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is True


def test_letter_by_letter_with_spaces_is_disqualified():
    j = _run_smuggle(lambda w: " ".join(w))
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is True


def test_letter_by_letter_with_hyphens_is_disqualified():
    j = _run_smuggle(lambda w: "-".join(w))
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is True


def test_letter_by_letter_with_newlines_is_disqualified():
    j = _run_smuggle(lambda w: "\n".join(w))
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is True


def test_letter_by_letter_with_periods_is_disqualified():
    j = _run_smuggle(lambda w: ".".join(w))
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is True


def test_word_padded_with_emoji_chars_is_disqualified():
    # Non-alphanumeric padding still gets stripped before the substring check.
    j = _run_smuggle(lambda w: f"!!! ~~~ {w} ~~~ !!!")
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is True


def test_reversed_word_is_disqualified():
    j = _run_smuggle(lambda w: w[::-1])
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is True


def test_safe_art_is_not_disqualified():
    # Use a deterministic stand-in for the word that doesn't contain it.
    j = _run_smuggle(lambda w: "X" * len(w))
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is False
    assert entry["yellow_art_disqualified"] is False


# --- Spaced-out non-target labels caught by the any-word check --------------
#
# The consecutive-letter check ('TOP', 'HOUSE') and the target-word check
# (letters-of-the-target with any/no separators) both catch obvious labels.
# The gap is a non-target label spelled out with same-line separators --
# 'A R O U N D', 'H-O-U-S-E', 'grid_view' -- which the consecutive check
# doesn't fire on and the target-word check ignores (wrong word). These
# tests lock in the same-line separator-aware any-word check. Any
# non-letter, non-newline joiner is treated as a separator, so evasions
# via '|', '/', ':', ',', '*', em-dash, non-breaking space etc. are all
# rejected.


def _art_check(art):
    from kaggle_environments.envs.word_art.word_art import _art_contains_any_word
    return _art_contains_any_word(art)


@pytest.mark.parametrize("art", [
    "A R O U N D",         # spaces
    "A.R.O.U.N.D",         # dots
    "H-O-U-S-E",           # dashes
    "T\tO\tP",             # tabs
    "grid_view",           # underscore compound
    "T O P",               # 3-letter spaced
    "H|O|U|S|E",           # pipes
    "H/O/U/S/E",           # slashes
    "H:O:U:S:E",           # colons
    "H,O,U,S,E",           # commas
    "H*O*U*S*E",           # asterisks
    "H=O=U=S=E",           # equals
    "H—O—U—S—E",  # em-dashes
    "H–O–U–S–E",  # en-dashes
    "H\xa0O\xa0U\xa0S\xa0E",         # non-breaking spaces
    "A1B2C",               # digits used as separators
])
def test_spaced_non_target_label_is_disqualified(art):
    assert _art_check(art) is True, f"expected {art!r} to be flagged as text"


@pytest.mark.parametrize("art", [
    "OOO",               # eye cluster
    "III",               # columns
    "V V V V",           # zigzag decoration
    "T T T\nT T T",      # texture grid (single letter across a 2D layout)
    "OO H2",             # short clusters + digit-adjacent
    "O   O\n  V\n \\_/", # smiley (letters spread across lines)
    "o o o\n o\no o o",  # die face
    "\n   *\n \\ | /\n---+---\n / | \\\n   *\n",  # snowflake example
    # Two-letter decorative patterns with separators. The spaced-out check
    # requires 3+ distinct letters, so these pass -- they're visual
    # patterns (dice, brickwork, zigzag), not spelled-out labels.
    "| o   X     X     X  o  |",  # die/domino row: 5 letters, 2 distinct
    "A B A B A B",                 # brick row
    "X . X . X . X",               # single-letter row with dot spacers
    "V W V W V W",                 # alternating zigzag
    "o X o",                       # 3-letter, 2-distinct decoration
])
def test_visual_letter_elements_pass(art):
    assert _art_check(art) is False, f"expected {art!r} to pass the any-word check"


def test_letters_on_different_lines_do_not_chain():
    """Same-line-only chaining: `T`, `O`, `P` each on their own line should
    NOT combine into 'TOP' -- otherwise a 2D layout with letter-based
    visual elements becomes unusable. The column check requires 4+
    letters with 4+ distinct, so a 3-letter column passes."""
    art = "T\n O\n  P"
    assert _art_check(art) is False


# --- Column labels (letters lined up down a column) ------------------------
#
# The row-oriented checks intentionally don't chain across newlines -- 2D
# art with letters scattered around must stay usable. The column check
# runs the same two regexes on the transposed art with tighter thresholds
# (4+ letters, 4+ distinct) to plug the specific hole where a model
# stacks a label like 'F' / 'L' / 'A' / 'G' one per line, or draws a
# column of same-letter rows whose leading letters spell a hint. Because
# the column check reuses `_SPACED_WORD_RE`, decorative dividers or blank
# rows between the label letters ('F' / '===' / 'L' / '===' / ...) are
# also caught -- the transposed row for that column is 'F===L===A===G',
# which reads as spaced text.


@pytest.mark.parametrize("art", [
    # FLAG boxed one-letter-per-line (real evasion caught in replay analysis).
    "     ___\n    |   |\n    | F |\n    | L |\n    | A |\n    | G |\n    |___|",
    # Rainbow mnemonic: same-letter row columns whose leading letters spell
    # 'ROYGB' (each row on its own has only 1 distinct letter so the
    # consecutive check doesn't fire).
    "  1. RRRRRRRRR\n  2. OOOOOOOOO\n  3. YYYYYYYYY\n  4. GGGGGGGGG\n  5. BBBBBBBBB",
    # Bare 4-letter column of distinct letters.
    "F\nL\nA\nG",
    # Longer column ('LABEL' - 4 distinct: L, A, B, E).
    "L\nA\nB\nE\nL",
    # Column of distinct letters with leading indentation on each line.
    "      W\n      O\n      R\n      D",
    # Blank rows between every letter -- transposed row is 'F L A G',
    # 4 letters with 3 separators, matches the column spaced check.
    "F\n\nL\n\nA\n\nG",
    # Decorative dividers between letters -- transposed column across the
    # label letters reads 'F===L===A===G' with '=' as separators.
    "| F |\n=====\n| L |\n=====\n| A |\n=====\n| G |",
])
def test_column_label_is_disqualified(art):
    assert _art_check(art) is True, f"expected {art!r} to be flagged as a column label"


@pytest.mark.parametrize("art", [
    # 3 letters -- under the 4-letter threshold, still visual decoration.
    "T\nO\nP",
    # 4 letters but only 3 distinct -- likely repeated decoration, not text.
    "O\nI\nI\nI",
    # 5 letters, 2 distinct -- alternating brick / zigzag pattern.
    "A\nB\nA\nB\nA",
    # Two consecutive letters + one gap + two consecutive letters: the
    # spaced regex needs letter-(sep+letter){2,}, so 'FL AG' (transposed
    # from FL / blank / AG) has only one separator group and passes.
    "F\nL\n\nA\nG",
])
def test_column_decoration_passes(art):
    assert _art_check(art) is False, f"expected {art!r} to pass the column check"


def test_tab_indented_column_label_is_disqualified():
    """The column check counts characters, not render width, so a tab-indented
    label lines up for the guesser but not for the check. `_sanitize_art`
    expands tabs first so both see the same grid."""
    tabbed = "\tF\n        L\n\tA\n        G"
    spaced = "        F\n        L\n        A\n        G"
    assert check_art(tabbed, "CAT", 4000) == check_art(spaced, "CAT", 4000)
    assert check_art(tabbed, "CAT", 4000)[0] == "contains_words"


def test_guesser_sees_placeholder_on_disqualification():
    """When the artist's art is disqualified, the guesser's teammate_art is
    replaced with the placeholder string and the original is NOT leaked."""

    captured = {"art_seen": None, "word": None}

    def recorder(observation, configuration):
        if observation.role == "artist":
            return observation.target_word  # will be disqualified
        if observation.team == "blue":
            captured["art_seen"] = observation.teammate_art
        return "WHATEVER"

    env = make("word_art", configuration={"num_rounds": 1, "seed": 3})
    env.run([recorder] * 4)
    j = env.toJSON()
    target = j["steps"][-1][0]["observation"]["history"][0]["word"]
    captured["word"] = target

    assert captured["art_seen"] is not None
    assert "disqualified" in captured["art_seen"].lower()
    # The actual word must not be present in what the guesser saw.
    assert target.lower() not in captured["art_seen"].lower()


def test_disqualification_does_not_block_guessing():
    """Even with a disqualified art panel, the guesser still gets their full
    attempt budget and can still score if they correctly guess the word."""

    def wrong_guesser(observation, configuration):
        if observation.role == "artist":
            return observation.target_word  # will be disqualified
        # Guesser ignores the placeholder and just guesses wrong every attempt
        # -- the test only checks structure (3 attempts allowed, no points).
        return f"WRONG{len(observation.previous_guesses)}"

    env = make("word_art", configuration={"num_rounds": 1, "seed": 7})
    env.run([wrong_guesser] * 4)
    j = env.toJSON()
    entry = j["steps"][-1][0]["observation"]["history"][0]
    # Both teams cheated → both disqualified, guessers still got their 3
    # attempts each.
    assert entry["blue_art_disqualified"] is True
    assert len(entry["blue_guesses"]) == 3
    assert entry["yellow_art_disqualified"] is True
    assert len(entry["yellow_guesses"]) == 3
    assert entry["blue_points"] == 0
    assert entry["yellow_points"] == 0


def test_substring_word_is_disqualified():
    """If the target word appears as a substring of a longer label (e.g. the
    artist labels their drawing 'a CAT-shape'), enforcement still trips. We
    accept that this also catches innocuous substrings — the artist should
    avoid labelling at all."""
    j = _run_smuggle(lambda w: f"A {w}-shape")
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is True


# --- Config surfaced on observation ----------------------------------------


def test_guess_points_surfaced_on_observation():
    """The env must surface guess_points on every agent's observation so
    the harness can render the scoring table. Hardcoding in the harness
    silently lies to the model when this config is non-default."""
    env = _make(num_rounds=1, seed=1, guess_points=[3, 2, 1])
    for s in env.state:
        assert list(s.observation.guess_points) == [3, 2, 1]


def test_include_art_history_surfaced_on_observation():
    """Same contract: the harness reads include_art_history off the obs to
    decide whether to include past art in the history block."""
    env = _make(num_rounds=1, seed=1, include_art_history=False)
    for s in env.state:
        assert s.observation.include_art_history is False


def test_max_art_chars_surfaced_on_observation():
    """Same contract as guess_points: the artist prompt needs to warn
    about the truncation length, so the env must hand it to the harness."""
    env = _make(num_rounds=1, seed=1, max_art_chars=2500)
    for s in env.state:
        assert s.observation.max_art_chars == 2500


def test_config_defaults_surfaced_on_observation():
    """Defaults match word_art.json (guess_points=[1]*max_attempts,
    include_art_history=True, max_art_chars=4000). first_try_bonus is
    no longer part of the schema.
    """
    env = _make(num_rounds=1, seed=1)
    for s in env.state:
        assert list(s.observation.guess_points) == [1, 1, 1]
        assert s.observation.include_art_history is True
        assert s.observation.max_art_chars == 4000
        assert "first_try_bonus" not in s.observation.keys()


def test_no_hidden_keys_leak_into_any_observation():
    """Round-scoped hidden state (word list, in-progress art, guesses, per-team
    done flags) must live on ``env``, not on any player's observation. A leak
    onto ``state[i].observation`` (typically under an underscore-prefixed key)
    would ship the target words and the opposing team's in-progress state
    into the replay JSON and into whatever the agent process receives -- a
    custom agent could then short-circuit the art channel entirely, and any
    future prompt change that dumps the raw obs would silently leak.

    We run a couple of rounds so both the art-phase and guess-phase code
    paths get a chance to set fields on the observation.
    """
    env = _make(num_rounds=2, seed=1)
    env.run(["random"] * 4)
    forbidden_prefixes = ("_words", "_round_")
    forbidden_keys = {"words", "target_words"}  # unprefixed variants
    for step in env.steps:
        for i, agent in enumerate(step):
            obs = agent.observation
            hidden = [
                k for k in obs.keys()
                if k in forbidden_keys or any(k.startswith(p) for p in forbidden_prefixes)
            ]
            assert not hidden, (
                f"agent {i}'s observation leaks hidden keys {hidden}; "
                "round-scoped state must live on env.word_art_state"
            )


def test_guessed_correctly_visible_mid_round():
    """When one team scores on their first try but the other keeps guessing,
    the winning team's `{team}_guessed_correctly` bool must be True on
    intermediate steps -- the visualizer keys its correct/wrong slot
    coloring off this field. Before it was exposed publicly, the correct
    guess showed red until the round rolled into history.
    """
    def blue_scores_first(observation, configuration):
        # Round 0: agent 0 is blue artist. Encode the word; agent 1 is blue
        # guesser and decodes on attempt 1 for the win.
        role = observation.role
        team = observation.team
        if role == "artist" and team == "blue":
            return _encode_word(observation.target_word)
        if role == "guesser" and team == "blue":
            return _decode_art(observation.teammate_art)
        # Yellow team: always wrong so the round drags on for max_attempts.
        if role == "artist":
            return "* * *"
        return f"WRONG{len(observation.previous_guesses)}"

    env = _make(num_rounds=1, max_attempts=3, seed=1)
    env.run([blue_scores_first] * 4)

    # There must be at least one intermediate guess-phase step where blue
    # is flagged correct but yellow is still guessing.
    saw_intermediate = False
    for step in env.steps:
        obs0 = step[0].observation
        if obs0.phase != "guess" or obs0.current_round < len(obs0.history):
            continue
        if obs0.blue_guessed_correctly and not obs0.yellow_guessed_correctly:
            saw_intermediate = True
            # Every agent's obs sees the same snapshot.
            for i in range(4):
                assert step[i].observation.blue_guessed_correctly is True
                assert step[i].observation.yellow_guessed_correctly is False
    assert saw_intermediate, (
        "test setup: no intermediate step where blue had scored and yellow "
        "was still guessing; can't verify the visualizer signal"
    )


def test_disqualification_flags_public_during_guess_phase():
    """The public per-team art_disqualified bool + reason are set at
    guess-phase entry and mirrored to every agent's observation, so the
    visualizer can render the DISQUALIFIED label on the losing team's art
    without relying on the removed `_round_*` private fields."""
    def blue_cheats(observation, configuration):
        role = observation.role
        team = observation.team
        if role == "artist" and team == "blue":
            return observation.target_word  # trip the target-word check
        if role == "artist":
            return "* * *"
        return "WRONG"

    env = _make(num_rounds=1, seed=1)
    env.run([blue_cheats] * 4)
    saw_guess = False
    for step in env.steps:
        obs0 = step[0].observation
        if obs0.phase != "guess" or obs0.current_round < len(obs0.history):
            continue
        saw_guess = True
        for i in range(4):
            assert step[i].observation.blue_art_disqualified is True
            assert step[i].observation.yellow_art_disqualified is False
            assert step[i].observation.blue_art_disqualification_reason == "target_word"
            assert step[i].observation.yellow_art_disqualification_reason is None
    assert saw_guess


def test_current_target_visible_in_replay_during_guess_phase():
    """The visualizer relies on `target_word` being present on SOME agent's
    observation during the guess phase so it can show the reader which word
    is currently being guessed (history only covers COMPLETED rounds). The
    artist is INACTIVE during the guess phase, so keeping target_word on
    their observation is safe: the LLM never gets called again this round,
    but the field remains in the serialized step for the renderer to find.
    """
    from kaggle_environments.envs.word_art.word_art import (
        _blue_artist, _yellow_artist, _blue_guesser, _yellow_guesser,
    )

    env = _make(num_rounds=2, seed=1)
    env.run(["random"] * 4)
    # Walk every step; on any guess-phase step where the round is still
    # live (not yet rolled into history), at least one agent's
    # target_word must be set so the renderer has something to show.
    saw_guess_step = False
    for step in env.steps:
        obs0 = step[0].observation
        if obs0.phase != "guess":
            continue
        rnd = obs0.current_round
        if rnd < len(obs0.history):
            # Round already closed and rolled into history; the renderer
            # reads the target from history[rnd], no live field needed.
            continue
        saw_guess_step = True
        targets = {step[i].observation.target_word for i in range(4)}
        targets.discard("")
        assert targets, (
            f"no agent exposes target_word at guess-phase step "
            f"(round {rnd}); the visualizer would show a blank"
        )
        # And the guessers must NOT see it (agents only see their own obs,
        # but assert anyway since target_word is what the renderer keys on).
        for g_idx in (_blue_guesser(rnd), _yellow_guesser(rnd)):
            assert step[g_idx].observation.target_word == "", (
                f"guesser {g_idx} sees target_word leak in round {rnd}"
            )
        # The artist(s) are where the target lives, and both artists on the
        # two teams share the same target this round.
        blue_target = step[_blue_artist(rnd)].observation.target_word
        yellow_target = step[_yellow_artist(rnd)].observation.target_word
        assert blue_target and blue_target == yellow_target
    assert saw_guess_step, "test setup: no guess-phase step was observed"


def test_mid_game_missing_word_art_state_raises_clearly():
    """env.word_art_state is not preserved across env.clone() or JSON
    round-trips (Environment.clone constructs a fresh instance from
    ``steps`` and doesn't carry over user attributes). Simulate the
    lost-state case by deleting the attribute mid-game and confirm the
    interpreter raises a clear RuntimeError instead of AttributeError-ing
    deep inside process_step.
    """
    env = _make(num_rounds=2, seed=1)
    env.step([None] * 4)  # advance out of the phase="" init branch
    assert env.state[0].observation.phase != ""
    del env.word_art_state
    with pytest.raises(RuntimeError, match="word_art_state is missing"):
        env.step([None] * 4)


# --- Failure-status preservation across phase / round transitions ----------
#
# When the framework marks an agent TIMEOUT/ERROR/INVALID, the interpreter
# must NOT flip that status back to ACTIVE on the next phase transition.
# Otherwise a timed-out artist gets silently resurrected, times out again,
# and the round's failure is invisible in the replay.


def test_artist_timeout_preserved_into_guess_phase():
    """An artist that raises DeadlineExceeded during the art phase must
    remain TIMEOUT after the env transitions into the guess phase — NOT
    get flipped to INACTIVE by _set_guess_statuses."""

    def slow_artist(observation, configuration):
        if observation.role == "artist":
            return DeadlineExceeded()
        return observation.teammate_art

    env = _make(num_rounds=2, seed=1)
    env.run([slow_artist, slow_artist, slow_artist, slow_artist])
    # After the env runs, both timed-out artists from round 0 (agents 0 and 2)
    # should still carry TIMEOUT — not be silently flipped back to ACTIVE/DONE.
    assert env.state[0].status == "TIMEOUT"
    assert env.state[2].status == "TIMEOUT"


def test_timed_out_artist_not_resurrected_in_later_round():
    """An agent that timed out in round 0's art phase must NOT be flipped
    back to ACTIVE when _set_art_statuses runs at the start of round 2
    (the next even-numbered round, where agent 0 would otherwise be the
    blue artist again)."""

    # Time out the first artist call; thereafter everyone plays normally.
    state_bag = {"timed_out_once": False}

    def timeout_first_blue_artist(observation, configuration):
        if (
            not state_bag["timed_out_once"]
            and observation.role == "artist"
            and observation.team == "blue"
            and observation.current_round == 0
        ):
            state_bag["timed_out_once"] = True
            return DeadlineExceeded()
        if observation.role == "artist":
            return observation.target_word
        return observation.teammate_art

    env = _make(num_rounds=3, seed=1)
    env.run([timeout_first_blue_artist] * 4)
    # Agent 0 (blue artist on even rounds) must stay TIMEOUT for the whole
    # episode — even though _set_art_statuses runs at the start of round 2
    # and would otherwise flip them back to ACTIVE.
    assert env.state[0].status == "TIMEOUT"
    assert env.state[0].reward is None
    # Scan every recorded step: agent 0 was never ACTIVE after the timeout
    # fired in step 1 (so it was never even given a chance to time out again).
    j = env.toJSON()
    post_timeout_active = [
        i
        for i, step in enumerate(j["steps"])
        if i > 0 and step[0]["status"] == "ACTIVE"
    ]
    assert post_timeout_active == [], (
        f"agent 0 was resurrected to ACTIVE at steps {post_timeout_active}"
    )


def test_guesser_timeout_preserved_across_round_boundary():
    """A guesser that errors out in round 1 must not be reactivated for
    round 2's art phase."""

    def fail_guesser(observation, configuration):
        if observation.role == "guesser" and observation.team == "yellow":
            return DeadlineExceeded()
        if observation.role == "artist":
            return observation.target_word
        return observation.teammate_art

    env = _make(num_rounds=2, seed=1)
    env.run([fail_guesser] * 4)
    # Yellow guesser in round 0 is agent 3. Should stay TIMEOUT — must not
    # be flipped to ACTIVE/INACTIVE for the round-1 art phase, and must
    # remain TIMEOUT after the game DONE sweep.
    assert env.state[3].status == "TIMEOUT"


# --- Singular/plural guess leniency ---------------------------------------
#
# The matcher accepts a guess whose plural/singular form matches the target,
# so a model that says CATS when the target is CAT (or vice versa) is not
# punished for English morphology. Direct tests on _matches_target cover the
# tricky irregular forms; the integration test below confirms the runtime
# actually plumbs it through _process_team_guess.


@pytest.mark.parametrize(
    "guess,target",
    [
        ("CAT", "CAT"),           # exact match baseline
        ("cats", "CAT"),          # regular plural, lowercase input
        ("CAT", "CATS"),          # inverse: target is plural, guess is singular
        ("BOXES", "BOX"),         # -ES suffix
        ("BUSES", "BUS"),         # BUS singularizes ambiguously; matcher must still find BUS
        ("LEAVES", "LEAF"),       # -F -> -VES
        ("KNIVES", "KNIFE"),      # -FE -> -VES
        ("BABIES", "BABY"),       # -Y -> -IES
        ("MICE", "MOUSE"),        # irregular
        ("TEETH", "TOOTH"),       # irregular
        ("CHILDREN", "CHILD"),    # irregular
        ("SNOWMEN", "SNOWMAN"),   # compound-suffix inheritance
        ("SHEEP", "SHEEP"),       # same-form noun
        ("CACTI", "CACTUS"),      # Latin plural
        ("CACTUSES", "CACTUS"),   # anglicised alt plural for the same word
        ("ANTENNAE", "ANTENNA"),  # Latin plural with -AE
    ],
)
def test_matches_target_accepts_plural_variants(guess, target):
    assert _matches_target(guess.upper(), target.upper())


@pytest.mark.parametrize(
    "guess,target",
    [
        ("DOG", "CAT"),           # unrelated words
        ("CAT", "CATFISH"),       # target substring is not a plural relation
        ("", "CAT"),              # empty guess never matches
        ("CAT", ""),              # empty target never matches
        ("WROTE", "WRITE"),       # verb tense variants are explicitly out of scope
        ("COUCH", "SOFA"),        # synonyms are out of scope
    ],
)
def test_matches_target_rejects_non_plural_variants(guess, target):
    assert not _matches_target(guess.upper(), target.upper())


def test_plural_guess_scores_via_full_env():
    """End-to-end: a guesser that always answers `<target>S` scores points,
    proving the matcher is wired into _process_team_guess (not just callable).
    """

    def plural_cheater(observation, configuration):
        if observation.role == "artist":
            return _encode_word(observation.target_word)
        return _decode_art(observation.teammate_art) + "S"

    env = _make(num_rounds=3, seed=42)
    env.run([plural_cheater] * 4)
    j = env.toJSON()
    # Some rounds may pick a word whose "+S" isn't a valid plural form of
    # the target (irregulars like MAN/CHILD are blocklisted, but SHEEP-style
    # same-form nouns exist and would reject "SHEEPS"). We only assert
    # SOMETHING scored -- otherwise the matcher plainly isn't wired up.
    assert j["rewards"][0] > 0
    assert j["rewards"][2] > 0


# --- Stratified sampling via word_mix -------------------------------------


def _all_words():
    from kaggle_environments.envs.word_art.word_art import _load_words
    return _load_words()


def test_word_mix_honors_per_tier_counts():
    """A word_mix with 4 easy + 2 hard must produce exactly that
    tier distribution, regardless of shuffle order."""
    env = _make(num_rounds=6, seed=42, word_mix={"easy": 4, "hard": 2})
    env.run([silent] * 4)
    j = env.toJSON()
    history = j["steps"][-1][0]["observation"]["history"]
    words_used = [h["word"] for h in history]
    tier_by_word = {w["word"]: w["tier"] for w in _all_words()}
    tiers = [tier_by_word[w] for w in words_used]
    assert tiers.count("easy") == 4
    assert tiers.count("hard") == 2


def test_word_mix_sum_mismatch_raises():
    """word_mix counts must sum to num_rounds; the check runs at env
    construction (make() calls initialize_game), so the raise happens
    before any agent runs."""
    with pytest.raises(ValueError, match="sum to num_rounds"):
        _make(num_rounds=8, seed=1, word_mix={"easy": 2, "hard": 2})


def test_word_mix_tier_pool_too_small_raises():
    """Requesting more words than a tier's pool holds must raise."""
    rng = __import__("random").Random(0)
    all_words = _all_words()
    hard_pool = sum(1 for w in all_words if w["tier"] == "hard")
    with pytest.raises(ValueError, match="pool has only"):
        _sample_words(all_words, hard_pool + 1, {"hard": hard_pool + 1}, rng)


def test_irregular_tables_stay_trimmed_to_csv():
    """Guard the CSV -> matcher-tables dependency (see word_art.py comment).

    Two failure modes, both surfaced with the specific offending entry:
      (a) A kept entry no longer has any form in the CSV -- drop it.
      (b) The CSV grew to need a dropped entry -- re-add it, or the
          matcher silently rejects the valid plural. (b) is checked
          against a fixed list of high-risk historical entries; expand
          it when dropping new entries from _IRREGULAR_PLURALS.
    """
    from kaggle_environments.envs.word_art.word_art import (
        _COMPOUND_IRREGULAR_SUFFIXES, _IRREGULAR_PLURALS, _SAME_FORM_NOUNS,
    )
    words = {w["word"] for w in _all_words()}

    # (a) every kept entry earns its keep.
    unused_irregulars = [
        sg for sg, pls in _IRREGULAR_PLURALS.items()
        if sg not in words and not any(pl in words for pl in pls)
    ]
    assert not unused_irregulars, (
        f"_IRREGULAR_PLURALS entries unused by words.csv (drop): {unused_irregulars}"
    )
    unused_same_form = [w for w in _SAME_FORM_NOUNS if w not in words]
    assert not unused_same_form, (
        f"_SAME_FORM_NOUNS entries absent from words.csv (drop): {unused_same_form}"
    )
    unused_compounds = [
        (sfx, pl_sfx)
        for sfx, pl_sfx in _COMPOUND_IRREGULAR_SUFFIXES.items()
        if not any(
            (len(w) > len(sfx) and w.endswith(sfx))
            or (len(w) > len(pl_sfx) and w.endswith(pl_sfx))
            for w in words
        )
    ]
    assert not unused_compounds, (
        f"_COMPOUND_IRREGULAR_SUFFIXES unused by any CSV word (drop): {unused_compounds}"
    )

    # (b) high-risk historical entries -- if the CSV grows to include any
    # of these, re-add the mapping to _IRREGULAR_PLURALS or the matcher
    # silently rejects the true plural. Expand when dropping new entries.
    dropped_that_would_break = {
        "MAN": ("MEN",), "WOMAN": ("WOMEN",),  # blocklisted; if unblocked, re-add
        "LIFE": ("LIVES",), "WIFE": ("WIVES",),
        "HALF": ("HALVES",), "SELF": ("SELVES",), "ELF": ("ELVES",),
        "SCARF": ("SCARVES",),
        "MATRIX": ("MATRICES",), "INDEX": ("INDICES",),
        "APPENDIX": ("APPENDICES",), "VERTEX": ("VERTICES",),
        "NUCLEUS": ("NUCLEI",), "RADIUS": ("RADII",),
        "CRITERION": ("CRITERIA",), "PHENOMENON": ("PHENOMENA",),
        "AXIS": ("AXES",), "BASIS": ("BASES",), "CRISIS": ("CRISES",),
        "ANALYSIS": ("ANALYSES",), "HYPOTHESIS": ("HYPOTHESES",),
        "DIAGNOSIS": ("DIAGNOSES",), "SYNOPSIS": ("SYNOPSES",),
        "THESIS": ("THESES",),
        "DATUM": ("DATA",), "MEDIUM": ("MEDIA",),
        "AMOEBA": ("AMOEBAE", "AMOEBAS"), "FORMULA": ("FORMULAE", "FORMULAS"),
    }
    regressions = {
        sg: pls for sg, pls in dropped_that_would_break.items()
        if sg in words or any(pl in words for pl in pls)
    }
    assert not regressions, (
        f"words.csv now contains {list(regressions)}; re-add to "
        f"_IRREGULAR_PLURALS in word_art.py: {regressions}"
    )


def test_default_uniform_sampling_backwards_compatible():
    """Empty word_mix (default) samples uniformly from the full pool -- the
    pre-CSV behaviour is preserved. Just check nothing raises and the round
    count is honoured."""
    env = _make(num_rounds=4, seed=7)
    env.run([silent] * 4)
    j = env.toJSON()
    history = j["steps"][-1][0]["observation"]["history"]
    assert len(history) == 4


# --- guess_points config ---------------------------------------------------


def test_guess_points_default_is_uniform_ones():
    """No guess_points config → every attempt scores 1 pt. lazy_second_try
    wins on attempt 2, so each team ends with 1 pt/round."""
    env = _make(num_rounds=3, seed=9)
    env.run([lazy_second_try] * 4)
    j = env.toJSON()
    assert j["rewards"] == [3, 3, 3, 3]
    # Also confirm history recorded 1 pt per team per round.
    for entry in j["steps"][-1][0]["observation"]["history"]:
        assert entry["blue_points"] == 1
        assert entry["yellow_points"] == 1


def test_guess_points_fractional_rewards():
    """Fractional guess_points produce float rewards end to end."""
    env = _make(num_rounds=2, seed=5, guess_points=[1.5, 0.5, 0.25])
    env.run([cheating] * 4)
    j = env.toJSON()
    # Cheating hits on attempt 1 → 1.5 pts/round * 2 rounds = 3.0.
    for r in j["rewards"]:
        assert r == 3.0
        assert isinstance(r, float)


def test_guess_points_length_mismatch_raises():
    """Non-empty guess_points whose length differs from max_attempts raises
    at env init (before any agent runs), matching the word_mix sum-check
    precedent."""
    with pytest.raises(ValueError, match="length max_attempts"):
        _make(num_rounds=3, guess_points=[1, 2])
    with pytest.raises(ValueError, match="length max_attempts"):
        _make(num_rounds=3, max_attempts=2, guess_points=[3, 2, 1])


def test_guess_points_second_attempt_scoring():
    """guess_points[1] governs the reward when the guesser hits on attempt 2."""
    env = _make(num_rounds=3, seed=6, guess_points=[10, 3, 1])
    env.run([lazy_second_try] * 4)
    j = env.toJSON()
    # 3 pts on attempt 2 * 3 rounds = 9 per agent.
    assert j["rewards"] == [9, 9, 9, 9]


# --- include_art_history config --------------------------------------------


def test_include_art_history_engine_side_is_noop():
    """The engine records history the same way regardless of
    include_art_history — the flag only affects harness prompt rendering.
    We verify that raw art still lives in history[i].blue_art / yellow_art
    even when include_art_history=False, so replays remain complete.
    """
    env = _make(num_rounds=2, seed=1, include_art_history=False, guess_points=[2, 1, 1])
    env.run([cheating, cheating, silent, silent])
    j = env.toJSON()
    history = j["steps"][-1][0]["observation"]["history"]
    assert len(history) == 2
    for entry in history:
        # Blue cheating agent encodes the word; art should be present in
        # history even though the harness would omit it from prompts.
        assert entry["blue_art"] != ""


# --- Homoglyph hardening ----------------------------------------------------


def test_sanitize_drops_non_ascii_letter_likes():
    """Both anti-text checks match [A-Za-z] only, so a Cyrillic or circled
    spelling of the target would render legibly while passing every check.
    Sanitizing them away closes the bypass at the source.
    """
    assert _sanitize_art("САТ") == ""       # Cyrillic CAT
    assert _sanitize_art("ⒸⒶⓉ") == ""       # circled CAT
    assert _sanitize_art("\U0001d402\U0001d400\U0001d413") == ""  # math-bold CAT


def test_sanitize_keeps_non_letter_decoration():
    """The drop must be narrow: box-drawing, blocks, shapes, arrows and
    Braille are the whole point of the non-ASCII allowance."""
    art = "─│┌█░●←★⠿ ½"
    assert _sanitize_art(art) == art


def test_homoglyph_target_word_conveys_nothing():
    def cyrillic_artist(observation, configuration):
        if observation.role != "artist":
            return ""
        cyr = {"A": "А", "C": "С", "E": "Е", "O": "О", "T": "Т"}
        return "".join(cyr.get(c, "А") for c in observation.target_word)

    env = _make(num_rounds=2, seed=3)
    env.run([cyrillic_artist] * 4)
    for entry in env.steps[-1][0].observation.history:
        assert entry["blue_art"] == ""


# --- check_art --------------------------------------------------------------


def test_check_art_names_the_offending_text():
    assert check_art("  O  <-- eye\n /|\\", "CHEEKBONE", 4000) == (
        "contains_words", "the letter run 'eye' (along a row)",
    )
    assert check_art("F..\nL..\nA..\nG..", "ZEBRA", 4000) == (
        "contains_words", "the letter run 'FLAG' (reading down a column)",
    )
    reason, detail = check_art("C A T", "CAT", 4000)
    assert reason == "target_word"
    assert "CAT" in detail


def test_check_art_passes_clean_drawing():
    assert check_art("  *\n \\|/\n--+--\n / \\", "SNOWFLAKE", 4000) is None


def test_check_art_agrees_with_the_interpreter_verdict():
    """check_art exists so a harness can predict the engine. If it applied a
    different sanitize/truncate pass it would reject art the engine accepts
    (wasted retries) or miss art the engine rejects (silent zeros)."""
    art = "x" * 30 + "eye"
    assert check_art(art, "CHEEKBONE", 4000)[0] == "contains_words"
    # Truncated below the offending run -> the engine never sees it either.
    assert check_art(art, "CHEEKBONE", 30) is None


# --- illegalMoveForfeit -----------------------------------------------------


def test_forfeit_submission_reads_as_an_empty_turn():
    """core_harness signals "no parseable answer after all retries" with
    submission=-1. Word Art has no action index, so it must land as a
    forfeited turn rather than art reading '-1'."""
    def forfeiter(observation, configuration):
        # The action shape core_harness returns when illegalMoveForfeit fires.
        return {"submission": -1, "actionString": None, "status": "forfeiting."}

    env = _make(num_rounds=2, seed=5)
    env.run([forfeiter] * 4)
    j = env.toJSON()
    assert [s["status"] for s in j["steps"][-1]] == ["DONE"] * 4
    for entry in j["steps"][-1][0]["observation"]["history"]:
        assert entry["blue_art"] == ""
        assert entry["blue_guesses"] == ["", "", ""]
    assert j["rewards"] == [0, 0, 0, 0]


def test_forfeited_art_is_preserved_and_disqualified():
    """An artist forfeit means the retries all produced rejected art, which
    core_harness hands back as `actionString`. Keep it: the guesser is told
    the drawing was pulled instead of guessing at a blank canvas, and the
    replay shows what was actually drawn."""
    rejected = "  C A T\n /\\_/\\"
    seen = {}

    def agent(observation, configuration):
        if observation.role == "artist":
            return {"submission": -1, "actionString": rejected, "status": "forfeit"}
        seen.setdefault(observation.team, observation.teammate_art)
        return "WHATEVER"

    env = _make(num_rounds=1, seed=5)
    env.run([agent] * 4)
    entry = env.toJSON()["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art"] == rejected
    assert entry["blue_art_disqualified"] is True
    assert "disqualified" in seen["blue"]
