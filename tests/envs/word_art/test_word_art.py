from kaggle_environments import make


def _make(**config):
    """Convenience: tests that exercise scoring/mechanics use the verbatim
    `cheating` agent, which the env's no-word-in-art enforcement would
    disqualify. Default these tests to ``enforce_no_word_in_art=False`` so
    they stay focused on the mechanic they're testing. The dedicated
    disqualification tests below override this.
    """
    config.setdefault("enforce_no_word_in_art", False)
    return make("word_art", configuration=config)


def silent(observation, configuration):
    return ""


def cheating(observation, configuration):
    """Artist sends the word verbatim; guesser parses it back on the first try."""
    if observation.role == "artist":
        return observation.target_word
    return observation.teammate_art


def lazy_second_try(observation, configuration):
    """Guesses 'NOPE' on the first attempt, then the correct word on the second."""
    if observation.role == "artist":
        return observation.target_word
    if not observation.previous_guesses:
        return "NOPE"
    return observation.teammate_art


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


def test_first_try_bonus():
    """Cheating both teams = both score 2 per round (1 base + 1 first-try bonus)."""
    env = _make(num_rounds=4, seed=1)
    env.run([cheating, cheating, cheating, cheating])
    j = env.toJSON()
    assert j["rewards"] == [8, 8, 8, 8]  # 2 points * 4 rounds


def test_second_try_scores_one():
    """A team that always lands on attempt 2 scores 1 per round (no bonus)."""
    env = _make(num_rounds=3, seed=9)
    env.run([lazy_second_try, lazy_second_try, lazy_second_try, lazy_second_try])
    j = env.toJSON()
    assert j["rewards"] == [3, 3, 3, 3]


def test_blue_first_yellow_misses():
    """Blue cheats (2 pts/round); yellow always wrong (0 pts/round)."""
    env = _make(num_rounds=4, seed=1)
    env.run([cheating, cheating, silent, silent])
    j = env.toJSON()
    assert j["rewards"][0] == 8
    assert j["rewards"][1] == 8
    assert j["rewards"][2] == 0
    assert j["rewards"][3] == 0


def test_history_shape_and_attempts():
    """History entries record per-team guess lists and points; the team that
    succeeds first has fewer entries than the team that exhausts all attempts."""
    env = _make(num_rounds=2, seed=1)
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
            return observation.target_word
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
            return observation.target_word
        return observation.teammate_art.lower()

    env = _make(num_rounds=2, seed=17)
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


def test_configurable_max_attempts_and_bonus():
    """Override max_attempts=2 and first_try_bonus=4. lazy_second_try wins on
    attempt 2 → base 1 point, no bonus → 1 per round.
    """
    env = _make(num_rounds=2, seed=6, max_attempts=2, first_try_bonus=4)
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
# These tests use the env's DEFAULT configuration (enforce_no_word_in_art=True)
# rather than the _make() helper, since the helper turns enforcement off.


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


def test_enforcement_can_be_disabled():
    """With enforce_no_word_in_art=False, the original verbatim cheating
    strategy works and scores 2 points."""
    env = make(
        "word_art",
        configuration={"num_rounds": 1, "seed": 5, "enforce_no_word_in_art": False},
    )
    env.run([cheating] * 4)
    j = env.toJSON()
    entry = j["steps"][-1][0]["observation"]["history"][0]
    assert entry["blue_art_disqualified"] is False
    assert entry["blue_points"] == 2


def test_disqualification_does_not_block_guessing():
    """Even with a disqualified art panel, the guesser still gets their full
    attempt budget and can still score if they correctly guess the word."""

    def smart_guesser(observation, configuration):
        if observation.role == "artist":
            return observation.target_word  # will be disqualified
        # Guesser ignores the placeholder, makes a smart guess using history.
        # On round 1, we cheat by reading target via a side channel: pull from
        # the global state via the env's hidden _words on agent 0. Not possible
        # via observation alone, so just guess randomly here -- the test only
        # checks the structure (3 attempts allowed).
        return f"WRONG{len(observation.previous_guesses)}"

    env = make("word_art", configuration={"num_rounds": 1, "seed": 7})
    env.run([smart_guesser] * 4)
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
