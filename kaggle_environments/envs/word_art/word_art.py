import json
import random
from os import path

dir_path = path.dirname(__file__)


def _load_words():
    words_path = path.abspath(path.join(dir_path, "words.txt"))
    with open(words_path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]


def get_team(agent_idx):
    return "blue" if agent_idx < 2 else "yellow"


def get_role(agent_idx, round_idx):
    """Within each team the artist alternates each round.

    Round r: team blue artist = agent (r % 2); team yellow artist = agent (2 + r % 2).
    The other team member is the guesser. This means every agent is artist on
    half the rounds and guesser on the other half (off by one if num_rounds is odd).
    """
    team_base = 0 if agent_idx < 2 else 2
    artist_idx = team_base + (round_idx % 2)
    return "artist" if agent_idx == artist_idx else "guesser"


def _blue_artist(round_idx):
    return round_idx % 2


def _yellow_artist(round_idx):
    return 2 + (round_idx % 2)


def _blue_guesser(round_idx):
    return 1 - (round_idx % 2)


def _yellow_guesser(round_idx):
    return 2 + (1 - (round_idx % 2))


def _unwrap(action):
    """Harnesses sometimes wrap actions as {'submission': ...}. Unwrap before use."""
    if isinstance(action, dict) and "submission" in action:
        return action["submission"]
    return action


def _coerce_str(value, max_chars):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value[:max_chars]


def _normalize_guess(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip().upper()


# Placeholder shown to the guesser when their teammate's art was disqualified
# for containing the target word. Kept short and unambiguous so a guesser can
# tell it apart from a normal empty submission.
DISQUALIFIED_ART_PLACEHOLDER = "<your teammate's drawing was disqualified for containing the target word>"


def _alnum_lower(s):
    """Strip every non-alphanumeric character and lowercase the rest.

    Used by the no-word-in-art check so that 'C A T', 'C-A-T', 'c.A_t',
    and 'C\\nA\\nT' all collapse to 'cat' before the substring check.
    """
    return "".join(c for c in s.lower() if c.isalnum())


def _art_contains_word(art, word):
    """Return True if `art`, after stripping non-alphanumerics and
    lowercasing, contains `word` (or its reverse) as a substring.

    Catches: verbatim ('CAT'), case variants ('Cat'), letter-by-letter
    with any/no separator ('C A T', 'C-A-T', 'C.A.T', 'C\\nA\\nT'),
    and reversed spellings ('TAC'). Does NOT catch semantically close
    words (NATO alphabet, synonyms, translations) — those require model
    judgement and the prompt asks the artist not to use them.
    """
    if not art or not word:
        return False
    art_norm = _alnum_lower(art)
    word_norm = _alnum_lower(word)
    if not word_norm:
        return False
    return word_norm in art_norm or word_norm[::-1] in art_norm


def _score_for_attempt(attempt_num, first_try_bonus):
    """Points awarded when the guesser hits on attempt `attempt_num` (1-indexed)."""
    base = 1
    return base + (first_try_bonus if attempt_num == 1 else 0)


def initialize_game(state, config):
    seed = config.get("seed")
    rng = random.Random(seed) if seed is not None else random

    num_rounds = config.num_rounds
    max_attempts = config.max_attempts
    first_try_bonus = config.get("first_try_bonus", 1)
    max_art_chars = config.get("max_art_chars", 4000)
    all_words = _load_words()
    if num_rounds > len(all_words):
        raise ValueError(f"num_rounds={num_rounds} exceeds the size of the word list ({len(all_words)}).")
    sampled = rng.sample(all_words, num_rounds)

    for i, s in enumerate(state):
        s.observation.num_rounds = num_rounds
        s.observation.max_attempts = max_attempts
        s.observation.first_try_bonus = first_try_bonus
        s.observation.max_art_chars = max_art_chars
        s.observation.current_round = 0
        s.observation.phase = "art"
        s.observation.role = get_role(i, 0)
        s.observation.team = get_team(i)
        s.observation.target_word = sampled[0] if s.observation.role == "artist" else ""
        s.observation.teammate_art = ""
        s.observation.previous_guesses = []
        s.observation.attempts_remaining = 0
        s.observation.blue_score = 0
        s.observation.yellow_score = 0
        s.observation.blue_attempts_used = 0
        s.observation.yellow_attempts_used = 0
        s.observation.history = []

    # Hidden round-scoped state lives on agent 0's observation: the full word
    # list, in-progress art, accumulated guesses, and per-team done flags.
    # Guessers must not see the words list, and the opposing team must not see
    # in-progress art or guesses, so these fields are kept off the spec and
    # only mirrored to the appropriate agents.
    obs0 = state[0].observation
    obs0._words = sampled
    _reset_round_internals(obs0)


def _reset_round_internals(obs0):
    obs0._round_blue_art = ""
    obs0._round_yellow_art = ""
    obs0._round_blue_art_disqualified = False
    obs0._round_yellow_art_disqualified = False
    obs0._round_blue_guesses = []
    obs0._round_yellow_guesses = []
    obs0._round_blue_done = False
    obs0._round_yellow_done = False
    obs0._round_blue_points = 0
    obs0._round_yellow_points = 0


# Statuses set by the kaggle framework when an agent fails. We must NOT
# overwrite them on phase transitions: a TIMEOUT'd or ERROR'd agent has
# forfeited and should stay in that state for the rest of the episode, so
# the framework stops calling them and the failure remains visible in the
# replay. Without this guard, an artist that times out gets silently
# resurrected as ACTIVE on the next round and times out again.
_TERMINAL_FAILURE_STATUSES = ("TIMEOUT", "ERROR", "INVALID")


def _set_art_statuses(state, round_idx):
    for i in range(4):
        if state[i].status in _TERMINAL_FAILURE_STATUSES:
            continue
        role = get_role(i, round_idx)
        state[i].status = "ACTIVE" if role == "artist" else "INACTIVE"


def _set_guess_statuses(state, round_idx, obs0):
    """During the guess phase: a team's guesser stays ACTIVE until they score
    or exhaust attempts; once done, they go INACTIVE. Artists are always
    INACTIVE in the guess phase. Agents in a terminal failure state
    (TIMEOUT/ERROR/INVALID) are left alone -- see _TERMINAL_FAILURE_STATUSES.
    """
    for i in range(4):
        if state[i].status in _TERMINAL_FAILURE_STATUSES:
            continue
        role = get_role(i, round_idx)
        if role == "artist":
            state[i].status = "INACTIVE"
            continue
        team = get_team(i)
        done = obs0._round_blue_done if team == "blue" else obs0._round_yellow_done
        state[i].status = "INACTIVE" if done else "ACTIVE"


def _process_team_guess(state, obs0, team, env_config, target_norm):
    """Read the active guesser's action, append to the team's guess list, and
    return (points_awarded, is_done_now) — both 0/False if the team was already
    done or inactive this step.
    """
    max_attempts = obs0.max_attempts
    first_try_bonus = env_config.get("first_try_bonus", 1)

    if team == "blue":
        if obs0._round_blue_done:
            return 0, False
        g_idx = _blue_guesser(obs0.current_round)
    else:
        if obs0._round_yellow_done:
            return 0, False
        g_idx = _yellow_guesser(obs0.current_round)

    if state[g_idx].status != "ACTIVE":
        # Guesser was not asked for an action this step (e.g. ERROR/TIMEOUT
        # propagated from a prior step). Treat as out of the round.
        if team == "blue":
            obs0._round_blue_done = True
        else:
            obs0._round_yellow_done = True
        return 0, True

    raw = _unwrap(state[g_idx].action)
    guess_str = raw if isinstance(raw, str) else (str(raw) if raw is not None else "")
    if team == "blue":
        obs0._round_blue_guesses.append(guess_str)
        used = len(obs0._round_blue_guesses)
    else:
        obs0._round_yellow_guesses.append(guess_str)
        used = len(obs0._round_yellow_guesses)

    guess_norm = _normalize_guess(raw)
    if guess_norm == target_norm and guess_norm != "":
        pts = _score_for_attempt(used, first_try_bonus)
        if team == "blue":
            obs0._round_blue_points = pts
            obs0._round_blue_done = True
            state[0].reward = (state[0].reward or 0) + pts
            state[1].reward = (state[1].reward or 0) + pts
        else:
            obs0._round_yellow_points = pts
            obs0._round_yellow_done = True
            state[2].reward = (state[2].reward or 0) + pts
            state[3].reward = (state[3].reward or 0) + pts
        return pts, True

    if used >= max_attempts:
        if team == "blue":
            obs0._round_blue_done = True
        else:
            obs0._round_yellow_done = True
        return 0, True

    return 0, False


def _enter_guess_phase(state, obs0, round_idx, blue_art, yellow_art):
    """Mutate every agent's observation for the start of the guess phase and
    activate both guessers.
    """
    max_attempts = obs0.max_attempts
    for i, s in enumerate(state):
        s.observation.phase = "guess"
        s.observation.target_word = ""
        s.observation.blue_attempts_used = 0
        s.observation.yellow_attempts_used = 0
        if get_role(i, round_idx) == "guesser":
            team = get_team(i)
            s.observation.teammate_art = blue_art if team == "blue" else yellow_art
            s.observation.attempts_remaining = max_attempts
            s.observation.previous_guesses = []
        else:
            s.observation.teammate_art = ""
            s.observation.attempts_remaining = 0
            s.observation.previous_guesses = []
    _set_guess_statuses(state, round_idx, obs0)


def _advance_after_round(state, obs0, round_idx, words, target):
    """Roll the per-team round state into history and advance to the next
    round's art phase (or finish the game).
    """
    new_blue_score = obs0.blue_score + obs0._round_blue_points
    new_yellow_score = obs0.yellow_score + obs0._round_yellow_points

    history_entry = {
        "word": target,
        "blue_art": obs0._round_blue_art,
        "blue_art_disqualified": obs0._round_blue_art_disqualified,
        "blue_guesses": list(obs0._round_blue_guesses),
        "blue_points": obs0._round_blue_points,
        "yellow_art": obs0._round_yellow_art,
        "yellow_art_disqualified": obs0._round_yellow_art_disqualified,
        "yellow_guesses": list(obs0._round_yellow_guesses),
        "yellow_points": obs0._round_yellow_points,
    }
    new_history = list(obs0.history) + [history_entry]

    next_round = round_idx + 1
    is_done = next_round >= obs0.num_rounds

    for i, s in enumerate(state):
        s.observation.blue_score = new_blue_score
        s.observation.yellow_score = new_yellow_score
        s.observation.history = new_history
        s.observation.teammate_art = ""
        s.observation.target_word = ""
        s.observation.previous_guesses = []
        s.observation.attempts_remaining = 0
        s.observation.blue_attempts_used = 0
        s.observation.yellow_attempts_used = 0
        if not is_done:
            s.observation.current_round = next_round
            s.observation.phase = "art"
            s.observation.role = get_role(i, next_round)
            if s.observation.role == "artist":
                s.observation.target_word = words[next_round]

    _reset_round_internals(obs0)

    if is_done:
        for i in range(4):
            if state[i].status in _TERMINAL_FAILURE_STATUSES:
                continue
            state[i].status = "DONE"
    else:
        _set_art_statuses(state, next_round)


def process_step(state, env):
    obs0 = state[0].observation
    phase = obs0.phase
    rnd = obs0.current_round
    words = obs0._words
    target = words[rnd]

    if phase == "art":
        max_chars = env.configuration.get("max_art_chars", 4000)
        enforce_no_word = env.configuration.get("enforce_no_word_in_art", True)
        blue_action = _unwrap(state[_blue_artist(rnd)].action)
        yellow_action = _unwrap(state[_yellow_artist(rnd)].action)
        blue_art = _coerce_str(blue_action, max_chars)
        yellow_art = _coerce_str(yellow_action, max_chars)

        # The artist's RAW submission is preserved in obs0._round_*_art (and
        # then in history) so the replay shows what they tried. What the
        # guesser sees may be replaced by a placeholder if the art smuggles
        # the target word in. Per-team disqualification flags drive both the
        # guesser's view and the history-entry annotation.
        obs0._round_blue_art = blue_art
        obs0._round_yellow_art = yellow_art
        obs0._round_blue_art_disqualified = enforce_no_word and _art_contains_word(blue_art, target)
        obs0._round_yellow_art_disqualified = enforce_no_word and _art_contains_word(yellow_art, target)

        blue_art_for_guesser = DISQUALIFIED_ART_PLACEHOLDER if obs0._round_blue_art_disqualified else blue_art
        yellow_art_for_guesser = DISQUALIFIED_ART_PLACEHOLDER if obs0._round_yellow_art_disqualified else yellow_art
        _enter_guess_phase(state, obs0, rnd, blue_art_for_guesser, yellow_art_for_guesser)
        return

    # phase == "guess" — a single sub-step. Each still-active guesser
    # contributes one attempt.
    target_norm = target.strip().upper()
    _process_team_guess(state, obs0, "blue", env.configuration, target_norm)
    _process_team_guess(state, obs0, "yellow", env.configuration, target_norm)

    blue_used = len(obs0._round_blue_guesses)
    yellow_used = len(obs0._round_yellow_guesses)
    max_attempts = obs0.max_attempts

    if obs0._round_blue_done and obs0._round_yellow_done:
        _advance_after_round(state, obs0, rnd, words, target)
        return

    # Round still in progress: update per-team counters on every agent's view
    # (these are public; both teams can see how many guesses the other has
    # used), and update each guesser's private `attempts_remaining` and
    # `previous_guesses` lists.
    blue_g_idx = _blue_guesser(rnd)
    yellow_g_idx = _yellow_guesser(rnd)
    for s in state:
        s.observation.blue_attempts_used = blue_used
        s.observation.yellow_attempts_used = yellow_used

    if not obs0._round_blue_done:
        state[blue_g_idx].observation.attempts_remaining = max_attempts - blue_used
        state[blue_g_idx].observation.previous_guesses = list(obs0._round_blue_guesses)
    else:
        state[blue_g_idx].observation.attempts_remaining = 0
        state[blue_g_idx].observation.previous_guesses = list(obs0._round_blue_guesses)

    if not obs0._round_yellow_done:
        state[yellow_g_idx].observation.attempts_remaining = max_attempts - yellow_used
        state[yellow_g_idx].observation.previous_guesses = list(obs0._round_yellow_guesses)
    else:
        state[yellow_g_idx].observation.attempts_remaining = 0
        state[yellow_g_idx].observation.previous_guesses = list(obs0._round_yellow_guesses)

    _set_guess_statuses(state, rnd, obs0)


def interpreter(state, env):
    if state[0].observation.phase == "":
        initialize_game(state, env.configuration)
        _set_art_statuses(state, 0)
        return state

    if env.done:
        return state

    process_step(state, env)
    return state


def renderer(state, env):
    obs = state[0].observation
    lines = []
    lines.append(f"Round {obs.current_round + 1}/{obs.num_rounds} -- phase: {obs.phase}")
    lines.append(f"Score: blue={obs.blue_score} yellow={obs.yellow_score}")

    for h in obs.history:
        b_guesses = ", ".join(repr(g) for g in h.get("blue_guesses", []))
        y_guesses = ", ".join(repr(g) for g in h.get("yellow_guesses", []))
        lines.append(
            f"  [{h['word']}]"
            f" blue=[{b_guesses}] +{h.get('blue_points', 0)}"
            f"  yellow=[{y_guesses}] +{h.get('yellow_points', 0)}"
        )

    if obs.phase == "art" and obs.current_round < obs.num_rounds:
        lines.append("Artists drawing word: <hidden>")
    elif obs.phase == "guess":
        lines.append(
            "Guessers viewing teammate's art "
            f"(blue used {obs.blue_attempts_used}/{obs.max_attempts}, "
            f"yellow used {obs.yellow_attempts_used}/{obs.max_attempts})."
        )

    return "\n".join(lines) + "\n"


json_path = path.abspath(path.join(dir_path, "word_art.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    """Reads the built web visualizer output and serves it for rendering."""
    jspath = path.join(dir_path, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""


from .agents import agents  # noqa: E402, F401
