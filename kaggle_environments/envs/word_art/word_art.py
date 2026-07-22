import csv
import json
import random
import re
import unicodedata
from os import path

dir_path = path.dirname(__file__)


def _load_words():
    """Load words.csv. Each entry is {word, category, tier, source}.

    Category ('noun'/'verb'/'abstract') and tier ('easy'/'medium'/'hard')
    are metadata used only by _sample_words for stratified episode
    composition; the runtime word channel exposes just the string.
    """
    words_path = path.abspath(path.join(dir_path, "words.csv"))
    with open(words_path, newline="") as f:
        return list(csv.DictReader(f))


def _sample_words(all_words, num_rounds, word_mix, rng):
    """Return `num_rounds` uppercased target words for an episode.

    - If `word_mix` is empty/falsy: uniform sample from the full pool
      (backwards-compatible with the pre-CSV behaviour).
    - Otherwise `word_mix` must be a mapping {tier_name: int_count} whose
      counts sum to `num_rounds`. Each tier's slice is sampled from the
      corresponding sub-pool without replacement; the result is then
      shuffled so tier order within the episode is randomised (otherwise
      a strong model could infer difficulty from round index).
    """
    if not word_mix:
        if num_rounds > len(all_words):
            raise ValueError(
                f"num_rounds={num_rounds} exceeds the size of the word list ({len(all_words)})."
            )
        return [w["word"] for w in rng.sample(all_words, num_rounds)]

    total = sum(word_mix.values())
    if total != num_rounds:
        raise ValueError(
            f"word_mix counts must sum to num_rounds={num_rounds}, got {total} (word_mix={word_mix})."
        )
    picked = []
    for tier, count in word_mix.items():
        pool = [w for w in all_words if w["tier"] == tier]
        if count > len(pool):
            raise ValueError(
                f"word_mix requests {count} words from tier '{tier}' but the pool has only {len(pool)}."
            )
        picked.extend(rng.sample(pool, count))
    rng.shuffle(picked)
    return [w["word"] for w in picked]


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


def _sanitize_art(s):
    """Drop characters that break monospace alignment.

    Keeps: printable ASCII, `\\n`/`\\t`, and any Unicode character whose
    East Asian Width is single-cell (Na/N/H/A) — box-drawing (─│┌┐),
    blocks (▀█░▒), geometric shapes (●○▲), arrows (←→), common symbols
    (★♥), Braille, etc.
    Drops: control/format/surrogate/private chars (Cc/Cf/Cs/Co/Cn),
    combining marks (Mn/Mc/Me), and wide/fullwidth glyphs
    (CJK ideographs, most emoji) — anything that would shift subsequent
    characters in a monospace grid and desync the guesser's view.
    """
    def _keep(ch):
        if ch in "\n\t":
            return True
        cat = unicodedata.category(ch)
        if cat[0] in ("C", "M"):
            return False
        return unicodedata.east_asian_width(ch) in ("Na", "N", "H", "A")
    return "".join(c for c in s if _keep(c))


def _coerce_str(value, max_chars):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    # Sanitize BEFORE truncating so the max_chars budget reflects
    # what actually reaches the guesser.
    return _sanitize_art(value)[:max_chars]


def _normalize_guess(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip().upper()


# --- Singular/plural leniency for guess matching ----------------------------
#
# Accepts CAT<->CATS in either direction so we test the game, not English
# morphology. Out of scope: synonyms, tenses, spelling variants.
#
# CSV DEPENDENCY: the three tables below are scoped to entries that apply
# to the current words.csv, not a general English reference. If the CSV
# changes, entries may need to be added (matcher will silently reject
# valid plurals) or removed (dead code). The guard test
# `test_irregular_tables_stay_trimmed_to_csv` names the exact diffs.

# Singular -> tuple of accepted plurals (tuple because ANTENNA accepts both
# ANTENNAE and ANTENNAS). _IRREGULAR_SINGULARS is the reverse map.
_IRREGULAR_PLURALS = {
    "CHILD": ("CHILDREN",), "FOOT": ("FEET",), "TOOTH": ("TEETH",),
    "MOUSE": ("MICE",), "GOOSE": ("GEESE",), "PERSON": ("PEOPLE",),
    "OX": ("OXEN",),
    "CACTUS": ("CACTI", "CACTUSES"),
    "FUNGUS": ("FUNGI", "FUNGUSES"),
    "OCTOPUS": ("OCTOPI", "OCTOPUSES"),
    "OASIS": ("OASES",),
    "KNIFE": ("KNIVES",), "LEAF": ("LEAVES",), "LOAF": ("LOAVES",),
    "WOLF": ("WOLVES",), "CALF": ("CALVES",), "SHELF": ("SHELVES",),
    "THIEF": ("THIEVES",),
    "ALGA": ("ALGAE", "ALGAS"),
    "LARVA": ("LARVAE", "LARVAS"),
    "ANTENNA": ("ANTENNAE", "ANTENNAS"),
    "LOUSE": ("LICE",),
    "QUIZ": ("QUIZZES",),  # CVC-doubling special case not handled by rules
}
_IRREGULAR_SINGULARS = {pl: sg for sg, plurals in _IRREGULAR_PLURALS.items() for pl in plurals}

# Compound suffixes that inherit the irregular pattern: SNOWMAN <-> SNOWMEN,
# STEPCHILD <-> STEPCHILDREN, CHAIRPERSON <-> CHAIRPEOPLE. Known misfire:
# MAN -> MEN wrongly triggers on HUMAN/OTTOMAN/SPECIMEN, but the realistic
# +S guess still matches via the -S fallback in _singularize.
_COMPOUND_IRREGULAR_SUFFIXES = {
    "MAN": "MEN", "CHILD": "CHILDREN",
    "GOOSE": "GEESE", "PERSON": "PEOPLE",
}

# Nouns whose plural form is identical to the singular.
_SAME_FORM_NOUNS = {
    "SHEEP", "DEER", "FISH", "SALMON", "TROUT", "BISON", "ELK",
}

# S-suffixes that are NOT plural markers -- keeps _singularize from stripping
# GAS/BUS/BASIS/CACTUS/ATLAS/ACTRESS/GLASS/MOSS. Property of English, not
# the CSV.
_S_KEEP_SUFFIXES = ("SS", "US", "IS", "OS", "AS")


def _pluralize(w):
    """Return plausible plural forms. Set-valued because -F is genuinely
    ambiguous (ROOFS but WOLVES); caller checks membership."""
    if not w:
        return set()
    if w in _SAME_FORM_NOUNS:
        return {w}
    if w in _IRREGULAR_PLURALS:
        return set(_IRREGULAR_PLURALS[w])
    # Compound-suffix irregular: SNOWMAN -> SNOWMEN, STEPCHILD -> STEPCHILDREN
    for sfx, plural_sfx in _COMPOUND_IRREGULAR_SUFFIXES.items():
        if len(w) > len(sfx) and w.endswith(sfx):
            return {w[:-len(sfx)] + plural_sfx}
    if len(w) < 2:
        return {w + "S"}
    if w.endswith(("S", "X", "Z", "CH", "SH")):
        return {w + "ES"}
    if w.endswith("Y") and w[-2] not in "AEIOU":
        return {w[:-1] + "IES"}
    if w.endswith("FE"):
        return {w[:-2] + "VES"}
    if w.endswith("F"):
        return {w[:-1] + "VES", w + "S"}
    return {w + "S"}


def _singularize(w):
    """Return plausible singular forms. Set-valued because e.g. BUSES could
    strip to BUS or BUSE; caller checks membership. Returns {w} if `w`
    doesn't look plural."""
    if not w or len(w) < 2:
        return {w} if w else set()
    if w in _SAME_FORM_NOUNS:
        return {w}
    if w in _IRREGULAR_SINGULARS:
        return {_IRREGULAR_SINGULARS[w]}
    # Compound-suffix irregular: SNOWMEN -> SNOWMAN, BUCKTEETH -> BUCKTOOTH
    for sing_sfx, plural_sfx in _COMPOUND_IRREGULAR_SUFFIXES.items():
        if len(w) > len(plural_sfx) and w.endswith(plural_sfx):
            return {w[:-len(plural_sfx)] + sing_sfx}
    candidates = set()
    if w.endswith("VES") and len(w) > 3:
        candidates.add(w[:-3] + "F")
        candidates.add(w[:-3] + "FE")
    if w.endswith("IES") and len(w) > 3:
        candidates.add(w[:-3] + "Y")
    if w.endswith("ES") and len(w) > 2:
        # Both BOXES->BOX (drop ES) and HOUSES->HOUSE (drop just S).
        candidates.add(w[:-2])
        candidates.add(w[:-1])
    if (
        w.endswith("S")
        and len(w) > 1
        and not any(w.endswith(sfx) for sfx in _S_KEEP_SUFFIXES)
    ):
        candidates.add(w[:-1])
    return candidates or {w}


def _matches_target(guess, target):
    """Accept guess if it equals target or is a plausible plural/singular
    of it in either direction. Inputs must be pre-normalised (strip+upper).
    Synonyms/tenses/spelling variants are out of scope -- the build-time
    filters keep those out of the word pool."""
    if not guess or not target:
        return False
    if guess == target:
        return True
    if guess in _pluralize(target) or target in _pluralize(guess):
        return True
    if guess in _singularize(target) or target in _singularize(guess):
        return True
    return False


# Guesser-visible placeholders when their teammate's art was disqualified.
# One variant per reason so the guesser can distinguish "artist named the
# word" from "artist added a label"; both are unambiguously non-empty.
DISQUALIFIED_ART_PLACEHOLDERS = {
    "target_word": (
        "<your teammate's drawing was disqualified for containing the target word>"
    ),
    "contains_words": (
        "<your teammate's drawing was disqualified for containing text "
        "(a run of 3+ letters with 2+ distinct chars, consecutive or "
        "separated by any non-letter, non-newline characters)>"
    ),
}


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
    and reversed spellings ('TAC').
    """
    if not art or not word:
        return False
    art_norm = _alnum_lower(art)
    word_norm = _alnum_lower(word)
    if not word_norm:
        return False
    return word_norm in art_norm or word_norm[::-1] in art_norm


_WORD_LIKE_RE = re.compile(r"[A-Za-z]{3,}")
# Spaced-out labels: 3+ letters interleaved with same-line separators
# (any run of non-letter, non-newline characters). Excludes newlines so
# letters in different rows of the art don't chain across a 2D layout
# into fake words. Digits are treated as separators too, so 'A1B2C' is
# caught. Any non-letter, non-newline joiner works so models can't evade
# with pipes ('H|O|U|S|E'), slashes, colons, commas, em-dashes, etc.
_SPACED_WORD_RE = re.compile(r"[A-Za-z](?:[^A-Za-z\n]+[A-Za-z]){2,}")


def _art_contains_any_word(art):
    """Return True if `art` contains any run of 3+ letters with 2+
    distinct characters (case-insensitive), whether the letters are
    consecutive ('TOP', 'HOUSE') or separated by any non-letter,
    non-newline characters ('T O P', 'A.R.O.U.N.D', 'H-O-U-S-E',
    'H|O|U|S|E', 'grid_view').

    Same-letter clusters ('OOO' for eyes, 'III' for columns, 'TTT' for
    texture, 'V V V' for a zigzag) pass so models can still use letters
    as visual elements. Complementary to _art_contains_word; that catches
    only the target word, this catches every OTHER word.

    The distinct-character count considers LETTERS only (not the
    separators between them), so a decorative row like 'V V V' evaluates
    as one distinct letter and passes even though the raw match string
    contains both 'v' and ' '.
    """
    if not art:
        return False
    for regex in (_WORD_LIKE_RE, _SPACED_WORD_RE):
        for m in regex.finditer(art):
            distinct_letters = {c for c in m.group(0).lower() if c.isalpha()}
            if len(distinct_letters) > 1:
                return True
    return False


def _disqualification_reason(art, target):
    """Return why this art submission is disqualified, or None if it passes.

    Priority order: target-word check first (more specific message for the
    artist to learn from), then the general any-word check. None means the
    art reached the guesser unmodified.
    """
    if not art:
        return None
    if _art_contains_word(art, target):
        return "target_word"
    if _art_contains_any_word(art):
        return "contains_words"
    return None


def _score_for_attempt(attempt_num, first_try_bonus):
    """Points awarded when the guesser hits on attempt `attempt_num` (1-indexed)."""
    base = 1
    return base + (first_try_bonus if attempt_num == 1 else 0)


class _WordArtState:
    """Round-scoped hidden state, kept on ``env`` -- never on a player's
    observation.

    Storing the word list or in-progress guesses on ``state[i].observation``
    (even under an underscore-prefixed key) leaks them into the replay JSON
    and into whatever the agent process receives; a custom agent could then
    short-circuit the art channel by reading the target word directly, and
    any future prompt change that dumps the raw obs would silently ship
    hidden fields to the LLM. Following the werewolf env's pattern, the
    interpreter is passed ``env`` on every call, so authoritative state
    lives there and each player's observation is rebuilt each step
    containing only public + role-appropriate fields.
    """

    def __init__(self, words):
        self.words = words
        self.reset_round()

    def reset_round(self):
        self.blue_art = ""
        self.yellow_art = ""
        self.blue_art_disqualified = False
        self.yellow_art_disqualified = False
        self.blue_disq_reason = None
        self.yellow_disq_reason = None
        self.blue_guesses = []
        self.yellow_guesses = []
        self.blue_done = False
        self.yellow_done = False
        self.blue_points = 0
        self.yellow_points = 0


def initialize_game(state, env):
    config = env.configuration
    seed = config.get("seed")
    rng = random.Random(seed) if seed is not None else random

    num_rounds = config.num_rounds
    max_attempts = config.max_attempts
    first_try_bonus = config.get("first_try_bonus", 1)
    max_art_chars = config.get("max_art_chars", 4000)
    word_mix = config.get("word_mix") or {}
    all_words = _load_words()
    sampled = _sample_words(all_words, num_rounds, word_mix, rng)

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

    env.word_art_state = _WordArtState(sampled)


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


def _set_guess_statuses(state, round_idx, wa_state):
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
        done = wa_state.blue_done if team == "blue" else wa_state.yellow_done
        state[i].status = "INACTIVE" if done else "ACTIVE"


def _process_team_guess(state, obs0, wa_state, team, env_config, target_norm):
    """Read the active guesser's action for `team` and mutate wa_state
    (append to the team's guess list, mark done, award points). No-op if
    the team was already done or its guesser wasn't asked to act this step.
    """
    max_attempts = obs0.max_attempts
    first_try_bonus = env_config.get("first_try_bonus", 1)

    if team == "blue":
        if wa_state.blue_done:
            return
        g_idx = _blue_guesser(obs0.current_round)
    else:
        if wa_state.yellow_done:
            return
        g_idx = _yellow_guesser(obs0.current_round)

    if state[g_idx].status != "ACTIVE":
        # Guesser was not asked for an action this step (e.g. ERROR/TIMEOUT
        # propagated from a prior step). Treat as out of the round.
        if team == "blue":
            wa_state.blue_done = True
        else:
            wa_state.yellow_done = True
        return

    raw = _unwrap(state[g_idx].action)
    guess_str = raw if isinstance(raw, str) else (str(raw) if raw is not None else "")
    if team == "blue":
        wa_state.blue_guesses.append(guess_str)
        used = len(wa_state.blue_guesses)
    else:
        wa_state.yellow_guesses.append(guess_str)
        used = len(wa_state.yellow_guesses)

    guess_norm = _normalize_guess(raw)
    if _matches_target(guess_norm, target_norm):
        pts = _score_for_attempt(used, first_try_bonus)
        if team == "blue":
            wa_state.blue_points = pts
            wa_state.blue_done = True
            state[0].reward = (state[0].reward or 0) + pts
            state[1].reward = (state[1].reward or 0) + pts
        else:
            wa_state.yellow_points = pts
            wa_state.yellow_done = True
            state[2].reward = (state[2].reward or 0) + pts
            state[3].reward = (state[3].reward or 0) + pts
        return

    if used >= max_attempts:
        if team == "blue":
            wa_state.blue_done = True
        else:
            wa_state.yellow_done = True


def _enter_guess_phase(state, wa_state, round_idx, blue_art, yellow_art, max_attempts):
    """Mutate every agent's observation for the start of the guess phase and
    activate both guessers.
    """
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
    _set_guess_statuses(state, round_idx, wa_state)


def _advance_after_round(state, obs0, wa_state, round_idx, words, target):
    """Roll the per-team round state into history and advance to the next
    round's art phase (or finish the game).
    """
    new_blue_score = obs0.blue_score + wa_state.blue_points
    new_yellow_score = obs0.yellow_score + wa_state.yellow_points

    history_entry = {
        "word": target,
        "blue_art": wa_state.blue_art,
        "blue_art_disqualified": wa_state.blue_art_disqualified,
        "blue_art_disqualification_reason": wa_state.blue_disq_reason,
        "blue_guesses": list(wa_state.blue_guesses),
        "blue_points": wa_state.blue_points,
        "yellow_art": wa_state.yellow_art,
        "yellow_art_disqualified": wa_state.yellow_art_disqualified,
        "yellow_art_disqualification_reason": wa_state.yellow_disq_reason,
        "yellow_guesses": list(wa_state.yellow_guesses),
        "yellow_points": wa_state.yellow_points,
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

    wa_state.reset_round()

    if is_done:
        for i in range(4):
            if state[i].status in _TERMINAL_FAILURE_STATUSES:
                continue
            state[i].status = "DONE"
    else:
        _set_art_statuses(state, next_round)


def process_step(state, env):
    obs0 = state[0].observation
    if not hasattr(env, "word_art_state"):
        # env.word_art_state lives on the Environment instance and is
        # not serialized into steps, so it's lost across env.clone() and
        # JSON round-trips. Mid-game recovery isn't possible: the target
        # words for future rounds were sampled from an RNG that's now
        # gone. Fail loudly rather than silently corrupting the episode.
        raise RuntimeError(
            "word_art: env.word_art_state is missing. Round-scoped hidden "
            "state does not survive env.clone() or JSON round-trips -- run "
            "an episode start-to-finish in a single Environment instance."
        )
    wa_state: _WordArtState = env.word_art_state
    phase = obs0.phase
    rnd = obs0.current_round
    words = wa_state.words
    target = words[rnd]

    if phase == "art":
        max_chars = env.configuration.get("max_art_chars", 4000)
        blue_action = _unwrap(state[_blue_artist(rnd)].action)
        yellow_action = _unwrap(state[_yellow_artist(rnd)].action)
        blue_art = _coerce_str(blue_action, max_chars)
        yellow_art = _coerce_str(yellow_action, max_chars)

        # The artist's RAW (already-sanitized-and-truncated) submission is
        # preserved in wa_state.*_art — and then in history — so the replay
        # shows what actually reached the engine. What the guesser sees may
        # be replaced by a placeholder if the art smuggles the target word
        # in or contains text-like letter clusters. Per-team disqualification
        # reason drives both the guesser's placeholder variant and the
        # history-entry annotation; the boolean is kept alongside so the
        # visualizer's disqualified-label check keeps working without
        # needing to know about reasons.
        wa_state.blue_art = blue_art
        wa_state.yellow_art = yellow_art
        wa_state.blue_disq_reason = _disqualification_reason(blue_art, target)
        wa_state.yellow_disq_reason = _disqualification_reason(yellow_art, target)
        wa_state.blue_art_disqualified = wa_state.blue_disq_reason is not None
        wa_state.yellow_art_disqualified = wa_state.yellow_disq_reason is not None

        blue_art_for_guesser = (
            DISQUALIFIED_ART_PLACEHOLDERS[wa_state.blue_disq_reason]
            if wa_state.blue_disq_reason else blue_art
        )
        yellow_art_for_guesser = (
            DISQUALIFIED_ART_PLACEHOLDERS[wa_state.yellow_disq_reason]
            if wa_state.yellow_disq_reason else yellow_art
        )
        _enter_guess_phase(
            state, wa_state, rnd, blue_art_for_guesser, yellow_art_for_guesser,
            obs0.max_attempts,
        )
        return

    # phase == "guess" — a single sub-step. Each still-active guesser
    # contributes one attempt.
    target_norm = target.strip().upper()
    _process_team_guess(state, obs0, wa_state, "blue", env.configuration, target_norm)
    _process_team_guess(state, obs0, wa_state, "yellow", env.configuration, target_norm)

    if wa_state.blue_done and wa_state.yellow_done:
        _advance_after_round(state, obs0, wa_state, rnd, words, target)
        return

    # Round still in progress: update per-team counters on every agent's view
    # (these are public; both teams can see how many guesses the other has
    # used), and update each guesser's private `attempts_remaining` and
    # `previous_guesses` lists. A done guesser's `attempts_remaining` goes
    # to 0; `previous_guesses` is refreshed either way.
    blue_used = len(wa_state.blue_guesses)
    yellow_used = len(wa_state.yellow_guesses)
    max_attempts = obs0.max_attempts
    blue_g_idx = _blue_guesser(rnd)
    yellow_g_idx = _yellow_guesser(rnd)
    for s in state:
        s.observation.blue_attempts_used = blue_used
        s.observation.yellow_attempts_used = yellow_used

    state[blue_g_idx].observation.previous_guesses = list(wa_state.blue_guesses)
    state[blue_g_idx].observation.attempts_remaining = (
        0 if wa_state.blue_done else max_attempts - blue_used
    )
    state[yellow_g_idx].observation.previous_guesses = list(wa_state.yellow_guesses)
    state[yellow_g_idx].observation.attempts_remaining = (
        0 if wa_state.yellow_done else max_attempts - yellow_used
    )

    _set_guess_statuses(state, rnd, wa_state)


def interpreter(state, env):
    if state[0].observation.phase == "":
        initialize_game(state, env)
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
