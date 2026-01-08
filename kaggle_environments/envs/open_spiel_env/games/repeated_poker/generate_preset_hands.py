#!/usr/bin/env python3
"""Utility for generating preset hand sequences for repeated poker."""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import random
from typing import Iterable

DEFAULT_DECK_SIZE = 52
DEFAULT_NUM_HANDS = 100
DEFAULT_CARDS_PER_HAND = 9

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used to sample cards.",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=DEFAULT_NUM_HANDS,
        help="Number of hands to generate for each preset group.",
    )
    parser.add_argument(
        "--cards-per-hand",
        type=int,
        default=DEFAULT_CARDS_PER_HAND,
        help="Number of card chance actions to sample for each hand.",
    )
    parser.add_argument(
        "--num-presets",
        type=int,
        default=1,
        help="Number of preset hand groups to emit.",
    )
    parser.add_argument(
        "--deck-size",
        type=int,
        default=DEFAULT_DECK_SIZE,
        help="Size of the deck to sample from (expected 52 for standard hold'em).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("preset_hands.jsonl"),
        help="Output JSONL path. Defaults to preset_hands.jsonl in the same directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip confirming the OpenSpiel chance action range (pyspiel required).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    return args


def confirm_chance_action_range(deck_size: int) -> None:
    """Confirm that OpenSpiel emits chance actions in [0, deck_size)."""
    try:
        import pyspiel  # type: ignore
    except ImportError:  # pragma: no cover - pyspiel unavailable in many dev envs.
        LOGGER.warning("pyspiel not installed; skipping chance action verification.")
        return

    game = pyspiel.load_game("repeated_poker")
    state = game.new_initial_state()
    if not state.is_chance_node():
        LOGGER.debug("Initial state not a chance node; advancing until chance.")
        while not state.is_terminal() and not state.is_chance_node():
            legal = state.legal_actions()
            if not legal:
                break
            state.apply_action(legal[0])

    if not state.is_chance_node():
        raise RuntimeError("Failed to reach a chance node while verifying deck range.")

    outcomes, _ = zip(*state.chance_outcomes())
    observed = set(outcomes)
    expected = set(range(deck_size))
    if observed != expected:
        raise ValueError(
            "Unexpected chance action range: "
            f"observed {min(observed)}-{max(observed)} covering {len(observed)} actions, "
            f"expected {deck_size} actions spanning 0-{deck_size - 1}."
        )
    LOGGER.info(
        "Verified OpenSpiel chance actions span 0-%d (%d entries).",
        deck_size - 1,
        deck_size,
    )


def generate_hands(
    rng: random.Random,
    *,
    num_hands: int,
    cards_per_hand: int,
    deck_size: int,
) -> list[list[int]]:
    if not 0 < cards_per_hand <= deck_size:
        raise ValueError(
            f"cards_per_hand must be in [1, deck_size]; got cards_per_hand={cards_per_hand}, deck_size={deck_size}."
        )
    return [rng.sample(range(deck_size), cards_per_hand) for _ in range(num_hands)]


def generate_presets(
    *,
    seed: int,
    num_presets: int,
    num_hands: int,
    cards_per_hand: int,
    deck_size: int,
) -> Iterable[dict[str, object]]:
    rng = random.Random(seed)
    for preset_index in range(num_presets):
        hands = generate_hands(
            rng,
            num_hands=num_hands,
            cards_per_hand=cards_per_hand,
            deck_size=deck_size,
        )
        yield {
            "presetHands": hands,
            "seed": seed,
            "presetIndex": preset_index,
            "numHands": num_hands,
            "cardsPerHand": cards_per_hand,
            "deckSize": deck_size,
        }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} already exists. Pass --force to overwrite.")

    if not args.skip_verify:
        confirm_chance_action_range(args.deck_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as outfile:
        for preset in generate_presets(
            seed=args.seed,
            num_presets=args.num_presets,
            num_hands=args.num_hands,
            cards_per_hand=args.cards_per_hand,
            deck_size=args.deck_size,
        ):
            outfile.write(json.dumps(preset))
            outfile.write("\n")
    LOGGER.info("Wrote %d preset hand group(s) to %s", args.num_presets, args.output.resolve())


if __name__ == "__main__":
    main()
