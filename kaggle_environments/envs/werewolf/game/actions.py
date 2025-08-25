from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .consts import PerceivedThreatLevel


def filter_language(text):
    """Remove inappropriate/violent language."""
    replacement_map = {
        # 'kill' variations
        'kill': 'eliminate',
        'kills': 'eliminates',
        'killed': 'eliminated',
        'killing': 'eliminating',
        'killer': 'eliminator',

        # 'lynch' variations
        'lynch': 'exile',
        'lynches': 'exiles',
        'lynched': 'exiled',
        'lynching': 'exiling',

        # 'murder' variations
        'murder': 'remove',
        'murders': 'removes',
        'murdered': 'removed',
        'murdering': 'removing',
        'murderer': 'remover'
    }

    # Create a single, case-insensitive regex pattern from all map keys.
    pattern = re.compile(r'\b(' + '|'.join(replacement_map.keys()) + r')\b', re.IGNORECASE)

    def replacer(match):
        """
        Finds the correct replacement and applies case based on a specific heuristic.
        """
        original_word = match.group(0)
        replacement = replacement_map[original_word.lower()]

        # Rule 1: Preserve ALL CAPS.
        if original_word.isupper():
            return replacement.upper()

        # Rule 2: Handle title-cased words with a more specific heuristic.
        if original_word.istitle():
            # Preserve title case if it's the first word of the string OR
            # if it's a form like "-ing" which can start a new clause.
            return replacement.title()

        # Rule 3: For all other cases (e.g., "Kill" mid-sentence), default to lowercase.
        return replacement.lower()

    return pattern.sub(replacer, text)


# ------------------------------------------------------------------ #
class Action(BaseModel):
    """Root of the discriminated-union tree."""
    day: int
    phase: str
    actor_id: str
    reasoning: Optional[str] = Field(
        default=None, max_length=4096,
        description="The self monologue that illustrate how you arrived at the action. "
                    "It will be invisible to other players.")

    perceived_threat_level: PerceivedThreatLevel = Field(
        default=PerceivedThreatLevel.SAFE,
        description="The self perceived threat level you are currently experiencing from other players. "
                    "The assessment will be invisible to other players."
    )

    @field_validator('reasoning', mode='before')
    @classmethod
    def filter_reasoning(cls, v):
        if v is None:
            return v
        return filter_language(v)

    def serialize(self):
        return {'action_type': self.__class__.__name__, 'kwargs': self.model_dump()}


# ——— Mix-in for actions that need a target ------------------------ #
class TargetedAction(Action):
    target_id: str = Field(description="The target player's id.")


# ——— Concrete leaf classes --------------------------------------- #
class HealAction(TargetedAction):
    pass


class InspectAction(TargetedAction):
    pass


class VoteAction(TargetedAction):
    pass


class EliminateProposalAction(VoteAction):
    pass


class ChatAction(Action):
    message: str = Field(default="", max_length=4096)

    @field_validator('message', mode='before')
    @classmethod
    def filter_message(cls, v):
        return filter_language(v)


class NoOpAction(Action):
    pass


# ------------------------------------------------------------ #
class BidAction(Action):
    """
    An amount the actor is willing to pay this round.
    Currency unit can be generic 'chips' or role-specific.
    """
    amount: int = Field(ge=0)


ACTIONS = [
    EliminateProposalAction,
    HealAction,
    InspectAction,
    VoteAction,
    ChatAction,
    BidAction,
    NoOpAction
]

ACTION_REGISTRY = {
    action.__name__: action for action in ACTIONS
}

def create_action(serialized):
    return ACTION_REGISTRY[serialized['action_type']](**serialized.get('kwargs', {}))