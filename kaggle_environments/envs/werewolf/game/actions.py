from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional, Tuple

from pydantic import Field, create_model, field_validator

from .base import BaseAction, BaseState, PlayerID
from .consts import EventName, PerceivedThreatLevel, Phase
from .records import DoctorHealActionDataEntry, SeerInspectActionDataEntry

ACTION_EVENT_MAP = {}


def register_event(event_name: EventName):
    """A class decorator to register an EventName for an Action class."""

    def decorator(cls):
        ACTION_EVENT_MAP[cls.__name__] = event_name
        setattr(cls, "event_name", event_name)
        return cls

    return decorator


_REPLACEMENT_MAP = {
    # 'kill' variations
    "kill": "eliminate",
    "kills": "eliminates",
    "killed": "eliminated",
    "killing": "eliminating",
    "killer": "eliminator",
    # 'lynch' variations
    "lynch": "exile",
    "lynches": "exiles",
    "lynched": "exiled",
    "lynching": "exiling",
    # 'mislynch' variations
    "mislynch": "mis-exile",
    "mislynches": "mis-exiles",
    "mislynched": "mis-exiled",
    "mislynching": "mis-exiling",
    # 'murder' variations
    "murder": "remove",
    "murders": "removes",
    "murdered": "removed",
    "murdering": "removing",
    "murderer": "remover",
}

_CENSOR_PATTERN = re.compile(r"\b(" + "|".join(_REPLACEMENT_MAP.keys()) + r")\b", re.IGNORECASE)


# Create a single, case-insensitive regex pattern from all map keys.
def replacer(match):
    """
    Finds the correct replacement and applies case based on a specific heuristic.
    """
    original_word = match.group(0)
    replacement = _REPLACEMENT_MAP[original_word.lower()]

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


def filter_language(text):
    """Remove inappropriate/violent language."""
    return _CENSOR_PATTERN.sub(replacer, text)


# ------------------------------------------------------------------ #
class Action(BaseAction):
    """Root of the discriminated-union tree."""

    day: int
    phase: Phase
    actor_id: PlayerID
    reasoning: Optional[str] = Field(
        default=None,
        max_length=1000000,
        description="The self monologue that illustrate how you arrived at the action. "
        "It will be invisible to other players.",
    )

    perceived_threat_level: PerceivedThreatLevel = Field(
        default=PerceivedThreatLevel.SAFE,
        description="The self perceived threat level you are currently experiencing from other players. "
        "The assessment will be invisible to other players.",
    )
    error: Optional[str] = None
    raw_prompt: Optional[str] = None
    raw_completion: Optional[str] = None
    cost: Optional[float] = Field(default=None, description="The cost of generating this action.")
    prompt_tokens: Optional[int] = Field(default=None, description="The prompt token usage for generating this action.")
    completion_tokens: Optional[int] = Field(
        default=None, description="The completion token usage for generating this action."
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def filter_reasoning(cls, v):
        if v is None:
            return v
        return filter_language(v)

    def serialize(self):
        return {"action_type": self.__class__.__name__, "kwargs": self.model_dump()}

    @classmethod
    def schema_for_player(cls, fields: Tuple = None, new_cls_name=None):
        """Many of the fields are for internal game record. This method is used to convert the response schema
        to a format friendly for players.
        """
        fields = fields or []
        if not new_cls_name:
            new_cls_name = cls.__name__ + "Data"
        field_definitions = {
            field: (
                cls.model_fields[field].annotation,
                # Pass the entire FieldInfo object, not just the default value
                cls.model_fields[field],
            )
            for field in fields
            if field in cls.model_fields
        }
        sub_cls = create_model(new_cls_name, **field_definitions)
        subset_schema = sub_cls.model_json_schema()
        return subset_schema

    @property
    def action_field(self) -> Optional[str]:
        return None

    def push_event(self, state: BaseState):
        # The following is just for internal record keeping.
        data = self.model_dump()
        state.push_event(
            description=f"Player {self.actor_id}, you submitted {data}",
            event_name=ACTION_EVENT_MAP[self.__class__.__name__],
            public=False,
            visible_to=[],
            data=data,
        )


# ——— Mix-in for actions that need a target ------------------------ #
class TargetedAction(Action):
    target_id: PlayerID = Field(description="The target player's id.")

    @classmethod
    @lru_cache(maxsize=10)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ["perceived_threat_level", "reasoning", "target_id"]
        return super(TargetedAction, cls).schema_for_player(fields, new_cls_name)

    @property
    def action_field(self):
        return "target_id"


# ——— Concrete leaf classes --------------------------------------- #
@register_event(EventName.HEAL_ACTION)
class HealAction(TargetedAction):
    def push_event(self, state: BaseState):
        action_data = DoctorHealActionDataEntry(
            actor_id=self.actor_id,
            target_id=self.target_id,
            reasoning=self.reasoning,
            perceived_threat_level=self.perceived_threat_level,
            action=self,
        )
        state.push_event(
            description=f"Player {self.actor_id}, you chose to heal player {self.target_id}.",
            event_name=EventName.HEAL_ACTION,
            public=False,
            visible_to=[self.actor_id],
            data=action_data,
        )


@register_event(EventName.INSPECT_ACTION)
class InspectAction(TargetedAction):
    def push_event(self, state: BaseState):
        action_data = SeerInspectActionDataEntry(
            actor_id=self.actor_id,
            target_id=self.target_id,
            reasoning=self.reasoning,
            perceived_threat_level=self.perceived_threat_level,
            action=self,
        )
        state.push_event(
            description=f"Player {self.actor_id}, you chose to inspect player {self.target_id}.",
            event_name=EventName.INSPECT_ACTION,
            public=False,
            visible_to=[self.actor_id],
            data=action_data,
        )


@register_event(EventName.VOTE_ACTION)
class VoteAction(TargetedAction):
    pass


@register_event(EventName.ELIMINATE_PROPOSAL_ACTION)
class EliminateProposalAction(VoteAction):
    pass


@register_event(EventName.DISCUSSION)
class ChatAction(Action):
    message: str = Field(default="", max_length=1000000)

    @field_validator("message", mode="before")
    @classmethod
    def filter_message(cls, v):
        return filter_language(v)

    @classmethod
    @lru_cache(maxsize=10)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ["perceived_threat_level", "reasoning", "message"]
        return super(ChatAction, cls).schema_for_player(fields, new_cls_name)

    @property
    def action_field(self):
        return "message"


@register_event(EventName.NOOP_ACTION)
class NoOpAction(Action):
    pass


# ------------------------------------------------------------ #
@register_event(EventName.BID_ACTION)
class BidAction(Action):
    """
    An amount the actor is willing to pay this round.
    Currency unit can be generic 'chips' or role-specific.
    """

    amount: int = Field(ge=0)

    @classmethod
    @lru_cache(maxsize=10)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ["perceived_threat_level", "reasoning", "amount"]
        return super(BidAction, cls).schema_for_player(fields, new_cls_name)

    @property
    def action_field(self):
        return "amount"


ACTIONS = [EliminateProposalAction, HealAction, InspectAction, VoteAction, ChatAction, BidAction, NoOpAction]

ACTION_REGISTRY = {action.__name__: action for action in ACTIONS}


def create_action(serialized):
    return ACTION_REGISTRY[serialized["action_type"]](**serialized.get("kwargs", {}))
