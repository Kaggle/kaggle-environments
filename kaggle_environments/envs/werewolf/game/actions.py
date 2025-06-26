from __future__ import annotations
from enum import Enum
from typing import Literal, Annotated, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ActionKind(str, Enum):
    VOTE = "vote"
    ELIMINATE = "eliminate"
    HEAL = "heal"
    INSPECT = "inspect"
    CHAT = "chat"
    NOOP = "noop"


# ------------------------------------------------------------------ #
class Action(BaseModel):
    """Root of the discriminated-union tree."""
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)
    actor_id: str
    kind: ActionKind
    reasoning: Optional[str] = Field(default=None, max_length=4096)


# ——— Mix-in for actions that need a target ------------------------ #
class TargetedAction(Action):
    target_id: str

    @model_validator(mode="after")
    def check_target(self):
        if self.target_id == self.actor_id:
            raise ValueError("target_id cannot be the actor himself")
        return self


# ——— Concrete leaf classes --------------------------------------- #
class EliminateAction(TargetedAction):
    kind: Literal[ActionKind.ELIMINATE] = Field(default=ActionKind.ELIMINATE,
                                                serialization_alias="type")


class EliminateProposalAction(TargetedAction):
    """
    A werewolf's *suggestion* for tonight’s victim to be eliminated.
    These actions never reach the public log.
    """
    kind: Literal["eliminate_proposal"] = "eliminate_proposal"


class HealAction(TargetedAction):
    kind: Literal[ActionKind.HEAL] = ActionKind.HEAL


class InspectAction(TargetedAction):
    kind: Literal[ActionKind.INSPECT] = ActionKind.INSPECT


class VoteAction(TargetedAction):
    kind: Literal[ActionKind.VOTE] = ActionKind.VOTE
    # -1 can represent “abstain”; validation in VotingProtocol for exile votes


class ChatAction(Action):
    kind: Literal[ActionKind.CHAT] = ActionKind.CHAT
    message: str = Field(min_length=1, max_length=280)


class NoOpAction(Action):
    kind: Literal[ActionKind.NOOP] = ActionKind.NOOP


# ------------------------------------------------------------ #
class BidAction(Action):
    """
    An amount the actor is willing to pay this round.
    Currency unit can be generic 'chips' or role-specific.
    """
    kind: Literal["bid"] = "bid"  # type: ignore
    amount: int = Field(ge=0)


ACTION_REGISTRY = {
    ActionKind.ELIMINATE: EliminateAction,
    ActionKind.HEAL: HealAction,
    ActionKind.INSPECT: InspectAction,
    ActionKind.VOTE: VoteAction,
    ActionKind.CHAT: ChatAction,
    ActionKind.NOOP: NoOpAction,
}


def create_action(kind: ActionKind, **kwargs) -> Action:
    return ACTION_REGISTRY[kind](**kwargs)
