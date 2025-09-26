from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Protocol, Type

from pydantic import BaseModel, StringConstraints

from .consts import EVENT_HANDLER_FOR_ATTR_NAME, MODERATOR_ID, EventName

# The ID regex supports Unicode letters (\p{L}), numbers (\p{N}) and common symbol for ID.
ROBUST_ID_REGEX = r"^[\p{L}\p{N} _.-]+$"

PlayerID = Annotated[str, StringConstraints(pattern=ROBUST_ID_REGEX, min_length=1, max_length=128)]


class BasePlayer(BaseModel, ABC):
    id: PlayerID
    """The unique id of the player. Also, how the player is referred to in the game."""

    alive: bool = True

    @abstractmethod
    def set_role_state(self, key, value):
        """Set role related state, which is a dict."""

    @abstractmethod
    def get_role_state(self, key, default=None):
        """Get role related state."""


class BaseAction(BaseModel):
    pass


class BaseState(BaseModel):
    @abstractmethod
    def push_event(
        self,
        description: str,
        event_name: EventName,
        public: bool,
        visible_to: Optional[List[PlayerID]] = None,
        data: Any = None,
        source=MODERATOR_ID,
    ):
        """Publish an event."""


class BaseEvent(BaseModel):
    event_name: EventName


class BaseModerator(ABC):
    @abstractmethod
    def advance(self, player_actions: Dict[PlayerID, BaseAction]):
        """Move one Kaggle environment step further. This is to be used within Kaggle 'interpreter'."""

    @abstractmethod
    def request_action(
        self,
        action_cls: Type[BaseAction],
        player_id: PlayerID,
        prompt: str,
        data=None,
        event_name=EventName.MODERATOR_ANNOUNCEMENT,
    ):
        """This can be used by event handler to request action from a player."""

    @abstractmethod
    def record_night_save(self, doctor_id: str, target_id: str):
        """To be used by a special Role to perform night save. This is implemented in moderator level, since
        coordinating between safe and night elimination is cross role activity.
        """

    @property
    @abstractmethod
    def state(self) -> BaseState:
        """Providing current state of the game, including player info, event messaging and caching."""


def on_event(event_type: EventName):
    def decorator(func):
        setattr(func, EVENT_HANDLER_FOR_ATTR_NAME, event_type)
        return func

    return decorator


class EventHandler(Protocol):
    """A callable triggered by an event."""

    def __call__(self, event: BaseEvent) -> Any:
        pass


class RoleEventHandler(Protocol):
    """A role specific event handler."""

    def __call__(self, me: BasePlayer, moderator: BaseModerator, event: BaseEvent) -> Any:
        pass


class BaseRole(BaseModel, ABC):
    """Special abilities should be implemented as RoleEventHandler in each subclass of BaseRole, so that Moderator
    doesn't need to be overwhelmed by role specific logic.
    """

    def get_event_handlers(self) -> Dict[EventName, RoleEventHandler]:
        """Inspects the role instance and collects all methods decorated with @on_event"""
        handlers = {}
        for attr_name in dir(self):
            if not attr_name.startswith("__"):
                attr = getattr(self, attr_name)
                if callable(attr) and hasattr(attr, EVENT_HANDLER_FOR_ATTR_NAME):
                    event_type = getattr(attr, EVENT_HANDLER_FOR_ATTR_NAME)
                    handlers[event_type] = attr
        return handlers
