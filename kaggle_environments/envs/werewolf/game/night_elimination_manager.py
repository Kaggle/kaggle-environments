from typing import Dict, List, Optional

from .base import PlayerID
from .consts import RevealLevel, Team
from .records import DoctorSaveDataEntry, EventName, WerewolfNightEliminationDataEntry
from .states import GameState


class NightEliminationManager:
    """
    Manages the state and resolution of nighttime eliminations.
    """

    def __init__(self, state: GameState, reveal_level: RevealLevel = RevealLevel.ROLE):
        self._state = state
        self._reveal_level = reveal_level
        self._saves: Dict[PlayerID, List[PlayerID]] = {}  # Key: target_id, Value: [doctor_id]

    def reset(self):
        """Clears all recorded actions for the start of a new night."""
        self._saves.clear()

    def record_save(self, doctor_id: PlayerID, target_id: PlayerID):
        """Records a save action from a Doctor."""
        self._saves.setdefault(target_id, []).append(doctor_id)

    def resolve_elimination(self, werewolf_target_id: Optional[PlayerID]):
        """
        Resolves the werewolf attack against any saves, eliminates a player
        if necessary, and pushes the resulting events to the game state.
        """
        if not werewolf_target_id:
            self._state.push_event(
                description="Last night, the werewolves did not reach a consensus (or no valid target was chosen)."
                " No one was eliminated by werewolves.",
                event_name=EventName.MODERATOR_ANNOUNCEMENT,
                public=False,
                visible_to=[player.id for player in self._state.get_players_by_team(Team.WEREWOLVES)],
            )
            self._state.push_event(
                description="Last night, No one was eliminated.",
                event_name=EventName.MODERATOR_ANNOUNCEMENT,
                public=True,
            )
            return

        target_player = self._state.get_player_by_id(werewolf_target_id)
        if not target_player:
            self._state.push_event(
                description=f'Last night, werewolves targeted player "{werewolf_target_id}", but this player '
                f"could not be found. No one was eliminated by werewolves.",
                event_name=EventName.ERROR,
                public=False,
                visible_to=[player.id for player in self._state.get_players_by_team(Team.WEREWOLVES)],
            )
            self._state.push_event(
                description="Last night, no one was eliminated.",
                event_name=EventName.MODERATOR_ANNOUNCEMENT,
                public=True,
            )
            return

        if werewolf_target_id in self._saves:
            # The player was saved.
            saving_doctor_ids = self._saves[werewolf_target_id]
            save_data = DoctorSaveDataEntry(saved_player_id=werewolf_target_id)
            self._state.push_event(
                description=f'Your heal on player "{werewolf_target_id}" was successful!',
                event_name=EventName.HEAL_RESULT,
                public=False,
                data=save_data,
                visible_to=saving_doctor_ids,
            )
            self._state.push_event(
                description="Last night, no one was eliminated.",
                event_name=EventName.MODERATOR_ANNOUNCEMENT,
                public=True,
            )
        else:
            # The player is eliminated.
            original_role_name = target_player.role.name
            self._state.eliminate_player(werewolf_target_id)

            team = None
            role = None
            descriptions = [f'Last night, player "{werewolf_target_id}" was eliminated by werewolves.']
            if self._reveal_level == RevealLevel.ROLE:
                team = target_player.role.team
                role = target_player.role.name
                descriptions.append(f'Their role was a "{original_role_name}".')
            elif self._reveal_level == RevealLevel.TEAM:
                team = target_player.role.team
                descriptions.append(f'Their team was "{team}".')

            data = WerewolfNightEliminationDataEntry(
                eliminated_player_id=werewolf_target_id,
                eliminated_player_role_name=role,
                eliminated_player_team_name=team,
            )
            description = " ".join(descriptions)
            self._state.push_event(description=description, event_name=EventName.ELIMINATION, public=True, data=data)
