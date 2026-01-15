from typing import Any

import requests

base_url: str = "https://www.kaggle.com/requests/EpisodeService/"
get_url: str = base_url + "GetEpisodeReplay"
list_url: str = base_url + "ListEpisodes"


def get_episode_replay(episode_id: int) -> dict[str, Any]:
    body = {"EpisodeId": episode_id}

    response = requests.post(get_url, json=body)
    return response.json()


def list_episodes(episode_ids: list[int]) -> dict[str, Any]:
    return __list_episodes({"Ids": episode_ids})


def list_episodes_for_team(team_id: int) -> dict[str, Any]:
    return __list_episodes({"TeamId": team_id})


def list_episodes_for_submission(submission_id: int) -> dict[str, Any]:
    return __list_episodes({"SubmissionId": submission_id})


def __list_episodes(body: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(list_url, json=body)
    return response.json()
