import requests

from typing import *

base_url = "https://www.kaggle.com/requests/EpisodeService/"
get_url = base_url + "GetEpisodeReplay"
list_url = base_url + "ListEpisodes"


def get_episode_replay(episode_id: int):
    body = {
        "EpisodeId": episode_id
    }

    response = requests.post(get_url, json=body)
    return response.json()


def list_episodes(episode_ids: List[int]):
    return __list_episodes({
        "Ids": episode_ids
    })


def list_episodes_for_team(team_id: int):
    return __list_episodes({
        "TeamId": team_id
    })


def list_episodes_for_submission(submission_id: int):
    return __list_episodes({
        "SubmissionId": submission_id
    })


def __list_episodes(body):
    response = requests.post(list_url, json=body)
    return response.json()
