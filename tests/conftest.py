"""
Pytest configuration and fixtures for kaggle-environments tests.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "multicontainer: marks tests requiring multi-container setup")


@pytest.fixture(scope="session")
def env_list():
    """Fixture providing list of available environments."""
    import kaggle_environments

    return list(kaggle_environments.environments.keys())


@pytest.fixture
def rps_env():
    """Fixture providing a fresh RPS environment."""
    from kaggle_environments import make

    return make("rps", configuration={"episodeSteps": 10})


@pytest.fixture
def connectx_env():
    """Fixture providing a fresh ConnectX environment."""
    from kaggle_environments import make

    return make("connectx")
